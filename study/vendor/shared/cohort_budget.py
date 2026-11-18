"""Reserve an entire matched round before any arm advances. Offline-testable."""
from decimal import Decimal
from functools import wraps
from threading import RLock
from pathlib import Path
from protocol import read, save

def locked(method):
    @wraps(method)
    def guarded(self,*args,**kwargs):
        with self._lock:
            return method(self,*args,**kwargs)
    return guarded

class CohortMeter:
    def __init__(self, ledger, key, bound, journal):
        self._lock = RLock()
        self.ledger, self.key, self.bound = ledger, key, Decimal(str(bound))
        self.path = Path(journal)
        if self.path.exists():
            raise RuntimeError('Cohort already attempted; reconcile rather than replay')
        if not ledger.reserve(key, self.bound, {'kind':'matched_cohort', 'bound_usd':str(self.bound)}):
            raise RuntimeError('Cohort reservation already exists')
        if not ledger.dispatch(key):
            raise RuntimeError('Cohort dispatch already exists')
        self.calls = {}; self._save('active')

    @locked
    def _save(self, status='active'):
        save(self.path, {'cohort':self.key, 'bound_usd':str(self.bound), 'status':status, 'calls':self.calls})

    @locked
    def reserve(self, key, bound, spec):
        amount = Decimal(str(bound))
        if not amount.is_finite() or amount < 0:
            raise ValueError('Invalid reservation')
        if key in self.calls:
            return False
        exposure = sum(Decimal(v.get('actual',v['reserved'])) for v in self.calls.values() if v['status']!='cancelled')
        if exposure+amount > self.bound:
            raise RuntimeError('Subrequest exceeds pre-reserved matched-round bound')
        self.calls[key] = {'reserved':str(amount), 'status':'reserved', 'spec':spec}
        self._save(); return True

    @locked
    def dispatch(self, key):
        item = self.calls[key]
        if item['status']!='reserved': return False
        item['status']='dispatched'; self._save(); return True

    @locked
    def settle(self, key, actual):
        item=self.calls[key]; amount=Decimal(str(actual))
        if item['status']!='dispatched' or not amount.is_finite() or amount<0:
            raise ValueError('Invalid settlement')
        if amount>Decimal(item['reserved']):
            item['actual']=str(amount); item['status']='bound_breach'; self._save('frozen')
            # Preserve/charge all other uncertain reservations and freeze at the
            # parent level if this also exceeds its full-round bound.
            self.ledger.freeze('A subrequest exceeded its bound; full cohort reservation retained for reconciliation')
            raise RuntimeError('Provider usage exceeded admitted bound')
        item.update(actual=str(amount),status='settled'); self._save()

    @locked
    def cancel_undispatched(self,key):
        if self.calls[key]['status']!='reserved': raise RuntimeError('Cannot cancel dispatched request')
        self.calls[key]['status']='cancelled'; self._save()

    @locked
    def close(self):
        if any(v['status'] not in ('settled','cancelled') for v in self.calls.values()):
            raise RuntimeError('Unknown request outcome: retain full cohort reservation')
        actual=sum((Decimal(v['actual']) for v in self.calls.values() if v['status']=='settled'),Decimal(0))
        self.ledger.settle(self.key,actual); self._save('settled'); return actual
