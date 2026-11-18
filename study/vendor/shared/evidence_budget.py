"""Isolated, fail-closed budget journal. No provider imports or calls.

A dispatched request with an unknown outcome retains its full reservation.
Duplicate IDs never authorize a second dispatch. New dispatch is frozen after a
bound breach. One local host must own the SQLite file; do not share it across
hosts or copy it to start a fresh budget. This module cannot enforce unmetered
provider calls or indefinite checkpoint storage, so transport and retention
must also obey the reviewed plan.
"""
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import fcntl
import json
import os
import sqlite3
import time
import threading
from collections import deque

# Serialize same-process ledger transactions in arrival order before flock.
# Network calls remain outside this queue. The cross-process timeout is unchanged.
_QUEUE_GUARD = threading.Lock()
_QUEUES = {}

@contextmanager
def _local_turn(path):
    with _QUEUE_GUARD:
        condition, queue = _QUEUES.setdefault(str(path), (threading.Condition(), deque()))
    token = object()
    with condition:
        queue.append(token)
        condition.wait_for(lambda: queue[0] is token)
    try:
        yield
    finally:
        with condition:
            assert queue.popleft() is token
            condition.notify_all()


STUDY_ID = 'claim2_evidence_ablations_20261111'
HARD_CAP_USD = Decimal('1000')

class BudgetExceeded(RuntimeError): pass
class BudgetFrozen(RuntimeError): pass

def micros(usd):
    value = Decimal(str(usd))
    if not value.is_finite() or value < 0:
        raise ValueError('Cost must be finite and nonnegative')
    return int((value * 1000000).to_integral_value(rounding=ROUND_CEILING))

class Ledger:
    def __init__(self, path, cap_usd=50, study_id=STUDY_ID, lock_timeout_seconds=30, *, cash_stop_enabled=True):
        if study_id != STUDY_ID:
            raise ValueError('This journal belongs only to the isolated Claim 2 study')
        self.cash_stop_enabled = cash_stop_enabled
        if type(cash_stop_enabled) is not bool:raise ValueError('Explicit cash policy required')
        cap = micros(cap_usd)
        if not 0 < cap <= micros(HARD_CAP_USD):
            raise ValueError('Study cap must be positive and cannot exceed $1000')
        path = Path(path).resolve()
        if STUDY_ID not in path.parts or path.name != 'budget.sqlite':
            raise ValueError('Budget path must be inside the isolated study and named budget.sqlite')
        if not 0 < lock_timeout_seconds <= 60:
            raise ValueError('Lock timeout must be between 0 and 60 seconds')
        self.path = path
        self.lock_timeout_seconds = lock_timeout_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = path.with_suffix('.lock')
        with self.tx() as c:
            c.execute('CREATE TABLE IF NOT EXISTS meta(id INTEGER PRIMARY KEY CHECK(id=1),study TEXT NOT NULL,cap INTEGER NOT NULL,frozen INTEGER NOT NULL DEFAULT 0)')
            c.execute('CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY,spec TEXT NOT NULL,reserved INTEGER NOT NULL,charged INTEGER NOT NULL,status TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,event TEXT NOT NULL,reason TEXT NOT NULL,unix_time REAL NOT NULL)')
            prior = c.execute('SELECT study,cap FROM meta WHERE id=1').fetchone()
            if prior is None:
                # Refuse adoption of a nonempty foreign or damaged journal.
                if c.execute('SELECT COUNT(*) FROM calls').fetchone()[0]:
                    raise ValueError('Unowned nonempty budget journal')
                c.execute('INSERT INTO meta(id,study,cap) VALUES(1,?,?)', (study_id, cap))
            elif prior[0] != study_id or (cash_stop_enabled and prior[1] != cap):
                raise ValueError('Existing study/cap mismatch; budget cannot be reset')

    @contextmanager
    def tx(self):
        with _local_turn(self.lock_path):
            with self._process_tx() as connection:
                yield connection

    @contextmanager
    def _process_tx(self):
        deadline = time.monotonic() + self.lock_timeout_seconds
        with self.lock_path.open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Budget admission lock timeout; no operation authorized')
                    time.sleep(.025)
            c = sqlite3.connect(str(self.path), timeout=self.lock_timeout_seconds, isolation_level=None)
            try:
                c.execute('PRAGMA synchronous=FULL')
                c.execute('BEGIN IMMEDIATE')
                yield c
                # Do not repeat a transaction body after an uncertain commit.
                c.execute('COMMIT')
            except BaseException:
                if c.in_transaction:
                    c.execute('ROLLBACK')
                raise
            finally:
                c.close()
                fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve_many(self, operations):
        """Atomically reserve [(id, upper_bound_usd, spec), ...].

        Return True only if every ID is new. All identical prior IDs return
        False (never dispatch them again); mixed new/prior requests fail closed.
        """
        rows = []
        for call_id, bound, spec in operations:
            if not isinstance(call_id, str) or not call_id or micros(bound) <= 0:
                raise ValueError('Every billable attempt needs an ID and positive bound')
            rows.append((call_id, json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False), micros(bound)))
        if not rows or len({r[0] for r in rows}) != len(rows):
            raise ValueError('Expected nonempty unique attempt IDs')
        with self.tx() as c:
            prior_count = 0
            for call_id, spec, amount in rows:
                prior = c.execute('SELECT spec,reserved FROM calls WHERE id=?', (call_id,)).fetchone()
                if prior is not None:
                    if prior != (spec, amount):
                        raise ValueError('Reused ID with changed operation')
                    prior_count += 1
            if prior_count:
                if prior_count != len(rows):
                    raise ValueError('Partially existing reservation group; reconcile without replay')
                return False
            cap, frozen = c.execute('SELECT cap,frozen FROM meta WHERE id=1').fetchone()
            if frozen:
                raise BudgetFrozen('Ledger frozen; no new calls allowed')
            used = c.execute('SELECT COALESCE(SUM(charged),0) FROM calls').fetchone()[0]
            if self.cash_stop_enabled and used + sum(r[2] for r in rows) > cap:
                raise BudgetExceeded('Cannot reserve full worst-case cost')
            c.executemany('INSERT INTO calls VALUES(?,?,?,?,?)', [(i,s,a,a,'reserved') for i,s,a in rows])
        return True

    def reserve(self, call_id, upper_bound_usd, spec):
        return self.reserve_many([(call_id, upper_bound_usd, spec)])

    def dispatch(self, call_id):
        with self.tx() as c:
            if c.execute('SELECT frozen FROM meta WHERE id=1').fetchone()[0]:
                raise BudgetFrozen('Ledger frozen')
            return c.execute("UPDATE calls SET status='dispatched' WHERE id=? AND status='reserved'", (call_id,)).rowcount == 1

    def settle(self, call_id, actual_usd):
        """Complete verified usage only; missing usage keeps the original bound."""
        actual = micros(actual_usd)
        breach = False
        with self.tx() as c:
            row = c.execute('SELECT reserved,charged,status FROM calls WHERE id=?', (call_id,)).fetchone()
            if row is None:
                raise KeyError(call_id)
            reserved, charged, status = row
            if status == 'settled':
                if actual != charged:
                    raise ValueError('Conflicting settlement')
                return
            if status != 'dispatched':
                raise ValueError('Only dispatched calls can settle')
            c.execute("UPDATE calls SET charged=?,status='settled' WHERE id=?", (actual, call_id))
            if actual > reserved:
                c.execute('UPDATE meta SET frozen=1 WHERE id=1')
                c.execute('INSERT INTO events(event,reason,unix_time) VALUES(?,?,?)',('freeze',f'Call {call_id} actual cost exceeded its reservation',time.time()))
                breach = True
        if breach:
            raise BudgetFrozen('Actual exceeded reservation; full charge retained and dispatch frozen')

    def cancel_undispatched(self, call_id):
        with self.tx() as c:
            changed = c.execute("UPDATE calls SET charged=0,status='cancelled' WHERE id=? AND status='reserved'", (call_id,)).rowcount
            if changed != 1:
                raise ValueError('Cannot release a dispatched, settled, cancelled or unknown attempt')

    def freeze(self,reason):
        """Stop new dispatch after a subrequest/accounting breach; retain charges."""
        if not isinstance(reason,str) or not reason.strip():
            raise ValueError('A freeze reason is required')
        with self.tx() as c:
            c.execute('UPDATE meta SET frozen=1 WHERE id=1')
            c.execute('INSERT INTO events(event,reason,unix_time) VALUES(?,?,?)',('freeze',reason,time.time()))

    def snapshot(self):
        with self.tx() as c:
            study, cap, frozen = c.execute('SELECT study,cap,frozen FROM meta WHERE id=1').fetchone()
            used = c.execute('SELECT COALESCE(SUM(charged),0) FROM calls').fetchone()[0]
            states = dict(c.execute('SELECT status,COUNT(*) FROM calls GROUP BY status').fetchall())
            reason = c.execute("SELECT reason FROM events WHERE event='freeze' ORDER BY id DESC LIMIT 1").fetchone()
        return dict(study=study,cap_microusd=cap,charged_or_reserved_microusd=used,remaining_microusd=cap-used,frozen=bool(frozen),states=states,freeze_reason=reason[0] if reason else None)

class GrantLedger:
    """Separate count-limited grant journal; never charges the cash budget.

    Cap applies to all reserved/dispatched/completed unique attempts. Missing
    usage remains unknown and consumes a call slot; it never authorizes retry.
    """
    tx = Ledger.tx
    _process_tx = Ledger._process_tx

    def __init__(self,path,max_calls=46440,study_id=STUDY_ID,lock_timeout_seconds=30):
        if study_id != STUDY_ID or type(max_calls) is not int or not 0 < max_calls <= 46440:
            raise ValueError('Invalid isolated grant study or reviewed request cap')
        self.path=Path(path).resolve()
        if STUDY_ID not in self.path.parts or self.path.name != 'grant_usage.sqlite':
            raise ValueError('Grant path must be isolated and named grant_usage.sqlite')
        if not 0 < lock_timeout_seconds <= 60:
            raise ValueError('Invalid lock timeout')
        self.lock_timeout_seconds=lock_timeout_seconds
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock_path=self.path.with_suffix('.lock')
        with self.tx() as c:
            c.execute('CREATE TABLE IF NOT EXISTS grant_meta(id INTEGER PRIMARY KEY CHECK(id=1),study TEXT NOT NULL,max_calls INTEGER NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS grant_calls(id TEXT PRIMARY KEY,spec TEXT NOT NULL,status TEXT NOT NULL,usage TEXT,economic_microusd INTEGER)')
            prior=c.execute('SELECT study,max_calls FROM grant_meta WHERE id=1').fetchone()
            if prior is None:
                if c.execute('SELECT COUNT(*) FROM grant_calls').fetchone()[0]:
                    raise ValueError('Unowned grant journal')
                c.execute('INSERT INTO grant_meta VALUES(1,?,?)',(study_id,max_calls))
            elif prior != (study_id,max_calls):
                raise ValueError('Existing grant cap cannot be reset or changed')

    def reserve(self,call_id,spec):
        if not isinstance(call_id,str) or not call_id:
            raise ValueError('Grant request needs a stable ID')
        encoded=json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False)
        with self.tx() as c:
            prior=c.execute('SELECT spec FROM grant_calls WHERE id=?',(call_id,)).fetchone()
            if prior is not None:
                if prior[0]!=encoded:
                    raise ValueError('Reused grant ID with different request')
                return False
            cap=c.execute('SELECT max_calls FROM grant_meta WHERE id=1').fetchone()[0]
            used=c.execute("SELECT COUNT(*) FROM grant_calls WHERE status!='cancelled'").fetchone()[0]
            if used>=cap:
                raise BudgetExceeded('Grant request-count cap reached; no further requests')
            c.execute("INSERT INTO grant_calls(id,spec,status) VALUES(?,?,'reserved')",(call_id,encoded))
        return True

    def dispatch(self,call_id):
        with self.tx() as c:
            return c.execute("UPDATE grant_calls SET status='dispatched' WHERE id=? AND status='reserved'",(call_id,)).rowcount==1

    def settle_usage(self,call_id,usage,economic_usd):
        if not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0 for k in ('input_tokens','output_tokens')):
            raise ValueError('Complete provider input/output token usage required')
        encoded=json.dumps(usage,sort_keys=True,separators=(',',':'),allow_nan=False)
        amount=micros(economic_usd)
        with self.tx() as c:
            row=c.execute('SELECT status,usage,economic_microusd FROM grant_calls WHERE id=?',(call_id,)).fetchone()
            if row is None:
                raise KeyError(call_id)
            if row[0]=='settled':
                if row[1:]!=(encoded,amount):
                    raise ValueError('Conflicting grant settlement')
                return
            if row[0]!='dispatched':
                raise ValueError('Only dispatched grant requests can settle')
            c.execute("UPDATE grant_calls SET status='settled',usage=?,economic_microusd=? WHERE id=?",(encoded,amount,call_id))

    def cancel_undispatched(self,call_id):
        with self.tx() as c:
            if c.execute("UPDATE grant_calls SET status='cancelled' WHERE id=? AND status='reserved'",(call_id,)).rowcount!=1:
                raise ValueError('Cannot cancel dispatched or unknown grant request')

    def snapshot(self):
        with self.tx() as c:
            study,cap=c.execute('SELECT study,max_calls FROM grant_meta WHERE id=1').fetchone()
            states=dict(c.execute('SELECT status,COUNT(*) FROM grant_calls GROUP BY status').fetchall())
            used=c.execute("SELECT COUNT(*) FROM grant_calls WHERE status!='cancelled'").fetchone()[0]
            value=c.execute('SELECT COALESCE(SUM(economic_microusd),0) FROM grant_calls').fetchone()[0]
            usage=[json.loads(r[0]) for r in c.execute("SELECT usage FROM grant_calls WHERE status='settled'")]
        return dict(study=study,max_calls=cap,reserved_or_dispatched_calls=used,remaining_calls=cap-used,states=states,cash_usd=0,known_economic_microusd=value,unknown_usage_calls=states.get('dispatched',0),known_input_tokens=sum(r['input_tokens'] for r in usage),known_output_tokens=sum(r['output_tokens'] for r in usage))
