"""Retry only entry to a frozen ledger transaction before its body can execute."""
from contextlib import contextmanager
import hashlib,random,time
LOCK_TIMEOUT='Budget admission lock timeout; no operation authorized'

def entry_retry(original, *, attempts=6, sleep=time.sleep, delay=lambda n:min(60,2**n)+random.uniform(0,1), on_wait=lambda event:None):
 if not 1<=attempts<=10:raise ValueError('Bounded acquisition attempts required')
 @contextmanager
 def wrapped(self):
  for attempt in range(1,attempts+1):
   context=original(self)
   try:
    connection=context.__enter__()
    break
   except TimeoutError as error:
    if str(error)!=LOCK_TIMEOUT or attempt==attempts:raise
    # The original source is hash-bound: this exact pre-yield exception occurs
    # before sqlite connect/BEGIN. No transaction body or provider call reruns.
    seconds=delay(attempt)
    if not 0<=seconds<=61:raise ValueError('Backoff out of bounds')
    on_wait({'kind':'pre_transaction_lock_wait','attempt':attempt,'maximum_attempts':attempts,'sleep_seconds':seconds})
    sleep(seconds)
  try:
   yield connection
  except BaseException as error:
   # Exceptions after yield, including settlement/COMMIT uncertainty, never
   # return to the acquisition loop or repeat the caller's transaction body.
   if not context.__exit__(type(error),error,error.__traceback__):raise
  else:
   context.__exit__(None,None,None)
 return wrapped

def install(module, expected_source_sha256, **policy):
 actual=hashlib.sha256(open(module.__file__,'rb').read()).hexdigest()
 if actual!=expected_source_sha256:raise ValueError('Unreviewed ledger source')
 if getattr(module.Ledger,'_claim2_entry_retry_installed',False):raise ValueError('Already installed')
 original=module.Ledger._process_tx
 if module.GrantLedger._process_tx is not original:raise ValueError('Unexpected grant transaction alias')
 wrapped=entry_retry(original,**policy)
 module.Ledger._process_tx=wrapped
 module.GrantLedger._process_tx=wrapped
 module.Ledger._claim2_entry_retry_installed=True
 return {'ledger_source_sha256':actual,'retry_phase':'pre_transaction_entry_only','body_replay':False}
