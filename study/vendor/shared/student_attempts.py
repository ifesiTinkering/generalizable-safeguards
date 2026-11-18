"""Three physical attempts per requested student completion, never replay unknown."""
import time
from evidence_judge_fallback import ReplaceRow
ELIGIBLE={'APITimeoutError','TimeoutError','ReadTimeout','ConnectTimeout','ConnectError','RemoteProtocolError','UncertainOperation'}
def sample_three(backend,prompt,*,seed,max_tokens,request_id):
 for attempt in range(1,4):
  physical=request_id+f':physical_attempt:{attempt}'
  try:
   result=backend.sample(prompt,seed=seed,max_tokens=max_tokens,request_id=physical)
   return dict(result,physical_attempt=attempt,physical_request_id=physical)
  except BaseException as exc:
   transient=type(exc).__name__ in ELIGIBLE or getattr(exc,'status_code',None)in(429,500,502,503,504)
   if not transient:raise
   if attempt<3:time.sleep(min(2**attempt,8))
 raise ReplaceRow('Student completion exhausted three physical attempts; same-domain replacement required')
