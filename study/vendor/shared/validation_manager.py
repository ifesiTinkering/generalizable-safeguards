from copy import copy,deepcopy
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from cohort_budget import CohortMeter
from protocol import save
from async_validation_engine import evaluate_snapshot,apply_result
from judge_factory import make_judge
from evidence_judge_fallback import ReplaceRow
class ValidationManager:
 def __init__(self,root,ledger,grants,bound,namespace,workers=8):
  self.root,self.ledger,self.grants,self.bound,self.namespace=root,ledger,grants,bound,namespace
  self.pool=ThreadPoolExecutor(max_workers=workers);self.pending={};self.pre_admitted={}
  self.samples=ThreadPoolExecutor(max_workers=16);self.judges=ThreadPoolExecutor(max_workers=8)
 def enqueue(self,engine):
  update=engine.state['updates'];job=engine.job['job_id'];key=f'{self.namespace}:validation:{job}:{update}'
  if update in self.pending.get(job,{}):raise RuntimeError('Duplicate validation enqueue')

  meter=CohortMeter(self.ledger,key,self.bound+Decimal('.01'),self.root/'validation_cohorts'/f'{job}_{update}.json')
  pair={k:engine.state[k]for k in ('checkpoint','sampling_path')}
  snapshot={'update':update,'pair':pair,'state':deepcopy(engine.state)}
  engine.state.setdefault('pending_validation_pairs',{})[str(update)]=pair
  save(engine.path,engine.state) # durable pin precedes submission/pruning
  save(self.root/'validation_queue'/f'{job}_{update}.json',dict(snapshot,status='enqueued'))
  def work():
   clone=copy(engine);clone.judge=make_judge(engine.judge.root,meter,self.grants)
   clone.sample_pool=self.samples;clone.judge_pool=self.judges
   try:return evaluate_snapshot(clone,snapshot,meter)
   except ReplaceRow as exc:
    result={'complete':False,'metrics':{'complete':False,'incomplete_reasons':[str(exc)]},'snapshot':snapshot}
    save(self.root/'validation_queue'/f'{job}_{update}_incomplete.json',result)
    return result
  self.pending.setdefault(job,{})[update]=self.pool.submit(work)
 def poll(self,engine):
  pending=self.pending.get(engine.job['job_id'],{})
  while pending:
   update=min(pending);future=pending[update]
   if not future.done():break
   result=future.result();apply_result(engine,result)
   save(self.root/'validation_queue'/f'{engine.job["job_id"]}_{update}.json',dict(result['snapshot'],status='applied',complete=result['complete']))
   del pending[update] # only AFTER durable decision and metric pins
 def has_pending(self,engine):return bool(self.pending.get(engine.job['job_id']))
 def close(self):
  self.pool.shutdown(wait=True);self.samples.shutdown(wait=True);self.judges.shutdown(wait=True)
