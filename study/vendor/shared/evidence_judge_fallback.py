"""Versioned six-attempt evidence grading, with explicit paid-attempt adapters.

Importing does no provider work. Each model adapter must enforce independently
verified access, rates and accounting. A physical attempt is never replayed.
"""
import hashlib,json,fcntl
from pathlib import Path
from astra_contract import INSTRUCTIONS,SCHEMA
from astra_judge import canonical,save
class ReplaceRow(RuntimeError):pass
class PolicyRefusal(ReplaceRow):pass
class EligibleJudgeFailure(RuntimeError):pass
POLICY_CODES={'cyber_policy','content_policy_violation','policy_violation','safety_violation'}

def decode_response(response,ids):
 error=response.get('error')or{}
 if error.get('code')in POLICY_CODES or any(part.get('type')=='refusal'for item in response.get('output',[])for part in item.get('content',[])):raise PolicyRefusal('Explicit provider policy refusal; no fallback or repeated denied request')
 if not response.get('usage'):raise EligibleJudgeFailure('Unknown grading usage/outcome')
 if response.get('status')!='completed':raise EligibleJudgeFailure('Incomplete grading')
 texts=[p['text']for item in response.get('output',[])if item.get('type')=='message'for p in item.get('content',[])if p.get('type')=='output_text']
 if len(texts)!=1:raise EligibleJudgeFailure('Missing structured grading')
 try:value=json.loads(texts[0])
 except (ValueError,TypeError)as exc:raise EligibleJudgeFailure('Malformed grading JSON')from exc
 rules=value.get('rules')if isinstance(value,dict)else None
 if not isinstance(rules,list)or len(rules)!=len(ids)or any(not isinstance(r,dict)for r in rules):raise EligibleJudgeFailure('Incomplete rule grading')
 if {r.get('rule_id')for r in rules}!=set(ids):raise EligibleJudgeFailure('Wrong grader rule IDs')
 for r in rules:
  if type(r.get('claim_supported'))is not bool or type(r.get('evidence_supported'))is not bool or not isinstance(r.get('reason'),str):raise EligibleJudgeFailure('Malformed grading fields')
  if r['evidence_supported']and not r['claim_supported']:raise EligibleJudgeFailure('Inconsistent grading fields')
 return rules

class RetryingEvidenceJudge:
 def __init__(self,root,adapters,model_ids):
  if set(adapters)!= {'astra','luna'}or set(model_ids)!=set(adapters):raise ValueError('Both independently audited model adapters required')
  if not all(a.verified_model_id==model_ids[k]for k,a in adapters.items()):raise ValueError('Model availability must be verified before dispatch')
  self.root=Path(root);self.adapters=adapters;self.model_ids=model_ids
 def __call__(self,context,p):
  candidate={k:p[k]for k in ('is_violation','violated_rule_ids','rule_assessments')}
  logical={'instructions':INSTRUCTIONS,'input':canonical({**context,'candidate':candidate}),'max_output_tokens':8192,'text':{'format':{'type':'json_schema','name':'per_rule_support','strict':True,'schema':SCHEMA}},'store':False,'service_tier':'default'}
  digest=hashlib.sha256(canonical(logical).encode()).hexdigest();folder=self.root/'judge_retry_cache'/digest;folder.mkdir(parents=True,exist_ok=True)
  context_hash=hashlib.sha256(canonical(context).encode()).hexdigest();blocked=self.root/'judge_unavailable_contexts'/context_hash
  if blocked.exists():raise PolicyRefusal('Previously denied context; skip directly')
  with(folder/'lock').open('a')as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if(folder/'success.json').exists():return json.loads((folder/'success.json').read_text())
   if(folder/'exhausted.json').exists():raise ReplaceRow('Six physical attempts already exhausted; no replay')
   attempts=[]
   for role in ('astra','luna'):
    adapter=self.adapters[role];model=self.model_ids[role]
    for index in range(1,4):
     physical=f'evidence-grade:{digest}:{model}:{index}';path=folder/f'{role}_{index}.json';body=dict(logical,model=model)
     if path.exists():
      record=json.loads(path.read_text());response=record.get('response')
      if response is None:
       attempts.append({'model':model,'attempt':index,'physical_id':physical,'outcome':'prior_unknown_preserved'})
       continue # authorized next physical attempt, NEVER repeat this one
     else:
      record={'physical_id':physical,'model':model,'attempt':index,'status':'prepared','request_sha256':hashlib.sha256(canonical(body).encode()).hexdigest()};save(path,record)
      if adapter.reserve(physical,body)is not True:raise RuntimeError('Attempt already reserved; reconcile without replay')
      if adapter.dispatch(physical)is not True:raise RuntimeError('Attempt already dispatched; never replay')
      record['status']='dispatched';save(path,record)
      try:response=adapter.invoke(body)
      except adapter.eligible_transport_errors as exc:
       record.update(status='unknown',error_type=type(exc).__name__,original_hold_preserved=True);save(path,record)
       attempts.append({'model':model,'attempt':index,'physical_id':physical,'outcome':'unknown_transport'})
       adapter.backoff(index)
       continue
      record.update(response=response,status='response_received');save(path,record)
     if response.get('usage') and not record.get('accounting_settled'):
      adapter.settle(physical,response)
      record['accounting_settled']=True;save(path,record)
     try:rules=decode_response(response,candidate['violated_rule_ids'])
     except PolicyRefusal:
      blocked.parent.mkdir(parents=True,exist_ok=True);save(blocked,{'physical_id':physical,'policy_refusal':True,'never_fallback':True});raise
     except EligibleJudgeFailure as exc:
      record.update(status='unusable',reason=str(exc));save(path,record)
      attempts.append({'model':model,'attempt':index,'physical_id':physical,'outcome':'unusable_grading'})
      continue
     record['status']='complete';save(path,record)
     attempts.append({'model':model,'attempt':index,'physical_id':physical,'outcome':'usable'})
     result={'rules':rules,'request_sha256':record['request_sha256'],'model':model,'usage':response['usage'],'judge_attempts':attempts,'judge_fallback_used':role=='luna','judge_funding_source':getattr(adapter,'funding_source','test_adapter'),'judge_attempt_count':len(attempts),'student_malformed_penalty':False}
     save(folder/'success.json',result);return result
   save(folder/'exhausted.json',{'attempts':attempts,'replacement_required':True,'domain_must_be_preserved':True,'unknown_holds_preserved':True})
   raise ReplaceRow('Three Astra and three Luna attempts unusable; replace row in same domain')
