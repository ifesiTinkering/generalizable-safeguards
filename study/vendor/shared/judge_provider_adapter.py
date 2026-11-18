"""Once-only Responses transport with separate grant/cash attempt accounting."""
import os,time,json
from decimal import Decimal
import httpx,sqlite3
from evidence_budget import BudgetExceeded
from evidence_judge_fallback import EligibleJudgeFailure
class RetryableServiceFailure(RuntimeError):pass
class ProviderAccessFailure(RuntimeError):pass
RATES={'gpt-6-astra':(Decimal('10'),Decimal('50')),'gpt-6-luna':(Decimal('.10'),Decimal('.50'))}
def upper(body):
 # UTF-8 byte count bounds tokenized text; explicit overhead includes framing.
 n=len(json.dumps(body,ensure_ascii=False).encode())+4096
 a,b=RATES[body['model']]
 return (n*a+body['max_output_tokens']*b)/1000000
class MeteredJudgeAdapter:
 eligible_transport_errors=(httpx.TransportError,RetryableServiceFailure)
 def __init__(self,model,meter,grant=None,post=None):
  self.verified_model_id=model;self.meter=meter;self.grant=grant;self.post=post or httpx.post
  self.funding={};self.funding_source='grant'if grant is not None else'cash'
 def reserve(self,key,body):
  funding='cash'
  if self.grant is not None:
   try:
    if not self.grant.reserve(key,{'request_sha256':key,'model':body['model'],'max_output_tokens':8192}):return False
    funding='grant'
   except BudgetExceeded:pass
   except sqlite3.IntegrityError as exc:
    if 'Grant capacity reserved for child allocation' not in str(exc):raise
  spec={'kind':'judge','model':body['model'],'funding':funding,'max_output_tokens':8192,'rates_input_output_per_million':[str(x)for x in RATES[body['model']]],'unknown_hold_preserved':True}
  self.funding[key]=funding
  return self.meter.reserve(key,max(Decimal('.000001'),upper(body)if funding=='cash'else Decimal(0)),spec)
 def dispatch(self,key):
  if self.funding[key]=='grant' and not self.grant.dispatch(key):return False
  return self.meter.dispatch(key)
 def invoke(self,body):
  response=self.post('https://api.openai.com/v1/responses',headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']},json=body,timeout=180,trust_env=False)
  try:value=response.json()
  except ValueError:value={'status':'incomplete','error':{'code':'non_json_grading'}}
  if not isinstance(value,dict):value={'status':'incomplete','error':{'code':'non_object_grading'}}
  code=(value.get('error')or{}).get('code')
  if response.status_code in (401,404)or code in ('insufficient_quota','model_not_found','invalid_api_key'):raise ProviderAccessFailure('Provider access/billing unavailable; no automatic fallback around account failure')
  if response.status_code>=500 or response.status_code==429:raise RetryableServiceFailure(f'Transient provider HTTP{response.status_code}')
  value['_http_status']=response.status_code
  return value
 def settle(self,key,response):
  usage=response['usage'];a,b=RATES[self.verified_model_id]
  amount=(Decimal(usage['input_tokens'])*a+Decimal(usage['output_tokens'])*b)/1000000
  funding=self.funding.get(key)or self.meter.calls.get(key,{}).get('spec',{}).get('funding')
  if funding not in ('grant','cash'):raise RuntimeError('Unknown attempt funding; reconcile original hold')
  if funding=='grant':self.grant.settle_usage(key,usage,amount)
  charge=Decimal(0)if funding=='grant'else amount
  prior=self.meter.calls.get(key)
  if prior is None:raise RuntimeError('Known response belongs to earlier meter; reconcile original accounting before reuse')
  if prior['status']=='settled':
   if Decimal(prior['actual'])!=charge:raise RuntimeError('Conflicting known settlement')
   return
  self.meter.settle(key,charge)
 def backoff(self,index):time.sleep(min(2**index,8))
