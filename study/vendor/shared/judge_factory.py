from evidence_judge_fallback import RetryingEvidenceJudge
from judge_provider_adapter import MeteredJudgeAdapter,upper
from astra_contract import INSTRUCTIONS,SCHEMA
from astra_judge import canonical
MODELS={'astra':'gpt-6-astra','luna':'gpt-6-luna'}
def make_judge(root,meter,grant=None):
 return RetryingEvidenceJudge(root,{k:MeteredJudgeAdapter(v,meter,grant if k=='astra'else None)for k,v in MODELS.items()},MODELS)
def grading_upper(row):
 # Reserve a conservative payload bound: full context plus up to23k student
 # tokens, each no more than four UTF-8 bytes in the source JSON estimate.
 context_bytes=len(canonical(row).encode())+23000*16+65536
 body={'model':'gpt-6-astra','max_output_tokens':8192,'input':'x'*context_bytes,'instructions':INSTRUCTIONS,'schema':SCHEMA}
 astra=upper(body);body['model']='gpt-6-luna';luna=upper(body)
 return 3*(astra+luna)
