"""Pure native Harmony stage conditioning; sampled tokens are never rewritten."""
import hashlib,json,math
CLOSE='<|end|>';FINAL='<|start|>assistant<|channel|>final<|message|><answer>\n'
def token_hash(x):return hashlib.sha256(json.dumps(list(x)).encode()).hexdigest()
def assemble(base,result,tok,context=65536):
 tokens=list(result['tokens']);logs=list(result['logprobs']);close=tok.encode(CLOSE,add_special_tokens=False);prefix=tok.encode('<|start|>assistant<|channel|>analysis<|message|>',add_special_tokens=False)
 if not prefix or list(base)[-len(prefix):]!=prefix:raise ValueError('Wrong analysis prompt')
 if len(tokens)>10000 or len(tokens)!=len(logs)or any(type(t)is not int or t<0 for t in tokens)or any(not math.isfinite(x)for x in logs):raise ValueError('Sample cap/alignment')
 raw=tok.decode(tokens,skip_special_tokens=False)
 if any(t in raw for t in ['<|return|>','<|call|>','<|start|>','<|channel|>','<|message|>']):raise ValueError('Premature terminal/tool/nested channel')
 hits=[i for i in range(len(tokens)-len(close)+1)if tokens[i:i+len(close)]==close];finish=result.get('finish_reason',result.get('stop_reason'));supplied=[]
 if hits:
  if len(hits)!=1 or hits[0]+len(close)!=len(tokens)or raw.count(CLOSE)!=1:raise ValueError('Duplicate/nonterminal analysis closure')
  kind='natural_sampled_delimiter'
 elif CLOSE in raw:raise ValueError('Delimiter text/token mismatch')
 elif finish in ('stop','stop_sequence'):kind='natural_stop_delimiter_excluded';supplied=close
 elif len(tokens)==10000 and finish in (None,'length','max_tokens','max_length'):kind='forced_at_reasoning_cap';supplied=close
 else:raise ValueError('Unexplained short reasoning stop')
 final=tok.encode(FINAL,add_special_tokens=False);prompt=list(base)+tokens+supplied+final
 if len(prompt)+23000>context:raise ValueError('Context exceeds full final allowance')
 return {'stage2_prompt_tokens':prompt,'stage2_prompt_sha256':token_hash(prompt),'closure_kind':kind,'stage1_sampled_tokens':tokens,'stage1_sampled_logprobs':logs,'supplied_closure_tokens':supplied,'supplied_answer_prefill_tokens':final,'supplied_tokens_are_sampled':False,'context_tokens_upper_bound':len(prompt)+23000,'reasoning_text':raw,'score_final_only':True}
