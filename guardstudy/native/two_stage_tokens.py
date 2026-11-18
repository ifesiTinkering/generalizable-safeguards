"""Pure two-request conditioning/decoding. No SDK client, retries or provider calls.

Stage1 sampled tokens are preserved exactly. Injected closure/prefill tokens are
conditioning, never sampled actions. Only stage2 final text is eligible for score.
The caller reserves both requests, uses one immutable sampler, and schedules no
more than one outstanding request per row (400 rows maximum in flight).
"""
import hashlib,json,math
from .thinking_adapter import build_generation_prompt,decode_with_audit
THINK_CLOSE='</think>'
ANSWER_SUFFIX='\n\n<answer>\n'
STAGE1_CAP=10000
STAGE2_CAP=23000

def token_hash(tokens):return hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()
def _tokens(xs):
 if not isinstance(xs,(list,tuple)) or any(type(t)is not int or t<0 for t in xs):raise ValueError('Invalid token array')
 return list(xs)
def _sample(result,cap):
 tokens=_tokens(result['tokens']);logs=list(result.get('logprobs',result.get('token_logprobs',[])))
 if len(tokens)>cap or len(tokens)!=len(logs) or any(type(x)not in(int,float)or not math.isfinite(x) for x in logs):raise ValueError('Sample cap/logprob alignment violation')
 return tokens,logs

def make_stage1_prompt(on_renderer,messages):
 return build_generation_prompt(on_renderer,messages,thinking=True).to_ints()
def stage1_stop_sequences():return [THINK_CLOSE]

def assemble_stage2(base_prompt_tokens,reasoning_result,tokenizer,*,context_limit,terminal_token_ids=(),requested_stop_sequences=(THINK_CLOSE,)):
 """Return exact stage2 prompt and provenance, or fail before a second request.

 stop_reason/finish_reason='stop' is needed when the provider excludes the closing
 delimiter. The sole requested stop must be </think>; returned EOS/end tokens
 are rejected. With absent stop metadata, only an included delimiter or a full
 10000-token cap is conclusive. No guessed closure after an unexplained short stop.
 """
 base=_tokens(base_prompt_tokens);tokens,logs=_sample(reasoning_result,STAGE1_CAP)
 if tuple(requested_stop_sequences)!=(THINK_CLOSE,):raise ValueError('Stage1 must request only the thinking-close stop')
 if type(context_limit)is not int or context_limit<1:raise ValueError('Explicit audited context limit required')
 suffix=tokenizer.encode('<|im_start|>assistant\n<think>\n',add_special_tokens=False)
 if not suffix or base[-len(suffix):]!=suffix:raise ValueError('Stage1 prompt is not exact ON renderer framing')
 if any(t in set(terminal_token_ids) for t in tokens):raise ValueError('Premature EOS/end token in reasoning; preserve and do not retry')
 close=tokenizer.encode(THINK_CLOSE,add_special_tokens=False)
 if not close:raise ValueError('Empty delimiter encoding')
 starts=[i for i in range(len(tokens)-len(close)+1) if tokens[i:i+len(close)]==close]
 raw=tokenizer.decode(tokens,skip_special_tokens=False)
 if '<think>' in raw or len(starts)>1 or raw.count(THINK_CLOSE)>1:raise ValueError('Nested or duplicated thinking delimiter')
 finish=reasoning_result.get('finish_reason',reasoning_result.get('stop_reason'))
 if hasattr(finish,'value'):finish=finish.value
 supplied=[]
 if starts:
  pos=starts[0]
  if pos+len(close)!=len(tokens):raise ValueError('Stage1 continued beyond closing delimiter; no final-answer truncation allowed')
  if raw.count(THINK_CLOSE)!=1:raise ValueError('Delimiter token/text mismatch')
  kind='natural_sampled_delimiter';thought_tokens=tokens[:pos]
 elif THINK_CLOSE in raw:raise ValueError('Delimiter text/token mismatch')
 elif finish in ('stop','stop_sequence'):
  kind='natural_stop_delimiter_excluded';supplied=list(close);thought_tokens=tokens
 elif len(tokens)==STAGE1_CAP and finish in (None,'length','max_tokens','max_length'):
  kind='forced_at_reasoning_cap';supplied=list(close);thought_tokens=tokens
 else:raise ValueError('Unexplained short reasoning termination; no second request or retry')
 prefill=tokenizer.encode(ANSWER_SUFFIX,add_special_tokens=False)
 injected=supplied+prefill;prompt=base+tokens+injected
 if len(prompt)+STAGE2_CAP>context_limit:raise ValueError('Full23000 answer allowance exceeds audited context; no silent truncation')
 return {'stage2_prompt_tokens':prompt,'stage2_prompt_sha256':token_hash(prompt),'stage2_max_tokens':STAGE2_CAP,'context_limit':context_limit,'context_tokens_upper_bound':len(prompt)+STAGE2_CAP,'closure_kind':kind,'provider_stage1_finish_reason':finish,'stage1_sampled_tokens':tokens,'stage1_sampled_logprobs':logs,'stage1_sampled_token_count':len(tokens),'reasoning_tokens_excluding_close':thought_tokens,'reasoning_text':tokenizer.decode(thought_tokens,skip_special_tokens=False),'supplied_closure_tokens':supplied,'supplied_answer_prefill_tokens':prefill,'all_supplied_conditioning_tokens':injected,'supplied_tokens_are_sampled':False,'stage1_prompt_sha256':token_hash(base),'scoring_policy':'Score stage2 final answer only; reasoning is conditioning, not a scored completion'}

def decode_final(off_renderer,final_result,assembly):
 tokens,logs=_sample(final_result,STAGE2_CAP)
 if assembly.get('stage2_max_tokens')!=STAGE2_CAP or assembly.get('supplied_tokens_are_sampled')is not False:raise ValueError('Unverified assembly contract')
 audit=decode_with_audit(off_renderer,tokens,thinking=False)
 return {'text':audit['text'],'clean_termination':audit['clean'],'decoder_audit':audit,'final_sampled_tokens':tokens,'final_sampled_logprobs':logs,'final_sampled_token_count':len(tokens),'reasoning_sampled_token_count':assembly['stage1_sampled_token_count'],'closure_kind':assembly['closure_kind'],'forced_conditioning_token_count':len(assembly['all_supplied_conditioning_tokens']),'scoring_input':'final_text_only','provider_calls':0}
