"""Native Harmony framing: OFF23k vs ON10k TOTAL, no two-stage extension."""
OFF_PREFIX='<|channel|>final<|message|><answer>\n'
ON_PREFIX='<|channel|>analysis<|message|>'
FINAL_BOUNDARY='<|end|><|start|>assistant<|channel|>final<|message|>'
def prompt(renderer,messages,mode):
 if mode not in ('OFF','ON'):raise ValueError('Unknown mode')
 prefix=OFF_PREFIX if mode=='OFF' else ON_PREFIX
 tokens=renderer.build_generation_prompt(messages,prefill=prefix).to_ints()
 expected=renderer.tokenizer.encode('<|start|>assistant'+prefix,add_special_tokens=False)
 if tokens[-len(expected):]!=expected:raise ValueError('Native Harmony generation suffix mismatch')
 return tokens
def decode_text(raw,mode):
 clean=raw.endswith('<|return|>') and raw.count('<|return|>')==1 and '<|call|>' not in raw
 body=raw[:-len('<|return|>')] if raw.endswith('<|return|>') else raw
 reasons=[];thought=''
 if mode=='OFF':
  if any(x in body for x in ('<|channel|>','<|start|>','<|end|>','<|message|>')):clean=False;reasons.append('unexpected_Harmony_header_OFF')
  text='<answer>\n'+body
 elif mode=='ON':
  if body.count(FINAL_BOUNDARY)!=1:clean=False;reasons.append('missing_or_duplicate_native_final_boundary');text=''
  else:
   thought,text=body.split(FINAL_BOUNDARY)
   if any(x in thought or x in text for x in ('<|channel|>','<|start|>','<|end|>','<|message|>')):clean=False;reasons.append('unexpected_native_channel_sequence')
 else:raise ValueError('Unknown mode')
 if not raw.endswith('<|return|>'):reasons.append('no_terminal_return')
 return {'text':text,'clean':clean,'raw_sampled_text':raw,'thinking_enabled':mode=='ON','thinking_nonempty':bool(thought.strip()),'thinking_characters':len(thought),'decode_failure_reasons':reasons,'prefill_is_conditioning_not_generated':True,'single_call_total_budget':10000 if mode=='ON' else 23000}
