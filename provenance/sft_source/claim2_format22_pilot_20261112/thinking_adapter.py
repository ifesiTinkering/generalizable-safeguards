"""Actual Qwen3.5 SDK thinking framing, with strict final-answer extraction.

The adapter never slices or rewrites sampled token/logprob arrays. All sampled
thinking+answer tokens remain the training action sequence. Prefill tokens are
conditioning only and are never fabricated as sampled tokens.
"""
import re


def _check(thinking):
    if type(thinking) is not bool:raise ValueError('thinking must be boolean')


def generation_prefill(thinking):
    _check(thinking)
    return '' if thinking else '<answer>\n'


def renderer(thinking,tokenizer=None):
    _check(thinking)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('/workspace/tokenizer-qwen35',local_files_only=True)
    from tinker_cookbook import renderers
    return renderers.get_renderer('qwen3_5' if thinking else 'qwen3_5_disable_thinking',tokenizer)


def build_generation_prompt(r,messages,*,thinking):
    """Verify the real renderer suffix, not just a configuration string."""
    _check(thinking)
    prompt=r.build_generation_prompt(messages,prefill=generation_prefill(thinking))
    suffix='<|im_start|>assistant\n<think>\n' if thinking else '<|im_start|>assistant\n<think>\n\n</think>\n\n<answer>\n'
    expected=r.tokenizer.encode(suffix,add_special_tokens=False)
    tokens=prompt.to_ints()
    if tokens[-len(expected):]!=expected:raise RuntimeError('Renderer thinking/prefill suffix mismatch')
    return prompt


def decode_with_audit(r,tokens,*,thinking):
    _check(thinking)
    raw=r.tokenizer.decode(tokens,skip_special_tokens=False)
    # Pass a copy in case an upstream renderer normalizes its input in place.
    message,termination=r.parse_response(list(tokens))
    clean=bool(termination.is_clean);reasons=[]
    if not clean:reasons.append('renderer_unclean_termination')
    if message.get('tool_calls') or message.get('unparsed_tool_calls'):
        clean=False;reasons.append('unexpected_tool_output')
    content=message.get('content','');thought='';parts=[]
    if isinstance(content,str):parts=[{'type':'text','text':content}]
    elif isinstance(content,list):parts=content
    else:clean=False;reasons.append('unexpected_content_type')
    if thinking:
        if raw.count('</think>')!=1 or '<think>' in raw:
            clean=False;reasons.append('missing_or_duplicate_sampled_thinking_delimiter')
        if not parts or parts[0].get('type')!='thinking' or sum(p.get('type')=='thinking' for p in parts)!=1:
            clean=False;reasons.append('missing_or_misordered_thinking_part')
        elif any(p.get('type')!='text' for p in parts[1:]):
            clean=False;reasons.append('unexpected_post_thinking_part')
        else:thought=parts[0].get('thinking','')
        final=''.join(p.get('text','') for p in parts if p.get('type')=='text')
    else:
        if '<think>' in raw or '</think>' in raw or any(p.get('type')!='text' for p in parts):
            clean=False;reasons.append('thinking_emitted_in_disabled_mode')
        final=generation_prefill(False)+''.join(p.get('text','') for p in parts if p.get('type')=='text')
    thinking_tokens=0
    if thinking:
        close_ids=r.tokenizer.encode('</think>',add_special_tokens=False)
        thinking_tokens=(list(tokens).index(close_ids[0]) if len(close_ids)==1 and list(tokens).count(close_ids[0])==1 else None)
    return {'text':final,'clean':clean,'renderer_clean':bool(termination.is_clean),
            'raw_sampled_text':raw,'thinking_enabled':thinking,
            'thinking_characters':len(thought),'thinking_nonempty':bool(thought.strip()),
            'sampled_thinking_token_count':thinking_tokens,'thinking_token_count_excludes_prefill_and_closing_delimiter':True,
            'decode_failure_reasons':reasons,'sampled_token_count':len(tokens),
            'training_tokens_policy':'all_original_sampled_tokens_including_thinking'}


def decode(r,tokens,*,thinking):
    value=decode_with_audit(r,tokens,thinking=thinking)
    return value['text'],value['clean']


class Decoder:
    """Engine-compatible callable; engine may persist last_audit after each call."""
    def __init__(self,r,*,thinking):_check(thinking);self.renderer=r;self.thinking=thinking;self.last_audit=None
    def __call__(self,tokens):
        self.last_audit=decode_with_audit(self.renderer,tokens,thinking=self.thinking)
        return self.last_audit['text'],self.last_audit['clean']


def thinking_smoke_summary(records,*,thinking):
    """Preregistered manipulation check, not a measure of reasoning quality.

    ON fails if zero valid outputs contain nonempty sampled thought text.
    Majority/nonmajority rates are descriptive and never change this threshold.
    """
    _check(thinking)
    audits=[]
    for r in records:
        audit=r.get('decoder_audit')
        if not isinstance(audit,dict) or audit.get('thinking_enabled') is not thinking:
            raise ValueError('Every smoke output must retain its actual decoder audit')
        audits.append((r,audit))
    valid=[a for r,a in audits if r['valid']]
    nonempty=sum(bool(a['thinking_nonempty']) for _,a in audits)
    valid_nonempty=sum(bool(a['thinking_nonempty']) for a in valid)
    import statistics
    chars=[a['thinking_characters'] for _,a in audits]
    token_counts=[a.get('sampled_thinking_token_count') for _,a in audits]
    measured_tokens=[x for x in token_counts if x is not None]
    return {'thinking_enabled':thinking,'n_outputs':len(records),'n_valid':len(valid),
        'nonempty_thinking_outputs':nonempty,'nonempty_thinking_valid_outputs':valid_nonempty,
        'nonempty_fraction_all':nonempty/len(records) if records else None,
        'nonempty_fraction_valid':valid_nonempty/len(valid) if valid else None,
        'thinking_characters_mean':statistics.mean(chars) if chars else None,
        'sampled_thinking_tokens_total':sum(measured_tokens),'sampled_thinking_tokens_measured_outputs':len(measured_tokens),
        'thinking_characters_median':statistics.median(chars) if chars else None,
        'majority_valid_nonempty':bool(valid and valid_nonempty/len(valid)>=.5),
        'mostly_immediate_closure':bool(thinking and valid and valid_nonempty/len(valid)<.5),
        'failure_reasons':['no_nonempty_valid_thinking_output'] if thinking and valid_nonempty==0 else [],
        'preregistered_gate':'ON requires at least one valid output with nonempty sampled thinking; zero fails; majority rate descriptive only.',
        'not_a_reasoning_quality_measure':True,'prefill_excluded_from_thinking_text':True}
