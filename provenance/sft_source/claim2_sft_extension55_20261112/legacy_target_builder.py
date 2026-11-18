"""Offline contingency only. No SDK imports, provider clients, or launch path."""
import json


def target(row):
    import hashlib
    ids = row['ground_truth_rule_ids']
    allowed = {r['rule_id'] for r in row['all_policy_rules']}
    if any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise ValueError('Gold IDs must be unique original policy strings')
    if type(row['is_violation']) is not bool or bool(ids) != row['is_violation']:
        raise ValueError('Decision/gold IDs disagree')
    if not isinstance(row.get('row_id'),str): raise ValueError('Stable row ID required')
    h=int(hashlib.sha256(row['row_id'].encode()).hexdigest(),16)
    source=' '.join(str(row.get(k,'')) for k in ('policy','conversation'))
    words=source.replace('<','(').replace('>',')').split()
    if not words: raise ValueError('Original row context required')
    count=(32,96,192,384)[h%4]
    offset=(h//4)%len(words)
    body=' '.join(words[(offset+i)%len(words)] for i in range(count))+'.'
    q=f'{(h//16)%10/10+.05:.2f}'
    text='<answer>\n['; spans=[]
    for i,value in enumerate(ids):
        if i:text+=', '
        encoded=json.dumps(value,ensure_ascii=False)
        start=len(text); text+=encoded
        spans.append({'field':'answer_id', 'start':start+1,'end':len(text)-1})
    text+=']</answer><analysis>\nReview: '
    start=len(text);text+=body
    spans.append({'field':'analysis', 'start':start,'end':len(text)})
    text+='\n</analysis><confidence>\n'
    start=len(text);text+=q
    spans.append({'field':'confidence','start':start,'end':len(text)})
    text+='\n</confidence>'
    return text,spans


def response_weights(text, spans, offsets):
    """Mask only answer-ID/review/confidence payloads; reject tokens mixing them with syntax."""
    weights=[]
    for a,b in offsets:
        overlap=any(a<s['end'] and b>s['start'] for s in spans)
        if overlap:
            for i in range(a,min(b,len(text))):
                if not text[i].isspace() and not any(s['start']<=i<s['end'] for s in spans):
                    raise ValueError('Masked token crosses supervised structure')
        weights.append(0. if overlap else 1.)
    return weights


def build_record(renderer, messages, row, max_length=32768):
    """Use renderer-established target suffix; return pure cross-entropy arrays.

    Renderer/tokenizer are injected. The caller must never truncate these data.
    All prompt tokens receive zero target loss. Placeholder payload tokens are
    present as context but do not become confidence/reasoning training labels.
    """
    text, spans=target(row)
    model_input,_=renderer.build_supervised_example(messages+[{'role':'assistant','content':text}])
    tokens=model_input.to_ints()
    if len(tokens)>max_length: raise ValueError('Context overflow; no truncation')
    enc=renderer.tokenizer(text+'<|im_end|>',add_special_tokens=False,return_offsets_mapping=True)
    ids=enc['input_ids']; start=len(tokens)-len(ids)
    if start<1 or tokens[start:] != ids: raise ValueError('Renderer target suffix mismatch')
    rw=response_weights(text,spans,enc['offset_mapping'])
    weights=[0.]*start+rw
    if weights[-1]!=1: raise ValueError('EOS must be supervised')
    return {'input_tokens':tokens[:-1], 'target_tokens':tokens[1:], 'weights':weights[1:],
            'audit':{'target':text,'masked_spans':spans,'response_start':start,
                     'full_tokens':len(tokens),'supervised_tokens':sum(rw),
                     'masked_tokens':rw.count(0.),'confidence_payload_supervised':False}}
