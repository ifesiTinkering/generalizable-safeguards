"""Current four-tag answer-and-structure targets; original offset mask mechanism."""
import json
from legacy_target_builder import response_weights, target as legacy_target
import re

def target(row):
    ids=row['ground_truth_rule_ids'];allowed={str(x['rule_id'])for x in row['all_policy_rules']}
    if len(ids)!=len(set(ids))or not set(ids)<=allowed or bool(ids)!=row['is_violation']:raise ValueError('Invalid original labels')
    parts=[];spans=[];pos=0
    def add(s):
        nonlocal pos
        parts.append(s);pos+=len(s)
    def payload(s,name):
        start=pos;add(s);spans.append({'field':name,'start':start,'end':pos})
    add('<answer>\n[')
    for i,rid in enumerate(ids):
        if i:add(', ')
        encoded=json.dumps(rid,ensure_ascii=False);add(encoded)
    add(']</answer><evidence>\n')
    for rid in ids:
        # XML attribute escaping is mandatory; IDs remain masked, including entities.
        escaped=rid.replace('&','&amp;').replace('"','&quot;').replace('<','&lt;').replace('>','&gt;')
        add('<rule id="');payload(escaped,'evidence_rule_id');add('"><quote turn_id="');payload('1','turn_id');add('" speaker="');payload('assistant','speaker');add('"> ')
        payload('placeholder','quote');add(' </quote><justification> ');payload('placeholder','justification');add(' </justification></rule>\n')
    legacy,_=legacy_target(row)
    body=re.search(r'<analysis>\nReview: (.*?)\n</analysis>',legacy,re.S).group(1)
    confidence=re.search(r'<confidence>\n(.*?)\n</confidence>',legacy,re.S).group(1)
    add('</evidence><analysis>\nReview: ');payload(body,'analysis');add('\n</analysis><confidence>\n');payload(confidence,'confidence');add('\n</confidence>')
    return ''.join(parts),spans

def build_record(renderer,messages,row,max_length=32768):
    # Same renderer suffix alignment and response_weights guard as structural3.
    text,spans=target(row)
    inp,_=renderer.build_supervised_example(messages+[{'role':'assistant','content':text}])
    tokens=inp.to_ints()
    if len(tokens)>max_length:raise ValueError('Context overflow; no truncation')
    enc=renderer.tokenizer(text+'<|im_end|>',add_special_tokens=False,return_offsets_mapping=True)
    ids=enc['input_ids'];start=len(tokens)-len(ids)
    if start<1 or tokens[start:]!=ids:raise ValueError('Renderer target suffix mismatch')
    rw=response_weights(text,spans,enc['offset_mapping']);weights=[0.]*start+rw
    if weights[-1]!=1:raise ValueError('EOS must be supervised')
    return {'input_tokens':tokens[:-1],'target_tokens':tokens[1:],'weights':weights[1:],'audit':{'target':text,'masked_spans':spans,'response_start':start,'full_tokens':len(tokens),'supervised_tokens':sum(rw),'masked_tokens':rw.count(0.),'answer_ids_supervised':True,'synthetic_payload_supervised':False,'offsets':enc['offset_mapping'],'response_weights':rw}}
