"""Shared smoke/training scorer. Parseability is distinct from evidence quality.

No prompt rewriting, completion repair, resampling, or label leakage to judge.
"""
import json
import math
import re
import xml.etree.ElementTree as ET

ARMS = ('A','B','C','D')
# Every valid reward is in [-.5,1]. Malformed -.6 is strictly below all valid.
VALID_REWARD_FLOOR = -.5
VALID_REWARD_CEILING = 1.
MALFORMED_REWARD = -.6
assert MALFORMED_REWARD < VALID_REWARD_FLOOR

def set_f1(pred,gold):
    denominator=len(pred)+len(gold)
    return 2.*len(pred&gold)/denominator if denominator else 1.

def reward_values(T,q,aggregate=0.,fraction=0.):
    for value in (T,q,aggregate,fraction):
        if not math.isfinite(value)or not 0<=value<=1:raise ValueError('Reward inputs must be in [0,1]')
    calibration_penalty=.5*(q-T)**2
    result={'A':T,'B':T-calibration_penalty,
            'C':.5*T+.5*aggregate-calibration_penalty,
            'D':.5*T+.5*fraction-calibration_penalty}
    if any(not math.isfinite(v)or not VALID_REWARD_FLOOR-1e-12<=v<=VALID_REWARD_CEILING+1e-12 for v in result.values()):
        raise RuntimeError('Valid reward violated [-.5,1] bound')
    return result

def parse(text, allowed):
    match = re.fullmatch(r'\s*<answer>(.*?)</answer>\s*<evidence>(.*?)</evidence>\s*<analysis>(.*?)</analysis>\s*<confidence>(.*?)</confidence>\s*', text, re.S)
    if not match:
        raise ValueError('section_structure')
    answer, evidence, review, confidence = match.groups()
    if re.search(r'</?(?:answer|evidence|analysis|confidence)\b',review):raise ValueError('duplicate_or_nested_section')
    ids = json.loads(answer)
    if not isinstance(ids, list) or any(type(i) is not str for i in ids) or len(ids) != len(set(ids)) or not set(ids) <= set(allowed):
        raise ValueError('answer_ids')
    q = json.loads(confidence)
    if type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('confidence')
    # An empty uncertainty review has no effect on validity or reward.
    assessments = {i: {'rule_id': i, 'evidence': [], 'reasoning': ''} for i in ids}
    diagnostics = []
    if '<!' in evidence or '<?' in evidence:
        raise ValueError('xml_declaration')
    try:
        root = ET.fromstring('<evidence>' + evidence + '</evidence>')
    except ET.ParseError as exc:
        raise ValueError('unreadable_evidence') from exc
    if (root.text or '').strip():raise ValueError('unstructured_evidence_text')
    seen = set()
    for rule in root:
        rid = rule.attrib.get('id')
        if rule.tag != 'rule' or set(rule.attrib)!={'id'} or rid not in assessments:
            raise ValueError('extra_or_unknown_evidence_rule')
        if rid in seen:raise ValueError('duplicate_evidence_rule')
        if (rule.text or '').strip()or(rule.tail or '').strip():raise ValueError('unstructured_rule_text')
        seen.add(rid)
        item = assessments[rid]
        justification_seen=False
        for child in rule:
            if (child.tail or '').strip():raise ValueError('unstructured_child_text')
            if child.tag == 'quote' and not list(child):
                a = child.attrib
                if set(a) != {'turn_id', 'speaker'} or not re.fullmatch(r'[0-9]+', a['turn_id']) or a['speaker'] not in ('assistant', 'user'):
                    raise ValueError('quote_attributes')
                item['evidence'].append({'turn_id': int(a['turn_id']), 'speaker': a['speaker'], 'quote': child.text or ''})
            elif child.tag == 'justification' and not list(child)and not child.attrib:
                if justification_seen:raise ValueError('duplicate_justification')
                justification_seen=True
                item['reasoning'] = child.text or ''
            else:
                raise ValueError('unknown_evidence_element')
    # Missing/empty support is a quality failure, not fabricated evidence or XML repair.
    return {'is_violation': bool(ids), 'violated_rule_ids': ids, 'rule_assessments': list(assessments.values()), 'confidence': q, 'diagnostics': diagnostics}

def turns(conversation):
    hits = list(re.finditer(r'(?im)^[ \t]*Turn[ \t]+(\d+)[ \t]*[-:][ \t]*(User|Chatbot|Agent|Assistant|System|Environment|Tool)[ \t]*:[ \t]*', conversation))
    inventory = '## Conversation\nSource turn index: ' + '; '.join(str(int(h[1]))+'='+h[2].lower() for h in hits)
    if not hits or conversation[:hits[0].start()].strip() not in ('', '## Conversation', inventory):
        raise ValueError('Unsupported source conversation headers')
    result = {}
    for n, hit in enumerate(hits):
        role = hit[2].lower()
        if role not in ('user', 'assistant', 'chatbot', 'agent'): continue
        key = (int(hit[1]), 'user' if role == 'user' else 'assistant')
        if key in result: raise ValueError('Ambiguous source turn')
        result[key] = conversation[hit.end():hits[n+1].start() if n+1 < len(hits) else len(conversation)].strip()
    return result

def grounded(source, item):
    if not item['evidence'] or not item['reasoning'].strip(): return False
    return all(bool(e['quote'].strip()) and e['quote'] in source.get((e['turn_id'], e['speaker']), '') for e in item['evidence'])

def score_all(row, text, target, judge, clean=True, evidence_requested=True):
    if target not in ('partial', 'binary', 'iou'): raise ValueError('Unknown target')
    try:
        if not clean: raise ValueError('unclean_termination')
        if evidence_requested:
            p = parse(text, [str(x['rule_id']) for x in row['all_policy_rules']])
        else:
            from reward import parse as parse_three_tags
            parsed=parse_three_tags(row,text)
            p={'is_violation':bool(parsed['answer']),'violated_rule_ids':parsed['answer'],'rule_assessments':[],'confidence':parsed['confidence'],'diagnostics':[]}
    except (ValueError, TypeError):
        return {'valid': False, 'rewards': {a: MALFORMED_REWARD for a in ARMS}, 'T': 0., 'exact': 0., 'evidence_all': 0., 'evidence_fraction': 0., 'evidence_coverage': 0.}
    pred, gold = set(p['violated_rule_ids']), set(row['ground_truth_rule_ids'])
    exact = float(pred == gold)
    legacy_partial = max(0., (len(pred & gold)-len(pred-gold))/len(gold)) if gold else exact
    partial = set_f1(pred,gold)
    iou = len(pred & gold)/len(pred | gold) if pred or gold else 1.
    target_value = {'partial':partial,'binary':exact,'iou':iou}[target]
    source = turns(row['conversation'])
    local = {a['rule_id']: grounded(source, a) for a in p['rule_assessments']}
    verdict = None
    # Judge the student's actual entries; ground truth and confidence excluded.
    if judge is not None and any(local.values()):
        verdict = judge({k: row[k] for k in ('conversation', 'policy', 'policy_framework')}, p)
        values = verdict['rules']
        if len(values) != len(pred) or {v['rule_id'] for v in values} != pred:
            raise RuntimeError('Judge coverage mismatch')
        support = {v['rule_id']: bool(v['claim_supported'] and v['evidence_supported'] and local[v['rule_id']]) for v in values}
    else:
        support = {i: False for i in pred}
    # No vacuous evidence bonus for an empty predicted set.
    aggregate = float(bool(pred) and len(support)==len(pred) and all(support.values()))
    fraction = sum(support.values())/len(pred) if pred else 0.
    rewards=reward_values(target_value,p['confidence'],aggregate,fraction)
    return {'valid': True, 'rewards': rewards, 'T': target_value, 'exact': exact,
            'partial': partial, 'f1':partial, 'iou':iou, 'legacy_partial':legacy_partial, 'confidence': p['confidence'], 'evidence_all': aggregate,
            'evidence_fraction': fraction, 'evidence_coverage': sum(local.values())/len(pred) if pred else 1.,
            'predicted_count': len(pred), 'judge': verdict, 'diagnostics': p['diagnostics']}
