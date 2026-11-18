"""Offline tagged-answer contract and signed partial-correctness rewards.

No semantic judge, evidence scoring, service imports, or model calls. The optional
judge argument is deliberately ignored for engine call-site compatibility.
"""
import json
import math
import re

CORRECTNESS_WEIGHT = .9
EVIDENCE_WEIGHT = 0.
CALIBRATION_WEIGHT = .5
INVALID_REWARD = -.1
_PATTERN = re.compile(r'\A\s*<answer>([^<>]*)</answer>\s*<analysis>(.*?)</analysis>\s*<confidence>([^<>]*)</confidence>\s*\Z', re.DOTALL)

# Bare '<word' remains reserved as tag-like syntax (including incomplete tags).
# Ordinary arrows, numeric comparisons and spaced comparisons such as 'x < y'
# are plain text. Reserved contract tags are rejected even with malformed spaces.
_ANALYSIS_MARKUP = re.compile(
    r'<(?:[A-Za-z/!?]|\s*/|\s*(?:answer|analysis|confidence|think)\b)'
    r'|<\s+[A-Za-z][A-Za-z0-9:_-]*(?:\s+[^<>]*)?/?>', re.IGNORECASE)


def parse(row, text):
    """Exactly three ordered tags, once each, with no surrounding substantive text."""
    if not isinstance(text, str):
        raise ValueError('Output must be text')
    match = _PATTERN.fullmatch(text)
    if match is None:
        raise ValueError('Expected answer, analysis, confidence tags exactly once in order')
    answer = json.loads(match[1])
    if not isinstance(answer, list) or any(type(x) is not str for x in answer):
        raise ValueError('Answer must be a JSON array of original string rule IDs')
    if len(answer) != len(set(answer)):
        raise ValueError('Duplicate rule IDs')
    allowed = {str(r['rule_id']) for r in row['all_policy_rules']}
    if not set(answer) <= allowed:
        raise ValueError('Unknown rule ID')
    analysis = match[2].strip()
    if not analysis:
        raise ValueError('Analysis must be nonempty plain text')
    if _ANALYSIS_MARKUP.search(analysis):
        raise ValueError('Analysis must not contain nested or malformed tag-like markup')
    confidence = json.loads(match[3])
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Confidence must be a finite number in [0,1]')
    return {'answer': answer, 'analysis': analysis, 'confidence': confidence}


def quality(gold, predicted, decision_correct):
    if not decision_correct:
        return 0.
    g, p = set(gold), set(predicted)
    return 2. * len(g & p) / (len(g) + len(p)) if g else float(not p)


def reward_value(partial_score, confidence, *, arm='B', valid=True):
    if arm not in ('A', 'B', 'C'):
        raise ValueError('Arm must be A, B or C')
    if not valid:
        return INVALID_REWARD
    for x in (partial_score, confidence):
        if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError('Reward components must be finite numbers in [0,1]')
    return .9*partial_score - (0. if arm == 'A' else .5*(confidence-partial_score)**2)


def score(row, text, judge=None, *, clean_termination=True, arm='B'):
    if arm not in ('A', 'B', 'C'):
        raise ValueError('Arm must be A, B or C')
    gold = row['ground_truth_rule_ids']
    allowed = {str(r['rule_id']) for r in row['all_policy_rules']}
    if (type(row['is_violation']) is not bool or not isinstance(gold, list)
            or any(type(x) is not str for x in gold) or len(gold) != len(set(gold))
            or bool(gold) != row['is_violation'] or not set(gold) <= allowed):
        raise ValueError('Invalid source target')
    source_domain = row.get('domain')
    domain = (row.get('target_domain') or source_domain) if not row['is_violation'] else source_domain
    base = {'row_id': row.get('row_id'), 'domain': domain, 'source_domain': source_domain,
        'gold_positive': row['is_violation'], 'confidence': None, 'arm': arm,
        'exact_correct': 0., 'parsed_exact_correct': 0., 'quality_f1': 0.,
        'correctness_reward': 0., 'partial_score': 0., 'binary_correct': 0.,
        'evidence_supported': None, 'evidence_status': 'not_measured',
        'whole_correct': None, 'grounded': None, 'judge': None,
        'predicted_positive': None, 'f1_calibration_mse': None,
        'partial_calibration_mse': None, 'exact_calibration_mse': None,
        'calibration_target_type': 'floored_signed_partial_correctness',
        'calibration_target': None, 'calibration_penalty': None,
        'grounding_failure_reasons': [], 'predicted_rule_ids': None}
    try:
        if not clean_termination:
            raise ValueError('unclean_termination')
        obj = parse(row, text)
    except (ValueError, TypeError, KeyError) as exc:
        return base | {'valid': False, 'reward': INVALID_REWARD,
            'invalid_reason': 'unclean_termination' if not clean_termination else f'{type(exc).__name__}: {exc}',
            'reward_components': {'quality': 0., 'evidence': 0., 'calibration': 0., 'invalid': INVALID_REWARD}}
    p, g = set(obj['answer']), set(gold)
    decision_correct = bool(p) == row['is_violation']
    tp, fp, fn = len(p & g), len(p-g), len(g-p)
    partial = max(0., (tp-fp)/len(g)) if g and decision_correct else float(not g and not p)
    f1 = quality(g, p, decision_correct)
    exact = float(decision_correct and p == g)
    q = obj['confidence']
    mse = (q-partial)**2
    penalty = 0. if arm == 'A' else .5*mse
    return base | {'valid': True, 'invalid_reason': None,
        'reward': reward_value(partial, q, arm=arm), 'quality_f1': f1,
        'correctness_reward': partial, 'partial_score': partial,
        'exact_correct': exact, 'parsed_exact_correct': exact,
        'binary_correct': float(decision_correct), 'confidence': q,
        'predicted_positive': bool(p), 'predicted_rule_ids': obj['answer'],
        'f1_calibration_mse': (q-f1)**2, 'partial_calibration_mse': mse,
        'exact_calibration_mse': (q-exact)**2,
        'calibration_target': partial, 'calibration_penalty': penalty,
        'TP': tp, 'FP': fp, 'FN': fn,
        'reward_components': {'quality': .9*partial, 'evidence': 0.,
                              'calibration': -penalty, 'invalid': 0.}}
