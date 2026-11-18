"""User-authorized shared Qwen/GPT selection-component stopping rule; pure functions.

Selection definitions are unchanged: exact/F1/IoU correctness, invalid-zero
exact-confidence calibration, and aggregate/per-rule semantic evidence.
Unknown components never become failed votes. Historical adoption evaluates
only the latest five checkpoints, without inventing an earlier stop decision.
"""
from copy import deepcopy
import math

RULE = 'selection_three_majority_rolling_five_v1'
NAMES = ('correctness', 'calibration', 'evidence')
DELTA = .03


def bounded(value):
    return type(value) in (float, int) and math.isfinite(value) and -1e-12 <= value <= 1 + 1e-12


def components(metrics):
    """Accept canonical reporting metrics already stored in every complete history."""
    c = [metrics.get(k) for k in ('exact_set_accuracy', 'macro_f1', 'jaccard_iou')]
    e = [metrics.get(k) for k in ('semantic_aggregate_evidence', 'semantic_per_rule_evidence')]
    valid, brier = metrics.get('valid_rate'), metrics.get('exact_correctness_brier')
    calibration = 0. if valid == 0 and bounded(valid) else valid * (1 - brier) if bounded(valid) and bounded(brier) else None
    return {'correctness': sum(c) / 3 if all(bounded(v) for v in c) else None,
            'calibration': calibration,
            'evidence': sum(e) / 2 if all(bounded(v) for v in e) else None}


def evaluate(points, *, delta=DELTA):
    if not (type(delta) in (int, float) and math.isfinite(delta) and abs(delta-DELTA) < 1e-15):
        raise ValueError('User-authorized minimum delta is exactly .03')
    points = deepcopy(points)
    ids = [p['update'] for p in points]
    if any(type(u) is not int or u < 0 for u in ids) or ids != sorted(set(ids)):
        raise ValueError('Unique chronological successful-update IDs required')
    window = points[-5:]
    votes = {}
    for name in NAMES:
        values = [p['components'].get(name) for p in window]
        eligible = len(window) == 5 and all(bounded(v) for v in values)
        best = max(values[1:]) if eligible else None
        gain = best-values[0] if eligible else None
        improves = gain >= delta-1e-12 if eligible else None
        votes[name] = {'eligible': eligible, 'values': values, 'baseline': values[0] if values else None,
                       'best_newest_four': best, 'gain': gain, 'improves': improves,
                       'failed': improves is False}
    improved = [k for k, v in votes.items() if v['improves'] is True]
    failed = [k for k, v in votes.items() if v['improves'] is False]
    unknown = [k for k, v in votes.items() if v['improves'] is None]
    return {'rule': RULE, 'window': window, 'window_update_ids': [p['update'] for p in window],
            'delta': delta, 'metric_votes': votes, 'improved': improved, 'failed': failed,
            'unknown': unknown, 'stop': len(failed) >= 2,
            'decision': 'stop' if len(failed) >= 2 else 'continue' if len(improved) >= 2 else 'inconclusive',
            'complete_checks': len(points), 'validation_pending': False}


def migrate_latest(history, *, delta=DELTA):
    """Adopt now using saved complete scheduled history; do not replay old stop latches."""
    points = []
    for h in history:
        if h.get('complete') is False or h.get('incomplete') or not (h.get('grid') or h['updates'] == 0):
            continue
        points.append({'update': h['updates'], 'components': components(h['stop_metrics'])})
    result = evaluate(points, delta=delta)
    result['adoption'] = 'latest_complete_window_only_not_retroactive_first_stop'
    return result


def observe(previous, metrics, *, complete, scheduled, baseline=False, delta=DELTA, patience=None, checkpoint_update=None):
    state = deepcopy(previous or {})
    if state and state.get('rule') != RULE:
        raise ValueError('Explicit latest-history policy adoption required')
    if state.get('stop'):
        return state  # Existing decision under this exact rule is latched.
    if not complete:
        return dict(state or evaluate([], delta=delta), validation_pending=True)
    if not scheduled and not baseline:
        return state or evaluate([], delta=delta)
    if type(checkpoint_update) is not int or checkpoint_update < 0:
        raise ValueError('Successful-update ID required')
    points = state.get('window', [])
    if points and checkpoint_update <= points[-1]['update']:
        raise ValueError('Duplicate or out-of-order complete validation')
    result = evaluate(points + [{'update': checkpoint_update, 'components': components(metrics)}], delta=delta)
    result['complete_checks'] = state.get('complete_checks', 0) + 1
    return result
