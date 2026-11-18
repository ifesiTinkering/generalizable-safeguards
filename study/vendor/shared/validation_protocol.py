"""Offline staging only: common any-metric patience and durable retention plan.

No validity-rate/AP safeguards are added: every named metric is eligible under
its declared direction. Confidence-population and error-prevalence caveats must
remain visible in reports. Runtime integration is separately gated.
"""
import math
from copy import deepcopy
from validation_metric_registry import STOP_DIRECTIONS

def finite(value):return type(value)in(int,float)and math.isfinite(value)
def observe(previous,metrics,*,complete,scheduled,baseline=False,delta=.03,patience=None,checkpoint_update=None):
    """One shared five-checkpoint window for every canonical metric."""
    state=deepcopy(previous or {})
    if state and state.get('rule')!='rolling_five':raise ValueError('Explicit audited history migration required')
    state.setdefault('rule','rolling_five');state.setdefault('window',[])
    state.setdefault('complete_checks',0)
    if state.get('stop'):return dict(state,stop=True) # scientific decision is latched
    if not complete:return dict(state,validation_pending=True,stop=False,improved=[])
    if set(STOP_DIRECTIONS)-set(metrics):raise ValueError('Missing canonical metrics')
    if not scheduled and not baseline:return dict(state,validation_pending=False,stop=False,improved=[])
    if type(checkpoint_update)is not int:raise ValueError('Checkpoint update ID required')
    if state['window'] and checkpoint_update<=state['window'][-1]['update']:raise ValueError('Duplicate or out-of-order validation')
    point={'update':checkpoint_update,'metrics':{k:(metrics[k]if finite(metrics[k])else None)for k in STOP_DIRECTIONS}}
    state['window']=(state['window']+[point])[-5:];state['complete_checks']+=1
    votes={};improved=[]
    if len(state['window'])==5:
        for name,direction in STOP_DIRECTIONS.items():
            values=[p['metrics'][name]for p in state['window']]
            # Undefined within this shared window cannot pull older values forward.
            eligible=all(v is not None for v in values)
            best=(max(values[1:])if direction=='max'else min(values[1:]))if eligible else None
            gain=((best-values[0])if direction=='max'else(values[0]-best))if eligible else None
            vote=bool(eligible and gain>=delta-1e-12)
            votes[name]={'direction':direction,'baseline':values[0],'best_newest_four':best,'gain':gain,'eligible':eligible,'improves':vote}
            if vote:improved.append(name)
    return dict(state,validation_pending=False,stop=len(state['window'])==5 and not improved,improved=improved,delta=delta,window_update_ids=[p['update']for p in state['window']],metric_votes=votes)

def update_pins(previous,checkpoint,metrics,directions,*,complete):
    """Track exact full-precision best for EVERY registered metric; no .02 rule."""
    pins=deepcopy(previous or {})
    if not complete:return pins
    if set(metrics)-set(directions):raise ValueError('Every metric needs a frozen direction')
    for name,value in metrics.items():
        if not finite(value):continue
        direction=directions[name]
        if direction not in('min','max'):raise ValueError('Unknown direction')
        old=pins.get(name)
        if old is None or (value>old['value']if direction=='max'else value<old['value']):
            pins[name]={'checkpoint':checkpoint,'value':value,'direction':direction}
    return pins

def protected_checkpoints(pins,*,current=None,final=None,recovery=(),pending_reports=()):
    return sorted({v['checkpoint']for v in pins.values()}|set(recovery)|set(pending_reports)|{p for p in(current,final)if p is not None})

def grade_semantic_validation(rows,text_by_id,target,judge,clean_by_id=None):
    """Identical semantic grading for A/B/C/D; no arm-dependent judge bypass.

    A missing/unknown provider result must propagate to the caller as incomplete;
    it must never become a semantic zero or a patience failure.
    """
    from reward_pipeline import score_all
    if set(text_by_id)!={r['row_id']for r in rows}:raise ValueError('Validation population mismatch')
    clean_by_id=clean_by_id or {r['row_id']:True for r in rows}
    return [score_all(r,text_by_id[r['row_id']],target,judge,clean_by_id[r['row_id']],evidence_requested=True)for r in rows]
