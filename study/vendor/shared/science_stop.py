"""User-selected common any-metric milestone patience, no provider operations."""
def patience_status(state,config=None):
    return state.get('patience_status',{'anchors':{},'stale_checks':0,'complete_checks':0,'stop':False})
def should_stop(state,config):
    if state.get('pending'):return None
    if state.get('stop_decision_update') is not None:return 'validation_rolling_window'
    if state['updates']>=config['max_updates']:return 'maximum_updates'
    if patience_status(state,config).get('stop'):return 'validation_patience_exhausted'
    return None
def validation_due(state,*,updated,pass_end,config):
    return bool(updated and state['updates']%config['validation_every_updates']==0)
