"""User-requested exact successful-update archives; never fabricate missed states."""
from copy import deepcopy
MILESTONES=tuple(sorted(set(range(5,300,10))|{50}))
def checkpoint_name(config,job,updates,attempt,updated,pins):
 if updated and updates in MILESTONES and str(updates) not in pins:
  return f"claim2_fresh108_20261113-{config['target']}-{job['arm']}-seed{job['seed']}-update{updates:06d}"
 return f'a{attempt}'
def pin(state,job,target,updated):
 u=state['updates']
 if not updated or u not in MILESTONES:return
 pins=state.setdefault('milestone_pins',{})
 if str(u)in pins:return
 pins[str(u)]={k:deepcopy(state[k])for k in ('checkpoint','sampling_path','updates','attempt','queue_cursor')}
 pins[str(u)].update(readable_name=f"claim2_fresh108_20261113-{target}-{job['arm']}-seed{job['seed']}-update{u:06d}",full_optimizer_state=True,requested_exact_update=u,expires_at=None,prune_protected=True)
