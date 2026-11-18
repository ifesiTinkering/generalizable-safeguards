"""Immutable snapshot validation; training state is mutated only by its owner.

Runner supplies a dedicated admitted meter for each scheduled checkpoint. Never
reuse a mutable training-cohort meter after its round closes.
"""
from copy import copy,deepcopy
from pathlib import Path
from protocol import save
from validation_metric_adapter import metric_values
from validation_protocol import observe,update_pins

def evaluate_snapshot(engine, snapshot, meter):
    worker=copy(engine)
    worker.state=deepcopy(snapshot['state'])
    worker.backend=copy(engine.backend)
    worker.backend.meter=meter
    worker.backend.journal=engine.out/'async_validation'/str(snapshot['update'])/'transport_journal'
    worker.backend.journal.mkdir(parents=True,exist_ok=True)
    # set_sampling_path assigns only the cloned backend's immutable sampler.
    worker.backend.set_sampling_path(snapshot['pair']['sampling_path'])
    outputs=worker.sample_rows(worker.rows['validation'],'validation',snapshot['state']['attempt'],1)
    records=[outputs[r['row_id']][0][1]for r in worker.rows['validation']]
    packet=metric_values(records,worker.rows['validation'])
    dest=engine.out/'async_validation'/str(snapshot['update'])
    save(dest/'records.json',records);save(dest/'all_metrics.json',packet)
    if all(v['status']in('settled','cancelled')for v in meter.calls.values()):meter.close()
    else:save(dest/'unknown_holds.json',{'cohort_reservation_retained':True,'unknown_physical_ids':[k for k,v in meter.calls.items()if v['status']not in('settled','cancelled')]})
    return {'complete':packet['complete'],'metrics':packet,'snapshot':snapshot}

def apply_result(engine, result):
    """Called by the training owner in increasing checkpoint order only."""
    packet=result['metrics'];snapshot=result['snapshot']
    if not packet['complete']:
        engine.state.setdefault('incomplete_validations',[]).append({'update':snapshot['update'],'reasons':packet.get('incomplete_reasons',[])})
        save(engine.path,engine.state)
        return # no vote, no false zero, retain pending pin for audit
    update=snapshot['update'];pair=snapshot['pair']
    prior=engine.state.get('patience_status')
    decision=observe(prior,packet['stop_metrics'],complete=True,scheduled=True,
                     baseline=update==0,delta=engine.config['min_delta'],checkpoint_update=update)
    engine.state['patience_status']=decision
    engine.state['metric_pins']=update_pins(engine.state.get('metric_pins'),pair['checkpoint'],packet['tracked_metrics'],packet['directions'],complete=True)
    engine.state['pair_registry'][pair['checkpoint']]=pair
    engine.state.setdefault('history',[]).append({'updates':update,'attempt':snapshot['state']['attempt'],'grid':update>0,'reason':'async_validation','stop_metrics':packet['stop_metrics'],'patience_status':decision,'populations':packet['populations']})
    engine.state['evaluated_through_update']=update
    engine.state['validation_lag_updates']=engine.state['updates']-update
    engine.state.setdefault('pending_validation_pairs',{}).pop(str(update),None)
    if decision['stop']:engine.state.setdefault('stop_decision_update',update)
    save(engine.path,engine.state)
