"""Execution optimization only; frozen scalar estimator/RNG remains the reference."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import json
import math
from pathlib import Path
import time
import numpy as np
import analyze_frozen_v2 as a
from batched_fixed_metrics import evaluate_many,METRICS

def compute(task):
    slot,kind=task;out,arrays,baseline=a.prepare(slot,kind)
    if arrays is None:return a.compute((slot,kind,False))
    n=out['conversation_count'];point=a.evaluate(arrays,baseline=baseline);out['point']=point
    for m in METRICS:
        x,y=point['difference'][m],out['full24']['difference'][m]
        assert(x is None and y is None)or(x is not None and y is not None and abs(x-y)<2e-11)
    rng=np.random.default_rng(741920)
    weights=np.stack([rng.multinomial(n,np.full(n,1/n))for _ in range(1999)])
    if baseline:swaps=None
    else:
        rng=np.random.default_rng(192011)
        # Preserve per-call int8 RNG consumption exactly, including discarded buffers.
        swaps=np.stack([rng.integers(0,2,n,dtype=np.int8).astype(bool)for _ in range(1999)])
    boot=[];null=[]
    for start in range(0,1999,256):
        w=weights[start:start+256];boot.append(evaluate_many(arrays,w,baseline=baseline))
        if swaps is not None:null.append(evaluate_many(arrays,np.ones_like(w),swaps[start:start+256],baseline))
    boot=np.concatenate(boot);null=np.concatenate(null)if null else None
    # Actual-data parity checks include null and bootstrap cases on every scope.
    checks=0
    for ix in [0,1,11,198,1998]:
        for use_null in ([False,True]if not baseline else[False]):
            got=(null if use_null else boot)[ix]
            ref=a.evaluate(arrays,weights=np.ones(n,dtype=int)if use_null else weights[ix],swaps=swaps[ix]if use_null else None,baseline=baseline)['difference']
            for col,m in enumerate(METRICS):
                val=ref[m]
                assert(val is None and np.isnan(got[col]))or(val is not None and abs(val-got[col])<2e-11),(m,ix,val,got[col])
                checks+=1
    out['metrics']={}
    for col,m in enumerate(METRICS):
        v=point['difference'][m];bd=boot[:,col];dist=bd-v if baseline and v is not None else np.full(1999,np.nan)if baseline else null[:,col]
        known=np.isfinite(dist);defined_boot=np.isfinite(bd);available=v is not None and known.sum()>=.95*1999 and defined_boot.sum()>=.95*1999
        p=(1+int(np.sum(np.abs(dist[known])>=abs(v)-1e-14))+1999-int(known.sum()))/2000 if available else 1.
        out['metrics'][m]={'difference':v,'percentile95':np.quantile(bd[defined_boot],[.025,.975]).tolist()if defined_boot.sum()>=.95*1999 else None,'defined_bootstraps':int(defined_boot.sum()),'bootstrap_replicates':1999,'raw_p':p,'available':bool(available),'defined_null_replicates':int(known.sum()),'undefined_null_counted_as_extreme':1999-int(known.sum()),'null_method':'centered_paired_conversation_bootstrap'if baseline else'paired_conversation_arm_label_permutation'}
    out['batched_actual_data_scalar_checks']=checks
    return out

def main(args):
    a.init(args.config);tasks=[]
    for kind in ['recipe_family','baseline_family']:
        seen={tuple(s[k]for k in ('family','target','source','contrast')):s for s in a.PLAN[kind]['slots']}
        tasks.extend((s,kind)for s in seen.values())
    results=[];args.checkpoints.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=a.init,initargs=(args.config,))as pool:
        futures={pool.submit(compute,t):t for t in tasks}
        for future in as_completed(futures):
            r=future.result();results.append(r)
            name='__'.join(str(r[k])for k in ['family_kind','family','target','source','contrast'])+'.json'
            with(args.checkpoints/name).open('x')as f:json.dump(r,f,sort_keys=True,allow_nan=False)
            print(json.dumps({'done':len(results),'scope':[r[k]for k in ['family','target','source','contrast']]}),flush=True)
    a.adjust(results)
    out={'schema':'ID400-fixed320-plus320-results-v1','generated_unix':time.time(),'points_only':False,'source_sha256':a.sha(__file__),'reference_scalar_sha256':a.sha(a.P/'analyze_frozen_v2.py'),'batched_metric_sha256':a.sha(a.P/'batched_fixed_metrics.py'),'bindings':{**a.BINDINGS,**{str((a.P/n).resolve()):a.sha(a.P/n)for n in ['ANALYSIS_PLAN_FROZEN.json','SFT_ESTIMATOR_CLARIFICATION.json','FROZEN_ROSTER.json','SOURCE_PANEL.json','snapshot_readiness.py']},str(a.O/'weighted_panel_metrics.py'):a.sha(a.O/'weighted_panel_metrics.py'),str(a.REPORT/'snapshot_fixed_metrics.py'):a.sha(a.REPORT/'snapshot_fixed_metrics.py')},'coverage':a.COVERAGE,'family_sizes':{'recipe':320,'SFT':320},'results':results,'provider_calls':0,'optimization':'Batched evaluation of the same exact integer weights, same per-call RNG sequence and metric definitions; no changed hypotheses, data or methods.'}
    with args.output.open('xb')as f:f.write(gzip.compress(json.dumps(out,sort_keys=True,allow_nan=False).encode(),mtime=0))
    print(json.dumps({'path':str(args.output),'sha256':a.sha(args.output),'complete':sum(r['complete']for r in results),'scalar_parity_checks':sum(r.get('batched_actual_data_scalar_checks',0)for r in results)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--workers',type=int,default=4);main(p.parse_args())
