"""Full560 secondary policy comparisons; no endpoint or domain selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import json
from pathlib import Path
import sys
import time
import numpy as np
from receiver import receive,load,sha,P
from secondary_metrics import array,calculate,NAMES
ROOT=next(x for x in P.parents if (x/'output/pdf/qwen_reward_extension_20261116').is_dir())
sys.path.insert(0,str(ROOT/'output/pdf/qwen_reward_extension_20261116'))
from snapshot_fixed_metrics import summarize_records
DATA={};ROSTER={};PANEL={};PLAN={};COVERAGE={}

def init(config):
    global DATA,ROSTER,PANEL,PLAN,COVERAGE
    DATA,COVERAGE,ROSTER,PANEL,_=receive(config);PLAN=load(P/'SECONDARY_REMAINING14_PLAN_FROZEN.json')
    assert len(DATA)==10668 and all(r.get('available',True)for r in DATA.values())
    assert set(PLAN['metrics'])==set(NAMES)
    for path,h in PLAN['bindings'].items():assert sha(path)==h

def evaluate(arrays,weights,swaps=None):
    levels=[]
    for level in (0,2):
        changes=[]
        for seed,(a,b)in arrays[level].items():
            if swaps is None:aa,bb=a,b;wa=wb=weights
            else:
                aa=bb=np.concatenate([a,b]);wa=np.concatenate([weights*(~swaps),weights*swaps],axis=1);wb=np.concatenate([weights*swaps,weights*(~swaps)],axis=1)
            changes.append(calculate(bb,wb)-calculate(aa,wa))
        levels.append(np.mean(np.stack(changes),axis=0))
    return np.stack([levels[1],levels[1]-levels[0]],axis=1)

def compute(task):
    family,target,contrast,seeds=task;hi,lo=contrast.split('-');jobs={(j['target'],j['arm'],j['seed']):j for j in ROSTER[family]};arrays={0:{},2:{}};checks=0
    triplets=PANEL['triplets'];domains=np.array([t['domain']for t in triplets]);strata=[np.flatnonzero(domains==d)for d in sorted(set(domains))]
    assert len(strata)==7 and all(len(x)==4 for x in strata)
    levels={arm:{}for arm in(lo,hi)}
    for level in(0,2):
        for seed in seeds:
            raw={arm:[DATA[family,jobs[target,arm,seed]['sampling_path'],t['levels'][level]['row_id']]for t in triplets]for arm in(lo,hi)}
            values={arm:array(rs)for arm,rs in raw.items()};arrays[level][seed]=(values[lo],values[hi])
            # Necessary numerical replay includes actual bootstrapped and label-swapped populations.
            rng=np.random.default_rng(18301);weights=[np.ones(28,dtype=int),rng.multinomial(28,np.full(28,1/28))]
            for arm in(lo,hi):
                point=summarize_records(raw[arm])['metrics'];levels[arm][seed,level]=point
                for w in weights:
                    actual=calculate(values[arm],w[None,:])[0];expected=summarize_records([r for r,n in zip(raw[arm],w)for _ in range(int(n))])['metrics']
                    for mi,m in enumerate(NAMES):
                        ref=expected[m];assert(ref is None and np.isnan(actual[mi]))or(ref is not None and abs(ref-actual[mi])<2e-10),(family,target,contrast,seed,level,m,ref,actual[mi]);checks+=1
    cfg=PLAN['inference'];rng=np.random.default_rng(cfg['bootstrap_seed']);weights=np.zeros((1999,28),dtype=int)
    for rep in range(1999):
        for ix in strata:weights[rep,ix]=rng.multinomial(4,np.full(4,.25))
    point=evaluate(arrays,np.ones((1,28),int))[0];boot=evaluate(arrays,weights)
    rng=np.random.default_rng(cfg['permutation_seed']);swaps=rng.integers(0,2,(19999,28),dtype=np.int8).astype(bool)
    null=[]
    for start in range(0,19999,500):
        sw=swaps[start:start+500];null.append(evaluate(arrays,np.ones(sw.shape,int),sw))
    null=np.concatenate(null);results=[]
    for endpoint,ei in [('L2_direct',0),('L2_minus_L0_interaction',1)]:
        for mi,m in enumerate(NAMES):
            v=point[ei,mi];bb=boot[:,ei,mi];nn=null[:,ei,mi];bg=np.isfinite(bb);ng=np.isfinite(nn)
            available=np.isfinite(v)and bg.sum()>=.95*len(bb)and ng.sum()>=.95*len(nn)
            p=(1+int(np.sum(np.abs(nn[ng])>=abs(v)-1e-14))+len(nn)-int(ng.sum()))/(len(nn)+1)if available else 1.
            results.append(dict(scope=dict(family=family,target=target,contrast=contrast,paired_seeds=seeds,metric=m,endpoint=endpoint),
                test=dict(difference=float(v)if np.isfinite(v)else None,percentile95=np.quantile(bb[bg],[.025,.975]).tolist()if available else None,raw_p=p,available=bool(available),defined_bootstraps=int(bg.sum()),bootstrap_replicates=len(bb),defined_null_replicates=int(ng.sum()),null_replicates=len(nn),undefined_null_counted_as_extreme=len(nn)-int(ng.sum())),
                levels={arm:{str(level):float(np.mean([levels[arm][s,level][m]for s in seeds]))if all(levels[arm][s,level][m]is not None for s in seeds)else None for level in(0,2)}for arm in(lo,hi)},
                conversation_count=28,rows_per_checkpoint=84,canonical_parity_checks=checks))
    return results

def main(args):
    init(args.config);unique={(s['family'],s['target'],s['contrast']):s for s in PLAN['slots']};tasks=[(*k,s['paired_seeds'])for k,s in unique.items()];results=[];args.workunits.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(args.config,))as pool:
        futures={pool.submit(compute,t):t for t in tasks}
        for f in as_completed(futures):
            rows=f.result();results+=rows;task=futures[f];path=args.workunits/('__'.join(task[:3])+'.json');path.write_text(json.dumps(rows,allow_nan=False));print(json.dumps({'completed_scope':task[:3]}),flush=True)
    assert len(results)==560
    from analyze28 import correct
    correct(results)
    out=dict(schema='policy28_remaining14_secondary560_results_v1',unix=time.time(),results=results,coverage=COVERAGE,
             bindings={**COVERAGE['bindings'],str(P/'SECONDARY_REMAINING14_PLAN_FROZEN.json'):sha(P/'SECONDARY_REMAINING14_PLAN_FROZEN.json'),str(Path(__file__)):sha(__file__),str(P/'secondary_metrics.py'):sha(P/'secondary_metrics.py')},
             secondary_after_partial_descriptives=True,primary80_and64_unchanged=True,provider_calls=0)
    with args.output.open('xb')as f:f.write(gzip.compress(json.dumps(out,sort_keys=True,allow_nan=False).encode(),mtime=0))
    print(json.dumps({'path':str(args.output),'sha256':sha(args.output),'tests':560}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workunits',type=Path,required=True);p.add_argument('--workers',type=int,default=4);main(p.parse_args())
