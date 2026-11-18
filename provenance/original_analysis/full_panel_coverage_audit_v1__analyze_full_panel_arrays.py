"""Prospective paired conversation inference, fixed six-metric192-slot family.

Consumes immutable compact arrays only. Produces strict equal-domain estimates;
incomplete primary cohorts are never silently changed to complete-case cohorts.
"""
import argparse,collections,gzip,hashlib,itertools,json,math,time
from pathlib import Path
import numpy as np
from weighted_panel_metrics import calculate,NAMES
O=Path(__file__).resolve().parent
def load(p):
    p=Path(p);b=p.read_bytes();return json.loads(gzip.decompress(b)if p.suffix=='.gz'else b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean(v):return float(np.mean(v))if v and all(x is not None and math.isfinite(x)for x in v)else None
def diff(b,a):return b-a if b is not None and a is not None else None
def interval(v):
    good=[x for x in v if x is not None and math.isfinite(x)]
    return {'percentile95':np.quantile(good,[.025,.975]).tolist()if len(good)>=.95*len(v)else None,'defined_replicates':len(good),'total_replicates':len(v)}
def evaluate_pair(pairs,weights=None,swaps=None):
    result={}
    for (seed,domain),item in pairs.items():
        a,b,ix=item
        if swaps is not None:
            mask=swaps[ix,None]; aa=np.where(mask,b,a);bb=np.where(mask,a,b)
        else:aa,bb=a,b
        w=weights[ix]if weights is not None else None
        ma,mb=calculate(aa,w),calculate(bb,w)
        result[seed,domain]={'A':ma,'B':mb}
    seeds=sorted({s for s,d in result});domains=sorted({d for s,d in result})
    per_seed={str(s):{m:mean([diff(result[s,d]['B'][m],result[s,d]['A'][m])for d in domains])for m in NAMES}for s in seeds}
    arms={arm:{m:mean([result[s,d][arm][m]for s,d in result])for m in NAMES}for arm in 'AB'}
    return {'arms':arms,'B_minus_A':{m:mean([per_seed[str(s)][m]for s in seeds])for m in NAMES},'per_seed_B_minus_A':per_seed}
def run_group(cells,family,target,mode,source,regime,seeds,reps=1999):
    domains=['content_moderation','specialized_advice','intellectual_property']if regime=='OOD'else['privacy','fraud','defamation']
    required=[(target,arm,s,mode,source+':'+d)for arm,s,d in itertools.product('AB',seeds,domains)]
    absent=[k for k in required if k not in cells]
    base={'family':family,'target':target,'mode':mode,'source':source,'regime':regime,'paired_seeds':seeds,'domain_count':3,'missing_cells':absent}
    if absent:return dict(base,complete=False,reason='Exact planned arm/seed/domain cells not all known',metrics={m:{'difference':None,'percentile95':None,'raw_p':1.,'available':False}for m in NAMES})
    conversations=sorted({x for k in required for x in cells[k]['conversation_ids']});ci={x:i for i,x in enumerate(conversations)};pairs={}
    for s,d in itertools.product(seeds,domains):
        a,b=[cells[target,arm,s,mode,source+':'+d]for arm in 'AB']
        if(a['row_ids'],a['draws'],a['conversation_ids'])!=(b['row_ids'],b['draws'],b['conversation_ids']):raise RuntimeError('Arm task-set mismatch')
        pairs[s,d]=(np.asarray(a['values']),np.asarray(b['values']),np.asarray([ci[x]for x in a['conversation_ids']]))
    point=evaluate_pair(pairs);rng=np.random.default_rng(816731);boot={m:[]for m in NAMES};null={m:[]for m in NAMES}
    for _ in range(reps):
        w=rng.multinomial(len(conversations),np.full(len(conversations),1/len(conversations)))
        z=evaluate_pair(pairs,w)['B_minus_A']
        for m in NAMES:boot[m].append(z[m])
    for _ in range(reps):
        swaps=rng.integers(0,2,len(conversations),dtype=np.int8).astype(bool)
        z=evaluate_pair(pairs,swaps=swaps)['B_minus_A']
        for m in NAMES:null[m].append(z[m])
    metrics={}
    for m in NAMES:
        v=point['B_minus_A'][m];known=[x for x in null[m]if x is not None and math.isfinite(x)]
        # Undefined null draws count as nonrejections, conservatively; the slot
        # remains unavailable if fewer than95% are defined.
        available=v is not None and len(known)>=.95*reps
        p=(1+sum(abs(x)>=abs(v)-1e-14 for x in known)+(reps-len(known)))/(reps+1)if available else 1.
        metrics[m]={'difference':v,**interval(boot[m]),'raw_p':p,'available':available,'defined_permutations':len(known)}
    return dict(base,complete=True,conversations=len(conversations),rows_per_checkpoint=len({r for d in domains for r in cells[target,'A',seeds[0],mode,source+':'+d]['row_ids']}),responses_per_checkpoint=sum(len(cells[target,'A',seeds[0],mode,source+':'+d]['values'])for d in domains),point=point,metrics=metrics)
def corrections(results):
    slots=[r['metrics'][m]for r in results for m in NAMES];assert len(slots)==192
    order=sorted(range(192),key=lambda i:slots[i]['raw_p']);bh=[1.]*192;holm=[1.]*192
    running=1.
    for rank in range(191,-1,-1):
        i=order[rank];running=min(running,slots[i]['raw_p']*192/(rank+1));bh[i]=running
    running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,slots[i]['raw_p']*(192-rank)));holm[i]=running
    for i,s in enumerate(slots):s.update(BH_q=bh[i],Holm_p=holm[i])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qwen',type=Path,default=O/'CURRENT_QWEN930_ARRAYS.json.gz');ap.add_argument('--gpt',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--replicates',type=int,default=1999);a=ap.parse_args()
    if a.replicates!=1999:raise ValueError('Frozen1999 required')
    data={'Qwen':load(a.qwen),'GPT':load(a.gpt)if a.gpt else {'cells':[]}};results=[]
    for family,target,source,mode,regime in itertools.product(['Qwen','GPT'],['binary','iou'],['safepyramid','dynabench'],['OFF','TWO_STAGE_ON'],['ID','OOD']):
        cells={tuple(c['identity'][k]for k in('target','arm','seed','mode','cell')):c for c in data[family]['cells']}
        seeds=([0,1,2,3]if target=='binary'else[0,1,2,3,4,5,7,36,42])if family=='Qwen'else[2,4,5]
        r=run_group(cells,family,target,mode,source,regime,seeds);results.append(r)
        print(json.dumps({'completed_group':[family,target,source,mode,regime],'complete':r['complete']}),flush=True)
    corrections(results)
    out={'schema':'fixed192-full-panel-conversation-inference-v1','provider_calls':0,'observed_unix':time.time(),'analysis_plan_path':str(O/'ROOT_APPROVED_FULL_PANEL_ANALYSIS_PLAN.json'),'analysis_plan_sha256':sha(O/'ROOT_APPROVED_FULL_PANEL_ANALYSIS_PLAN.json'),'script_sha256':sha(__file__),'metric_source_sha256':sha(O/'weighted_panel_metrics.py'),'source_bindings':{str(p):sha(p)for p in [a.qwen,a.gpt]if p},'family_size':192,'results':results,'scope':'Conditional on fixed trained checkpoints; equal-weight three-domain means, paired conversation resampling and arm-label permutations. Unavailable slots retain p=1. Original seed-level and Holm8 families unchanged.'}
    with a.output.open('xb')as f:f.write(gzip.compress(json.dumps(out,sort_keys=True,allow_nan=False).encode(),mtime=0))
    print(json.dumps({'path':str(a.output),'sha256':sha(a.output)}))
if __name__=='__main__':main()
