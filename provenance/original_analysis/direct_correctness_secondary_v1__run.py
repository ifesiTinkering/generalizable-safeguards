"""Post-hoc uniform direct IoU-binary family; fixed models, conversation inference."""
import collections,gzip,hashlib,itertools,json,os,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;T=P.parent;B=T.parent.parent
N=B/'evaluation/thinking_on_extension_v1/full_panel_coverage_audit_v1/common926_v1';C=T/'Crefresh_analysis_v1'
sys.path.insert(0,str(T/'policy_triplets28_v3'))
import secondary_metrics as sm
METRICS=('rule_F1','Jaccard_IoU')+tuple(m for m in sm.NAMES if not m.startswith('supported_'))
assert len(METRICS)==14
Q=N/'QWEN_ARRAYS.json.gz';G=C/'GPT_OOD_SEMANTIC_ARRAYS.json.gz';GC=N/'GPT_COMPACT.json.gz';GN=C/'OOD_NEW_RECORDS.json.gz';ID=C/'ID400_JOINED_SEMANTIC_Crefresh_V1.json.gz';IP=T/'ID400_calibration_v1/SOURCE_PANEL.json'
SOURCES=[Q,G,GC,GN,ID,IP,N/'PLAN.json',Path(sm.__file__),T/'ID400_calibration_v1/batched_fixed_metrics.py',Path(__file__)]
def rd(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):p.write_bytes(gzip.compress(json.dumps(o,sort_keys=True,allow_nan=False).encode(),mtime=0))
def calculate(a,w):
    base=sm.calculate(a,w);n=w.sum(1);safe=np.where(n>0,n,1)
    f=(w@(a[:,0]*a[:,2]))/safe;j=(w@(a[:,0]*a[:,3]))/safe
    f[n==0]=np.nan;j[n==0]=np.nan
    return np.column_stack([f,j]+[base[:,sm.NAMES.index(m)] for m in METRICS[2:]])
def make(v,p,g):
    valid,q,f,j,e=v;tp=round(f*(p+g)/2) if valid else 0
    assert 0<=tp<=min(p,g)
    precision=tp/p if valid and p else float(valid and g==0)
    recall=tp/g if valid and g else float(valid and p==0)
    return [valid,q,f,j,e,None,None,precision,recall]
def prepare():
    plan={'schema':'direct-correctness-secondary1008-plan-v1','frozen_unix':time.time(),'family_size':1008,'metric_count':14,'metrics':METRICS,
          'scope':'3 within-model target contrasts (Qwen IoU-binary, Qwen F1-binary, GPT IoU-binary) x4arms x6available source/regime/mode cells x14nonsemantic metrics. Exact common eligible seeds per arm.',
          'secondary_after_descriptive_inspection':True,'all_prior_families_unchanged':True,'replicates':1999,
          'null':'One random label swap per unique conversation, shared across all domains/rows/draws/seeds of that conversation; recompute nonlinear metrics. Fixed checkpoints.',
          'bootstrap':'1999 global conversation-cluster bootstrap draws within each source/scope, preserving cross-domain duplicates. Equal domain then seed means.',
          'missing':'Undefined point or fewer than95% defined nulls retains rawp1 in full1008; undefined nulls counted as extreme. Evidence fraction/count excluded from inference, no missing-zero imputation.',
          'confidence':'Own-valid primary metric; common-valid descriptive sensitivity recomputed on paired intersection. No secondary test chosen from sensitivity.',
          'provider_calls':0,'source_bindings':{str(f.resolve()):sha(f) for f in SOURCES}}
    pf=P/'PLAN_FROZEN.json'
    if not pf.exists():pf.write_text(json.dumps(plan,indent=2)+'\n')
    else:assert rd(pf)['family_size']==1008
    ood={r['row_id']:r for r in rd(N/'PLAN.json')['retained_rows']};idrows={r['row_id']:r for r in rd(IP)['rows']};cells={};canonical={}
    for c in rd(Q)['cells']:
        i=c['identity'];key=('Qwen','OOD',i['target'],i['arm'],i['seed'],i['mode'],i['cell']);a=[]
        for rid,v,aux in zip(c['row_ids'],c['values'],c['aux_values']):a.append(make(v,aux[0],len(ood[rid]['gold_rule_ids'])))
        cells[key]=(a,list(zip(c['row_ids'],c['draws'])),c['conversation_ids'])
    for c in rd(Q)['fixed24_cells']:
        i=c['identity'];canonical['Qwen','OOD',i['target'],i['arm'],i['seed'],i['mode'],i['cell']]=c['summary']['metrics']
    records={}
    for path in [GC,GN]:
        for r in rd(path)['records']:
            records[r['target'],r['arm'],r['seed'],r['mode'],r['row_id'],r['draw']]=r
    for c in rd(G)['cells']:
        i=c['identity']
        if i['target'] not in ('binary','iou'):continue
        key=('GPT','OOD',i['target'],i['arm'],i['seed'],i['mode'],i['cell']);a=[]
        for rid,draw,v in zip(c['row_ids'],c['draws'],c['values']):
            r=records[i['target'],i['arm'],i['seed'],i['mode'],rid,draw];a.append(make(v,r['predicted_count'] if r['valid'] else 0,len(ood[rid]['gold_rule_ids'])))
        cells[key]=(a,list(zip(c['row_ids'],c['draws'])),c['conversation_ids'])
    for c in rd(G)['fixed24_cells']:
        i=c['identity'];canonical['GPT','OOD',i['target'],i['arm'],i['seed'],i['mode'],i['cell']]=c['summary']['metrics']
    buckets=collections.defaultdict(list)
    for r in rd(ID)['records']:
        if r['target'] not in ('binary','iou','partial'):continue
        row=idrows[r['row_id']];buckets[r['family'],'ID',r['target'],r['arm'],r['seed'],'TWO_STAGE_ON',row['source']+':'+row['domain']].append(r)
    for k,rs in buckets.items():
        rs.sort(key=lambda r:r['row_id']);a=sm.array(rs).tolist()
        # Semantic columns are deliberately absent from the new inference family.
        for x in a:x[5]=x[6]=None
        cells[k]=(a,[(r['row_id'],0)for r in rs],[idrows[r['row_id']]['conversation_hash']for r in rs])
    checks=0
    for k,metrics in canonical.items():
        if k[2] not in ('binary','iou','partial'):continue
        a=np.asarray(cells[k][0],float);point=calculate(a,np.ones((1,len(a)),int))[0]
        for j,m in enumerate(METRICS):
            ref=metrics[m];assert (ref is None and not np.isfinite(point[j]))or(ref is not None and abs(ref-point[j])<2e-11),(k,m,ref,point[j]);checks+=1
    work=[]
    for (family,target),arm,source,regime,mode in itertools.product((('Qwen','iou'),('Qwen','partial'),('GPT','iou')),'ABCD',('safepyramid','dynabench'),('ID','OOD'),('OFF','TWO_STAGE_ON')):
        if regime=='ID' and mode=='OFF':continue
        domains=('privacy','fraud','defamation')if regime=='ID'else('content_moderation','specialized_advice','intellectual_property')
        seeds=sorted(set(k[4]for k in cells if k[:4]==(family,regime,'binary',arm))&set(k[4]for k in cells if k[:4]==(family,regime,target,arm)))
        required=[(family,regime,t,arm,s,mode,source+':'+d)for t,s,d in itertools.product(('binary',target),seeds,domains)];assert seeds and all(k in cells for k in required)
        conv=sorted({x for k in required for x in cells[k][2]});ci={x:i for i,x in enumerate(conv)};arrays=[]
        for s,d in itertools.product(seeds,domains):
            a,ids,cc=cells[family,regime,'binary',arm,s,mode,source+':'+d];b,bids,bc=cells[family,regime,target,arm,s,mode,source+':'+d];assert ids==bids and cc==bc
            arrays.append({'seed':s,'domain':d,'a':a,'b':b,'ix':[ci[x]for x in cc]})
        spec=dict(family=family,high_target=target,arm=arm,source=source,regime=regime,mode=mode,paired_seeds=seeds,conversations=len(conv),arrays=arrays)
        p=P/('work_%02d.json.gz'%len(work));save(p,spec);work.append(str(p))
    (P/'PREPARED.json').write_text(json.dumps({'work':work,'canonical_OOD_point_checks':checks,'plan_sha256':sha(pf)},indent=2)+'\n')
    print(json.dumps({'prepared':len(work),'canonical_checks':checks}),flush=True)
def evaluate(arrays,weights,swaps=None):
    lo=[];hi=[]
    for a,b,ix in arrays:
        w=weights[:,ix]
        if swaps is None:aa,bb,aw,bw=a,b,w,w
        else:
            s=swaps[:,ix];aa=bb=np.concatenate([a,b]);aw=np.concatenate([w*(~s),w*s],1);bw=np.concatenate([w*s,w*(~s)],1)
        lo.append(calculate(aa,aw));hi.append(calculate(bb,bw))
    return np.mean(lo,0),np.mean(hi,0)
def finite(x):return float(x)if np.isfinite(x)else None
def run(path):
    spec=rd(Path(path));arrays=[(np.asarray(x['a'],float),np.asarray(x['b'],float),np.asarray(x['ix']))for x in spec.pop('arrays')];n=spec['conversations'];ones=np.ones((1,n),int);lo,hi=evaluate(arrays,ones);point=(hi-lo)[0]
    common=[]
    for a,b,ix in arrays:
        a=a.copy();b=b.copy();valid=(a[:,0]==1)&(b[:,0]==1);a[:,0]=valid;b[:,0]=valid;common.append((a,b,ix))
    clo,chi=evaluate(common,ones)
    brng=np.random.default_rng(773414);srng=np.random.default_rng(112709);boot=[];null=[]
    for start in range(0,1999,128):
        count=min(128,1999-start);w=brng.multinomial(n,np.full(n,1/n),size=count);l,h=evaluate(arrays,w);boot.append(h-l)
        swap=srng.integers(0,2,(count,n),dtype=np.int8).astype(bool);l,h=evaluate(arrays,np.ones((count,n),int),swap);null.append(h-l)
    boot=np.concatenate(boot);null=np.concatenate(null);tests={}
    for j,m in enumerate(METRICS):
        bb=boot[:,j];nn=null[:,j];bn=np.isfinite(bb);known=np.isfinite(nn);available=np.isfinite(point[j])and known.sum()>=.95*1999
        p=(1+np.sum(np.abs(nn[known])>=abs(point[j])-1e-14)+np.sum(~known))/2000 if available else 1.
        tests[m]=dict(binary=finite(lo[0,j]),high=finite(hi[0,j]),difference=finite(point[j]),raw_p=float(p),available=bool(available),defined_null=int(known.sum()),defined_bootstrap=int(bn.sum()),percentile95=np.quantile(bb[bn],[.025,.975]).tolist()if bn.sum()>=.95*1999 else None,
                      common_valid_descriptive={'binary':finite(clo[0,j]),'high':finite(chi[0,j]),'difference':finite((chi-clo)[0,j])})
    out=dict(spec,tests=tests);save(Path(path).with_name(Path(path).name.replace('work_','result_')),out);return out
def execute():
    prep=json.loads((P/'PREPARED.json').read_text());out=[]
    with ProcessPoolExecutor(max_workers=8)as pool:
        fs={pool.submit(run,p):p for p in prep['work']}
        for f in as_completed(fs):
            r=f.result();out.append(r);print(json.dumps({'done':[r[k]for k in ['family','arm','source','regime','mode']],'count':len(out)}),flush=True)
    slots=[r['tests'][m]for r in out for m in METRICS];assert len(slots)==1008;order=sorted(range(1008),key=lambda i:slots[i]['raw_p']);v=1.
    for rank in range(1007,-1,-1):i=order[rank];v=min(v,slots[i]['raw_p']*1008/(rank+1));slots[i]['BH_q']=v
    v=0.
    for rank,i in enumerate(order):v=max(v,min(1.,slots[i]['raw_p']*(1008-rank)));slots[i]['Holm_p']=v
    final=dict(schema='direct-correctness-secondary1008-results-v1',family_size=1008,metrics=METRICS,provider_calls=0,plan_sha256=sha(P/'PLAN_FROZEN.json'),source_sha256=sha(Path(__file__)),results=out,canonical_OOD_point_checks=prep['canonical_OOD_point_checks'])
    save(P/'RESULTS.json.gz',final);print(json.dumps({'finished':True,'sha256':sha(P/'RESULTS.json.gz')}),flush=True)
if __name__=='__main__':
    if sys.argv[-1]=='prepare':prepare()
    else:execute()
