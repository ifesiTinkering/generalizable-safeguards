"""ID400 fixed recipe320 and SFT320 analyses from immutable local compact exports."""
import argparse,collections,gzip,hashlib,itertools,json,math,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
H=P.parents[2];O=H/'evaluation/thinking_on_extension_v1/full_panel_coverage_audit_v1'
ROOT=H
while not (ROOT/'output/pdf/qwen_reward_extension_20261116').exists():ROOT=ROOT.parent
REPORT=ROOT/'output/pdf/qwen_reward_extension_20261116'
sys.path[:0]=[str(O),str(REPORT),str(P)]
from weighted_panel_metrics import calculate as core_calculate,array as core_array
from snapshot_fixed_metrics import summarize_records
from snapshot_readiness import build as coverage_check
DOMAINS=('defamation','fraud','privacy');METRICS=('rule_F1','IoU_ECE10','exact_error_AUROC','F1_AURC','F1_loss_capture_review20','confidence_F1_Spearman','supported_predicted_fraction','supported_prediction_count')
DATA={};PLAN={};ROWS={};ROSTER={};BINDINGS={};COVERAGE={}
def load(p):
 p=Path(p);return json.loads(gzip.decompress(p.read_bytes())if p.suffix=='.gz'else p.read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean(values):
 if not values or any(v is None for v in values):return None
 if isinstance(values[0],list):return[mean([x[i]for x in values])for i in range(len(values[0]))]
 return float(np.mean(values))if all(math.isfinite(v)for v in values)else None
def difference(hi,lo):
 if hi is None or lo is None:return None
 if isinstance(hi,list):return[difference(x,y)for x,y in zip(hi,lo)]
 return hi-lo
def calculate(a,weights=None):
 a=np.asarray(a,dtype=float);out=core_calculate(a[:,:5],weights);w=np.ones(len(a),dtype=int)if weights is None else np.asarray(weights);selected=w>0
 for k,col in [('supported_predicted_fraction',5),('supported_prediction_count',6)]:
  out[k]=float(np.sum(a[selected,col]*w[selected])/w.sum())if selected.any()and np.isfinite(a[selected,col]).all()else None
 return out
def init(config):
 global DATA,PLAN,ROWS,ROSTER,BINDINGS,COVERAGE
 PLAN=load(P/'ANALYSIS_PLAN_FROZEN.json');ROSTER=load(P/'FROZEN_ROSTER.json');panel=load(P/'SOURCE_PANEL.json');ROWS={r['row_id']:r for r in panel['rows']};DATA=collections.defaultdict(list);records=[];BINDINGS={}
 amendment=load(P/'SFT_ESTIMATOR_CLARIFICATION.json');assert amendment['approved_by_root']and amendment['original_plan_sha256']==sha(P/'ANALYSIS_PLAN_FROZEN.json')
 for spec in load(config)['snapshots']:
  p=Path(spec['path']);assert sha(p)==spec['sha256'];BINDINGS[str(p)]=sha(p);packet=load(p)
  for r in packet['records']:
   if 'family'not in r:r=dict(r,family=spec['family'])
   records.append(r)
 COVERAGE=coverage_check(records,ROSTER,panel,PLAN)
 # Coverage checker rejects conflicting aliases; keep the one canonical logical record.
 unique={(r['family'],r['sampling_path'],r['row_id']):r for r in records}
 for r in unique.values():
  row=ROWS[r['row_id']];DATA[r['family'],r['target'],r['arm'],r['seed'],row['source'],row['domain']].append(r)
 for rs in DATA.values():rs.sort(key=lambda r:r['row_id'])
def cell(f,target,arm,seed,source,domain):
 rs=DATA.get((f,target,arm,seed,source,domain),[]);expected=sorted(r for r,m in ROWS.items()if(m['source'],m['domain'])==(source,domain))
 if [r['row_id']for r in rs]!=expected or any(not r.get('available',True)for r in rs):return None
 return rs
def make_array(rs):
 core=core_array(rs);sem=np.array([[r.get('evidence_fraction'),r.get('supported_count',len(r['supported_rule_ids'])if r.get('supported_rule_ids')is not None else None)]for r in rs],dtype=float)
 return np.column_stack((core,sem))
def fullpoint(records_by_arm):
 result={}
 for arm,cells in records_by_arm.items():
  summaries={key:summarize_records(rs)for key,rs in cells.items()};names=next(iter(summaries.values()))['metrics'];perseed={}
  for seed in sorted({k[0]for k in cells}):perseed[str(seed)]={m:mean([summaries[seed,d]['metrics'][m]for d in DOMAINS])for m in names}
  result[arm]={'mean':{m:mean([v[m]for v in perseed.values()])for m in names},'per_seed':perseed,'domains':{str(k):v for k,v in summaries.items()}}
 return result
def prepare(slot,family_kind):
 f,t,source,c=[slot[k]for k in ('family','target','source','contrast')];hi,lo=c.split('-');seeds=slot.get('paired_seeds',slot.get('trained_seeds'));baseline=family_kind=='baseline_family';required={hi:{},lo:{}};missing=[]
 for arm in [hi,lo]:
  pairs=[(j['seed'],'SFT')for j in ROSTER[f]if j['target']=='SFT']if arm=='SFT'else[(s,t)for s in seeds]
  for s,target in pairs:
   for domain in DOMAINS:
    actual_arm=next(j['arm']for j in ROSTER[f]if j['target']=='SFT')if arm=='SFT'else arm
    rs=cell(f,target,actual_arm,s,source,domain)
    if rs is None:missing.append([target,arm,s,source,domain])
    else:required[arm][s,domain]=rs
 out={k:slot[k]for k in ('family','target','source','contrast','mode')};out.update(family_kind=family_kind,paired_seeds=seeds,complete=not missing,missing_cells=missing)
 if missing:return out,None,None
 cluster_ids=sorted({ROWS[r['row_id']]['conversation_hash']for cells in required.values()for rs in cells.values()for r in rs});lookup={c:i for i,c in enumerate(cluster_ids)};arrays={}
 base_seed=next(s for s,d in required[lo])if baseline else None
 for seed,domain in itertools.product(seeds,DOMAINS):
  rr=required[hi][seed,domain];ll=required[lo][base_seed if baseline else seed,domain];assert [r['row_id']for r in rr]==[r['row_id']for r in ll]
  arrays[seed,domain]=(make_array(ll),make_array(rr),np.array([lookup[ROWS[r['row_id']]['conversation_hash']]for r in rr]))
 full=fullpoint(required);full['difference']={m:difference(full[hi]['mean'][m],full[lo]['mean'][m])for m in full[hi]['mean']}
 out.update(conversation_count=len(cluster_ids),rows_per_checkpoint=sum(len(required[hi][seeds[0],d])for d in DOMAINS),responses_per_checkpoint=sum(len(required[hi][seeds[0],d])for d in DOMAINS),full24=full)
 # All domain and positive/negative controls are descriptive, without added tests.
 strata={}
 for label in ['positive','negative']:
  vals={}
  for arm,cells in required.items():
   summaries=[summarize_records([r for r in rs if ROWS[r['row_id']]['label']==label])for rs in cells.values()];names=next(iter(summaries))['metrics'];vals[arm]={m:mean([s['metrics'][m]for s in summaries])for m in names}
  strata[label]={'arms':vals,'difference':{m:difference(vals[hi][m],vals[lo][m])for m in vals[hi]}}
 out['descriptive_strata']=strata
 common={k:(a.copy(),b.copy(),ix)for k,(a,b,ix)in arrays.items()}
 for a,b,ix in common.values():mask=(a[:,0]==1)&(b[:,0]==1);a[:,0]=mask;b[:,0]=mask
 cv=evaluate(common)['difference'];out['common_valid_confidence_descriptive']={m:cv[m]for m in METRICS if m not in ('rule_F1','supported_predicted_fraction','supported_prediction_count')}
 return out,arrays,baseline
def evaluate(arrays,weights=None,swaps=None,baseline=False):
 low={};high={}
 for(seed,domain),(a,b,ix)in arrays.items():
  aa,bb=(np.where(swaps[ix,None],b,a),np.where(swaps[ix,None],a,b))if swaps is not None else(a,b)
  w=weights[ix]if weights is not None else None
  if not baseline or domain not in low:low[domain if baseline else(seed,domain)]=calculate(aa,w)
  high[seed,domain]=calculate(bb,w)
 levels={'low':{m:mean([x[m]for x in low.values()])for m in METRICS},'high':{m:mean([x[m]for x in high.values()])for m in METRICS}}
 return {'levels':levels,'difference':{m:difference(levels['high'][m],levels['low'][m])for m in METRICS}}
def interval(values):
 good=[x for x in values if x is not None and math.isfinite(x)];return {'percentile95':np.quantile(good,[.025,.975]).tolist()if len(good)>=.95*len(values)else None,'defined_bootstraps':len(good),'bootstrap_replicates':len(values)}
def compute(task):
 slot,family_kind,points_only=task;out,arrays,baseline=prepare(slot,family_kind)
 if arrays is None:return dict(out,metrics={m:{'difference':None,'percentile95':None,'raw_p':1.,'available':False,'reason':'Exact fixed source/checkpoint/domain outcome grid incomplete'}for m in METRICS})
 point=evaluate(arrays,baseline=baseline);out['point']=point
 for m in METRICS:
  a,b=point['difference'][m],out['full24']['difference'][m];assert(a is None and b is None)or(a is not None and b is not None and abs(a-b)<2e-11),(m,a,b)
 if points_only:return out
 n=out['conversation_count'];rng=np.random.default_rng(741920);boot={m:[]for m in METRICS};null={m:[]for m in METRICS}
 for _ in range(1999):
  w=rng.multinomial(n,np.full(n,1/n));z=evaluate(arrays,weights=w,baseline=baseline)['difference']
  for m in METRICS:boot[m].append(z[m])
 if not baseline:
  rng=np.random.default_rng(192011)
  for _ in range(1999):
   swap=rng.integers(0,2,n,dtype=np.int8).astype(bool);z=evaluate(arrays,swaps=swap)['difference']
   for m in METRICS:null[m].append(z[m])
 out['metrics']={}
 for m in METRICS:
  v=point['difference'][m];distribution=[x-v if x is not None and v is not None else None for x in boot[m]]if baseline else null[m];known=[x for x in distribution if x is not None and math.isfinite(x)];available=v is not None and len(known)>=.95*1999 and sum(x is not None and math.isfinite(x)for x in boot[m])>=.95*1999;p=(1+sum(abs(x)>=abs(v)-1e-14 for x in known)+1999-len(known))/2000 if available else 1.
  out['metrics'][m]={'difference':v,**interval(boot[m]),'raw_p':p,'available':available,'defined_null_replicates':len(known),'undefined_null_counted_as_extreme':1999-len(known),'null_method':'centered_paired_conversation_bootstrap'if baseline else'paired_conversation_arm_label_permutation'}
 return out
def adjust(results):
 for kind in ['recipe_family','baseline_family']:
  slots=[r['metrics'][m]for r in results if r['family_kind']==kind for m in METRICS];assert len(slots)==320;order=sorted(range(320),key=lambda i:slots[i]['raw_p']);run=1.
  for rank in range(319,-1,-1):i=order[rank];run=min(run,slots[i]['raw_p']*320/(rank+1));slots[i]['BH_q']=run
  run=0.
  for rank,i in enumerate(order):run=max(run,min(1.,slots[i]['raw_p']*(320-rank)));slots[i]['Holm_p']=run
def main(a):
 init(a.config);tasks=[]
 for kind in ['recipe_family','baseline_family']:
  seen={tuple(s[k]for k in ('family','target','source','contrast')):s for s in PLAN[kind]['slots']};assert len(seen)==40;tasks.extend((s,kind,a.points_only)for s in seen.values())
 results=[]
 if a.points_only:results=[compute(t)for t in tasks]
 else:
  with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(a.config,))as pool:
   futures={pool.submit(compute,t):t for t in tasks}
   for future in as_completed(futures):
    result=future.result();results.append(result);print(json.dumps({'completed':[result[k]for k in ('family','target','source','contrast')],'complete':result['complete']}),flush=True)
  adjust(results)
 out={'schema':'ID400-fixed320-plus320-results-v1','generated_unix':time.time(),'points_only':a.points_only,'source_sha256':sha(__file__),'bindings':{**BINDINGS,**{str((P/n).resolve()):sha(P/n)for n in ['ANALYSIS_PLAN_FROZEN.json','SFT_ESTIMATOR_CLARIFICATION.json','FROZEN_ROSTER.json','SOURCE_PANEL.json','snapshot_readiness.py']},str(O/'weighted_panel_metrics.py'):sha(O/'weighted_panel_metrics.py'),str(REPORT/'snapshot_fixed_metrics.py'):sha(REPORT/'snapshot_fixed_metrics.py')},'coverage':COVERAGE,'family_sizes':{'recipe':320,'SFT':320},'results':results,'provider_calls':0}
 with Path(a.output).open('xb')as f:f.write(gzip.compress(json.dumps(out,sort_keys=True,allow_nan=False).encode(),mtime=0))
 print(json.dumps({'output':str(a.output),'sha256':sha(a.output),'complete_scopes':sum(x['complete']for x in results)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--points-only',action='store_true');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();assert 1<=args.workers<=4;main(args)
