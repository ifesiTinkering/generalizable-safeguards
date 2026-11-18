"""Separate common926 sensitivity: unchanged six-metric inference functions."""
from pathlib import Path
import sys,json,gzip,itertools,time,argparse,math
from concurrent.futures import ProcessPoolExecutor,as_completed
D=Path(__file__).resolve().parent;O=D.parent;sys.path.insert(0,str(O))
from analyze_full_panel_arrays import load,sha,run_group,corrections,NAMES
def dump(p,x):
 with p.open('xb')as f:f.write(gzip.compress(json.dumps(x,sort_keys=True,allow_nan=False).encode(),mtime=0))
def seeds(family,target,contrast):
 if family=='GPT':return [2,4,5]
 if target=='iou':return [0,1,2,3,4,5,7,36,42]
 return {'B-A':[0,1,2,3],'C-B':[1,2,3,4,5,36,42],'D-B':[0,1,2,3,4,5,36,42]}[contrast]
def avg(v):
 if not v or any(x is None for x in v):return None
 if isinstance(v[0],list):return [avg([x[i]for x in v])for i in range(len(v[0]))]
 return sum(v)/len(v)
def delta(b,a):
 if a is None or b is None:return None
 return [delta(y,x)for x,y in zip(a,b)]if isinstance(a,list)else b-a
def point_tables():
 results=[]
 for family in('Qwen','GPT'):
  data=load(D/(family.upper()+'_ARRAYS.json.gz'));cells={tuple(c['identity'][k]for k in('target','arm','seed','mode','cell')):c for c in data['fixed24_cells']}
  for target,source,mode,contrast in itertools.product(('binary','iou'),('safepyramid','dynabench'),('OFF','TWO_STAGE_ON'),('B-A','C-B','D-B')):
   b,a=contrast.split('-');ss=seeds(family,target,contrast);domains=['content_moderation','specialized_advice','intellectual_property'];required=[(target,arm,s,mode,source+':'+d)for arm,s,d in itertools.product((a,b),ss,domains)];assert all(k in cells for k in required)
   names=list(cells[required[0]]['summary']['metrics']);arms={};perseed={}
   for arm in(a,b):arms[arm]={m:avg([cells[target,arm,s,mode,source+':'+d]['summary']['metrics'][m]for s,d in itertools.product(ss,domains)])for m in names}
   for s in ss:perseed[str(s)]={m:avg([delta(cells[target,b,s,mode,source+':'+d]['summary']['metrics'][m],cells[target,a,s,mode,source+':'+d]['summary']['metrics'][m])for d in domains])for m in names}
   results.append({'family':family,'target':target,'source':source,'mode':mode,'regime':'OOD','contrast':contrast,'paired_seeds':ss,'arms':arms,'difference':{m:avg([perseed[str(s)][m]for s in ss])for m in names},'per_seed_difference':perseed,'rows_per_checkpoint':328 if source=='safepyramid'else 598,'responses_per_checkpoint':984 if source=='safepyramid'else 1794,'aggregation':'Equal three-domain mean; fixed paired seeds; full known retained row/draw populations. Semantic metrics null unless every required record known.'})
 out={'schema':'common926-fixed24-descriptive-v1','provider_calls':0,'plan_sha256':sha(D/'PLAN.json'),'source_sha256':sha(__file__),'source_bindings':{str(D/(f+'_ARRAYS.json.gz')):sha(D/(f+'_ARRAYS.json.gz'))for f in('QWEN','GPT')},'results':results};p=D/'FIXED24_POINTS.json.gz';dump(p,out);print(json.dumps({'points':str(p),'sha256':sha(p),'comparisons':len(results)}),flush=True)
def calc(spec):
 family,target,source,mode,regime,contrast=spec;b,a=contrast.split('-');data=load(D/(family.upper()+'_ARRAYS.json.gz'));cells={}
 for c in data['cells']:
  i=c['identity'];arm=i['arm']
  if arm in(a,b):cells[i['target'],'A'if arm==a else'B',i['seed'],i['mode'],i['cell']]=c
 r=run_group(cells,family,target,mode,source,regime,seeds(family,target,contrast));r['contrast']=contrast;r['original_arms']={'A':a,'B':b};return r
def adjust(results,total,semantic=False):
 slots=[r['metrics'][m]for r in results for m in NAMES]
 if semantic:
  for r in results:
   for m in('supported_predicted_fraction','all_predicted_rules_supported'):
    r['metrics'][m]={'difference':None,'percentile95':None,'raw_p':1.,'available':False,'reason':'Semantic inference not yet available in this core-only sensitivity cut'};slots.append(r['metrics'][m])
 assert len(slots)==total
 order=sorted(range(total),key=lambda i:slots[i]['raw_p']);running=1.
 for rank in range(total-1,-1,-1):
  i=order[rank];running=min(running,slots[i]['raw_p']*total/(rank+1));slots[i]['BH_q']=running
 running=0.
 for rank,i in enumerate(order):running=max(running,min(1.,slots[i]['raw_p']*(total-rank)));slots[i]['Holm_p']=running
def main():
 p=argparse.ArgumentParser();p.add_argument('--points-only',action='store_true');p.add_argument('--family',choices=['calibration','recipe'],default='calibration');p.add_argument('--workers',type=int,default=6);a=p.parse_args()
 if a.points_only:point_tables();return
 contrasts=['B-A']if a.family=='calibration'else['C-B','D-B'];specs=list(itertools.product(('Qwen','GPT'),('binary','iou'),('safepyramid','dynabench'),('OFF','TWO_STAGE_ON'),('ID','OOD'),contrasts));results=[]
 with ProcessPoolExecutor(max_workers=a.workers)as pool:
  futures={}
  for s in specs:
   path=D/('scope_'+'.'.join(s)+'.json.gz')
   if path.exists():
    z=load(path);assert z['plan_sha256']==sha(D/'PLAN.json');assert z['source_bindings']=={str(D/(f+'_ARRAYS.json.gz')):sha(D/(f+'_ARRAYS.json.gz'))for f in('QWEN','GPT')};results.append(z['result'])
   else:futures[pool.submit(calc,s)]=(s,path)
  for f in as_completed(futures):
   s,path=futures[f];r=f.result();results.append(r);dump(path,{'result':r,'plan_sha256':sha(D/'PLAN.json'),'inference_source_sha256':sha(O/'analyze_full_panel_arrays.py'),'source_bindings':{str(D/(f+'_ARRAYS.json.gz')):sha(D/(f+'_ARRAYS.json.gz'))for f in('QWEN','GPT')}});print(json.dumps({'finished':s,'complete':r['complete']}),flush=True)
 results.sort(key=lambda r:(r['family'],r['target'],r['source'],r['mode'],r['regime'],r['contrast']));total=192 if a.family=='calibration'else 512;adjust(results,total,a.family=='recipe')
 out={'schema':'common926-separate-sensitivity-inference-v1','provider_calls':0,'family_size':total,'analysis':'calibration B-A'if a.family=='calibration'else'secondary recipe C-B/D-B','results':results,'plan_sha256':sha(D/'PLAN.json'),'source_sha256':sha(__file__),'inference_source_sha256':sha(O/'analyze_full_panel_arrays.py'),'metric_source_sha256':sha(O/'weighted_panel_metrics.py'),'source_bindings':{str(D/(f+'_ARRAYS.json.gz')):sha(D/(f+'_ARRAYS.json.gz'))for f in('QWEN','GPT')},'caution':'Separate uniform common-question missing-data sensitivity, conditional on retained926 questions and fixed checkpoints. Original primary192/384 results unchanged. C6 binary C2/C4 are interim saved checkpoints.'};path=D/(a.family.upper()+'_INFERENCE.json.gz');dump(path,out);print(json.dumps({'path':str(path),'sha256':sha(path)}),flush=True)
if __name__=='__main__':main()
