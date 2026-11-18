"""Complete frozen secondary support sensitivity; no runtime/provider imports."""
from pathlib import Path
import json,gzip,itertools,time,hashlib
import support_bounds_final as s
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((D/'ALL_SUPPORT_SECONDARY_PLAN.json').read_bytes());assert plan['evaluated_slots']==120
original_seeds=s.seeds
s.seeds=lambda family,target,contrast:[0,1,2,3,4,5,7,36,42]if family=='Qwen'and target=='partial'else original_seeds(family,target,contrast)
cells,bindings=s.build();results=[]
for f,t in [('Qwen','binary'),('Qwen','iou'),('GPT','binary'),('GPT','iou'),('Qwen','partial')]:
 for source,mode,c in itertools.product(plan['sources'],plan['modes'],plan['contrasts']):
  r=s.sensitivity(cells,f,t,source,mode,c);results.append(r);print(f,t,source,mode,c,flush=True)
slots=[x for r in results for x in r['metrics'].values()];assert len(slots)==120
p=[x['conservative_permutation_p_upper']for x in slots]+[1.]*392;order=sorted(range(512),key=lambda i:p[i]);q=[1.]*512;h=[1.]*512;running=1.
for rank in range(511,-1,-1):
 i=order[rank];running=min(running,p[i]*512/(rank+1));q[i]=running
running=0.
for rank,i in enumerate(order):running=max(running,min(1.,p[i]*(512-rank)));h[i]=running
for i,x in enumerate(slots):x.update(secondary_BH512_upper=q[i],secondary_Holm512_upper=h[i])
z={'schema':'all-evidence-support-secondary-sensitivity-v1','plan_path':str(D/'ALL_SUPPORT_SECONDARY_PLAN.json'),'plan_sha256':sha(D/'ALL_SUPPORT_SECONDARY_PLAN.json'),'source_sha256':sha(__file__),'method_source_sha256':sha(D/'support_bounds_final.py'),'source_bindings':bindings,'results':results,'evaluated_slots':120,'correction_family_size':512,'unavailable_or_unselected_metric_slots_p1':392,'provider_calls':0,'ordinary_points_unchanged':True,'original_inference_unchanged':True,'generated_unix':time.time()}
p=D/'ALL_SUPPORT_SECONDARY_RESULTS.json.gz';p.open('xb').write(gzip.compress(json.dumps(z,sort_keys=True,allow_nan=False).encode(),mtime=0));print('OUTPUT',p,sha(p),flush=True)
