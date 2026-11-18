"""Read-only support identification bounds and conservative missing-grade sensitivity.

All unknown fractions stay unknown in ordinary point tables. Interval endpoints
are separate identification bounds, not imputations or ordinary estimates.
"""
from pathlib import Path
import sys,json,gzip,itertools,collections
import numpy as np
D=Path(__file__).resolve().parent;O=D.parent
sys.path.insert(0,str(D));sys.path.insert(0,str(O))
from analyze import avg,seeds
from analyze_full_panel_arrays import load,sha
NAMES=('supported_predicted_fraction','supported_prediction_count','all_predicted_rules_supported')
STAMP='1794879311'
DOMAINS=['content_moderation','specialized_advice','intellectual_property']
def dump(p,z):
 with p.open('xb')as f:f.write(gzip.compress(json.dumps(z,sort_keys=True,allow_nan=False).encode(),mtime=0))
 print(json.dumps({'path':str(p),'sha256':sha(p)}),flush=True)
def ident(c):return tuple(c['identity'][k]for k in('target','arm','seed','mode','cell'))
def values(n,f,a):
 assert n>=0 and int(n)==n
 if f is not None:assert 0<=f<=1 and abs(n*f-round(n*f))<1e-7
 lo=[0. if f is None else f,0. if f is None else f*n,0. if a is None else a]
 hi=[1. if f is None else f,float(n)if f is None else f*n,1. if a is None else a]
 return lo,hi
def build():
 plan=load(D/'PLAN.json');rows={r['row_id']:r for r in plan['retained_rows']}
 q=load(D/'QWEN_ARRAYS.json.gz');g=load(D/('GPT_SEMANTIC_ARRAYS_'+STAMP+'.json.gz'));base=load(D/'QWEN_SFT_ARRAYS.json.gz')
 rs=load(O/('GPT_FULL_GRADED_COMPACT_'+STAMP+'.json.gz'))['records'];gi=collections.defaultdict(dict)
 for r in rs:
  if r['row_id']in rows:gi[r['target'],None if r['target']=='SFT'else r['arm'],r['seed'],r['mode'],r['domain']][r['row_id'],r['draw']]=r
 cells={};bindings={str(D/p):sha(D/p)for p in('PLAN.json','QWEN_ARRAYS.json.gz','QWEN_SFT_ARRAYS.json.gz','GPT_SEMANTIC_ARRAYS_'+STAMP+'.json.gz')}
 for fam,data in [('Qwen',q),('GPT',g)]:
  for c in data['cells']:
   lo=[];hi=[]
   if fam=='Qwen':triples=c['aux_values']
   else:
    records=gi[ident(c)];triples=[(records[r,d]['predicted_count'],records[r,d].get('evidence_fraction'),records[r,d].get('evidence_all'))for r,d in zip(c['row_ids'],c['draws'])]
   for n,f,a in triples:l,h=values(n,f,a);lo.append(l);hi.append(h)
   cells[(fam,)+ident(c)]={k:c[k]for k in('identity','row_ids','draws','conversation_ids')};cells[(fam,)+ident(c)].update(lower=lo,upper=hi)
 E=D.parents[2];broot=E/'qwen_sft930_readonly_audit';br={}
 for mode,suffix in [('OFF','off_after_grade_v1/row_proofs.json.gz'),('TWO_STAGE_ON','on_join_v1/results/row_audit.json.gz')]:
  p=broot/suffix;proof=load(p);bindings[str(p)]=sha(p)
  for r in proof['row_proofs']if isinstance(proof,dict)else proof:
   if r['row_id']not in rows:continue
   m=r['normalized_metrics'];n=int(m['TP']+m['FP'])if m['valid']else 0
   br[mode,r['row_id'],r['replicate']]=values(n,m.get('semantic_fraction'),m.get('semantic_all'))
 for c in base['cells']:
  vals=[br[c['identity']['mode'],r,d]for r,d in zip(c['row_ids'],c['draws'])]
  cells[('Qwen',)+ident(c)]={k:c[k]for k in('identity','row_ids','draws','conversation_ids')};cells[('Qwen',)+ident(c)].update(lower=[v[0]for v in vals],upper=[v[1]for v in vals])
 return cells,bindings
def level(cells,f,t,a,ss,mode,source):
 selected=[cells[f,t,a,s,mode,source+':'+d]for s,d in itertools.product(ss,DOMAINS)]
 return {m:{bound:avg([float(np.mean(np.asarray(c[bound])[:,i]))for c in selected])for bound in('lower','upper')}for i,m in enumerate(NAMES)}
def difference(b,a):return{m:{'lower':b[m]['lower']-a[m]['upper'],'upper':b[m]['upper']-a[m]['lower']}for m in NAMES}
def companions(cells,bindings):
 p=D/('FIXED24_POINTS_SEMANTIC_'+STAMP+'.json.gz');z=load(p)
 for r in z['results']:
  f,t,mode,source=[r[k]for k in('family','target','mode','source')];b,a=r['contrast'].split('-');ss=r['paired_seeds'];levels={arm:level(cells,f,t,arm,ss,mode,source)for arm in(a,b)}
  r['identification_bounds']={'arms':levels,'difference':difference(levels[b],levels[a]),'per_seed_difference':{str(s):difference(level(cells,f,t,b,[s],mode,source),level(cells,f,t,a,[s],mode,source))for s in ss}}
  for arm in(a,b):
   for m in NAMES:
    if r['arms'][arm][m]is not None:assert abs(r['arms'][arm][m]-levels[arm][m]['lower'])<1e-10 and abs(levels[arm][m]['upper']-levels[arm][m]['lower'])<1e-10
 z.update(bound_note='Identification ranges, not confidence intervals. Unknown fraction/all support is bounded0..1; unknown supported count0..predicted count. Original points/nulls unchanged.',bounds_source_sha256=sha(__file__),bounds_source_bindings=bindings)
 dump(D/('FIXED24_POINTS_WITH_BOUNDS_'+STAMP+'.json.gz'),z)
 p=D/('SFT_POINTS_SEMANTIC_'+STAMP+'.json.gz');z=load(p)
 for r in z['results']:
  if not r['complete']:continue
  f,t,mode,source=[r[k]for k in('family','target','mode','source')];arm=r['contrast'].split('_')[0];rl=level(cells,f,t,arm,r['RL_seeds'],mode,source);sft=level(cells,f,'SFT',None,[None],mode,source)
  r['identification_bounds']={'fixed24':{'SFT':sft,'RL':rl,'RL_minus_SFT':difference(rl,sft)}}
 z.update(bound_note='Identification ranges, not confidence intervals; single SFT checkpoint, original point/null fields unchanged.',bounds_source_sha256=sha(__file__),bounds_source_bindings=bindings)
 dump(D/('SFT_POINTS_WITH_BOUNDS_'+STAMP+'.json.gz'),z)
def sensitivity(cells,f,t,source='safepyramid',mode='TWO_STAGE_ON',contrast='D-B',reps=1999):
 b,a=contrast.split('-');ss=seeds(f,t,contrast);req=[cells[f,t,arm,s,mode,source+':'+d]for arm,s,d in itertools.product((a,b),ss,DOMAINS)];convs=sorted({c for r in req for c in r['conversation_ids']});ci={c:i for i,c in enumerate(convs)};n=len(convs);lower=np.zeros((3,n,3));upper=lower.copy();den=np.zeros((3,n));per={}
 for di,d in enumerate(DOMAINS):
  for si,s in enumerate(ss):
   ac,bc=[cells[f,t,arm,s,mode,source+':'+d]for arm in(a,b)];assert(ac['row_ids'],ac['draws'],ac['conversation_ids'])==(bc['row_ids'],bc['draws'],bc['conversation_ids']);ix=np.array([ci[c]for c in ac['conversation_ids']]);lo=np.asarray(bc['lower'])-np.asarray(ac['upper']);hi=np.asarray(bc['upper'])-np.asarray(ac['lower'])
   for j in range(3):lower[di,:,j]+=np.bincount(ix,weights=lo[:,j],minlength=n)/len(ss);upper[di,:,j]+=np.bincount(ix,weights=hi[:,j],minlength=n)/len(ss)
   if si==0:den[di]=np.bincount(ix,minlength=n)
 for s in ss:per[str(s)]=difference(level(cells,f,t,b,[s],mode,source),level(cells,f,t,a,[s],mode,source))
 def evaluate(w):
  ds=den@w;assert np.all(ds>0)
  return np.mean(np.einsum('dcm,c->dm',lower,w)/ds[:,None],axis=0),np.mean(np.einsum('dcm,c->dm',upper,w)/ds[:,None],axis=0)
 point=evaluate(np.ones(n));rng=np.random.default_rng(816731);bootlo=[];boothi=[]
 for _ in range(reps):l,h=evaluate(rng.multinomial(n,np.full(n,1/n)));bootlo.append(l);boothi.append(h)
 # For every arm-label permutation, take the maximum absolute statistic over
 # ALL missing-value assignments. This upper-bounds each completion's p-value,
 # unlike simply testing an adverse point-imputed data set.
 cl=np.mean(lower/den.sum(axis=1)[:,None,None],axis=0);ch=np.mean(upper/den.sum(axis=1)[:,None,None],axis=0);center=(cl+ch)/2;radius=np.sum((ch-cl)/2,axis=0);nullmax=[]
 for _ in range(reps):sign=1-2*rng.integers(0,2,n,dtype=np.int8);nullmax.append(np.abs(sign@center)+radius)
 nullmax=np.asarray(nullmax);threshold=np.where(point[0]>0,point[0],np.where(point[1]<0,-point[1],0.));p=(1+np.sum(nullmax>=threshold[None,:]-1e-14,axis=0))/(reps+1)
 metrics={m:{'identification_bounds':[float(point[0][j]),float(point[1][j])],'bootstrap95_envelope':[float(np.quantile(np.asarray(bootlo)[:,j],.025)),float(np.quantile(np.asarray(boothi)[:,j],.975))],'conservative_permutation_p_upper':float(p[j]),'conservative_Bonferroni512_upper':min(1.,float(p[j])*512),'positive_seed_lower_bounds':sum(per[str(s)][m]['lower']>0 for s in ss),'per_seed_identification_bounds':{str(s):per[str(s)][m]for s in ss},'leave_one_seed_out_bounds':{str(s):{'lower':avg([per[str(o)][m]['lower']for o in ss if o!=s]),'upper':avg([per[str(o)][m]['upper']for o in ss if o!=s])}for s in ss}}for j,m in enumerate(NAMES)}
 return {'family':f,'target':t,'source':source,'mode':mode,'contrast':contrast,'paired_seeds':ss,'conversations':n,'rows_per_checkpoint':len({r for c in req for r in c['row_ids']}),'responses_per_checkpoint':int(den.sum()),'metrics':metrics}
def main():
 cells,bindings=build();companions(cells,bindings)
if __name__=='__main__':main()
