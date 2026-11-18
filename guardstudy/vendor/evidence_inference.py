"""Pure conservative support tests extracted from the frozen analysis."""
import itertools
import numpy as np
NAMES=('supported_predicted_fraction','supported_prediction_count','all_predicted_rules_supported')
DOMAINS=['content_moderation','specialized_advice','intellectual_property']
def avg(v):
    return sum(v)/len(v) if v and all(x is not None for x in v) else None
def seeds(f,t,c):
    if f=='GPT':return [2,4,5]
    if t in ('iou','partial'):return [0,1,2,3,4,5,7,36,42]
    return {'B-A':[0,1,2,3],'C-B':[1,2,3,4,5,36,42],'D-B':[0,1,2,3,4,5,36,42]}[c]

def values(n,f,a):
 assert n>=0 and int(n)==n
 if f is not None:assert 0<=f<=1 and abs(n*f-round(n*f))<1e-7
 lo=[0. if f is None else f,0. if f is None else f*n,0. if a is None else a]
 hi=[1. if f is None else f,float(n)if f is None else f*n,1. if a is None else a]
 return lo,hi

def level(cells,f,t,a,ss,mode,source):
 selected=[cells[f,t,a,s,mode,source+':'+d]for s,d in itertools.product(ss,DOMAINS)]
 return {m:{bound:avg([float(np.mean(np.asarray(c[bound])[:,i]))for c in selected])for bound in('lower','upper')}for i,m in enumerate(NAMES)}

def difference(b,a):return{m:{'lower':b[m]['lower']-a[m]['upper'],'upper':b[m]['upper']-a[m]['lower']}for m in NAMES}

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
