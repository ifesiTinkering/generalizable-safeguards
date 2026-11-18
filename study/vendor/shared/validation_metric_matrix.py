"""Versioned post-hoc diagnostic extension; no model, scoring or selection changes."""
import math
from collections import defaultdict
from validation_aurc import aurc
from validation_metrics_extension import ece10,binary_ranking,mean
QUALITY=('C','F1','precision','recall','S','IoU')
THRESHOLDS=(.5,.8,1.)
def qualities(r):
 t,p,n=(r[k]for k in ('TP','FP','FN'));empty=t+p+n==0
 return {'C':float(r['exact_correct']),'F1':r['quality_f1'],'precision':t/(t+p) if t+p else float(empty),'recall':t/(t+n) if t+n else float(empty),'S':r['partial_score'],'IoU':t/(t+p+n) if t+p+n else 1.}
def capture20(qs,ys):
 if len(qs)!=len(ys):raise ValueError('Population mismatch')
 if any(type(v)not in(int,float)or not math.isfinite(v)or not 0<=v<=1 for v in qs+ys):raise ValueError('Unit-interval values required')
 losses=[1-y for y in ys];total=sum(losses)
 if not qs or total==0:return None
 groups=defaultdict(list)
 for q,l in zip(qs,losses):groups[q].append(l)
 left=.2*len(qs);captured=0.
 for q in sorted(groups):
  group=groups[q];take=min(left,len(group));captured+=take*sum(group)/len(group);left-=take
  if left<=0:break
 return captured/total

def summarize(rows):
 valid=[r for r in rows if r['valid']];qs=[r['confidence']for r in valid];vv=[qualities(r)for r in valid]
 if any(type(q)not in(int,float)or not math.isfinite(q)or not 0<=q<=1 for q in qs):raise ValueError('Malformed valid confidence')
 out={'n_all':len(rows),'n_valid':len(valid),'n_invalid':len(rows)-len(valid),'valid_rate':len(valid)/len(rows)if rows else None}
 for name in QUALITY:
  ys=[v[name]for v in vv]
  if any(type(y)not in(int,float)or not math.isfinite(y)or not 0<=y<=1 for y in ys):raise ValueError('Malformed quality')
  prefix=name+'.';out.update({prefix+'mean_all_invalid_zero':sum(ys)/len(rows)if rows else None,prefix+'mean_valid':mean(ys),prefix+'AURC_valid':aurc(qs,ys),prefix+'loss_capture20_valid':capture20(qs,ys),prefix+'MSE_valid':mean([(q-y)**2 for q,y in zip(qs,ys)]),prefix+'ECE10_valid':ece10(qs,ys)['value']})
  cuts=[1.]if name=='C'else THRESHOLDS
  for t in cuts:
   rank=binary_ranking(qs,[y<t for y in ys]);key=prefix+('exact_error'if name=='C'else 'error_lt_'+str(t))
   out.update({key+'.AUROC_valid':rank['auroc'],key+'.AP_valid':rank['average_precision'],key+'.AUPRC_trapezoid_valid':rank['pr_auc_trapezoid']})
 return out
