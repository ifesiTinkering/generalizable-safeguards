"""Diagnostic expected-uniform-tie risk coverage; no scoring or model changes."""
import math
from collections import defaultdict
def aurc(qs,qualities):
 if len(qs)!=len(qualities):raise ValueError('Population mismatch')
 if not qs:return None
 if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in list(qs)+list(qualities)):raise ValueError('Finite unit interval values required')
 groups=defaultdict(list)
 for q,y in zip(qs,qualities):groups[q].append(1-y)
 loss=0.;count=0;risks=[]
 for q in sorted(groups,reverse=True):
  values=groups[q];mean=sum(values)/len(values)
  risks.extend((loss+k*mean)/(count+k) for k in range(1,len(values)+1));loss+=sum(values);count+=len(values)
 return sum(risks)/len(qs)
def precision(r):
 tp,fp,fn=(r[k] for k in ('TP','FP','FN'));return tp/(tp+fp) if tp+fp else float(tp+fp+fn==0)
def summarize(rows):
 valid=[r for r in rows if r['valid'] and type(r.get('confidence')) in (int,float) and math.isfinite(r['confidence']) and 0<=r['confidence']<=1]
 q=[r['confidence'] for r in valid];out={'n_all':len(rows),'n_valid_confidence':len(valid),'n_excluded':len(rows)-len(valid),'population':'valid finite confidence only; invalids excluded from ranking, counted above'}
 for name,ys in [('binary',[float(r['exact_correct']) for r in valid]),('F1',[r['quality_f1'] for r in valid]),('precision',[precision(r) for r in valid])]:
  value=aurc(q,ys);oracle=aurc(ys,ys);risk=sum(1-y for y in ys)/len(ys) if ys else None
  out[name]={'aurc':value,'mean_risk_valid':risk,'oracle_aurc_same_answers':oracle,'excess_aurc_over_oracle':None if value is None else value-oracle,'aurc_minus_random_order':None if value is None else value-risk,'mean_quality_all_invalid_zero':sum(ys)/len(rows) if rows else None}
 return out
