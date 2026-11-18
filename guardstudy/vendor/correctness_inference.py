"""Pure functions extracted from the frozen correctness analysis.

Only imports and artifact I/O are redirected. The conservative V2 availability
rule is applied by the public wrapper after the preserved V1 calculation.
"""
from pathlib import Path
import numpy as np
from . import secondary_metrics as sm
from ..io import read_json as rd, write_json as save
METRICS=('rule_F1','Jaccard_IoU')+tuple(m for m in sm.NAMES if not m.startswith('supported_'))

def calculate(a,w):
    base=sm.calculate(a,w);n=w.sum(1);safe=np.where(n>0,n,1)
    f=(w@(a[:,0]*a[:,2]))/safe;j=(w@(a[:,0]*a[:,3]))/safe
    f[n==0]=np.nan;j[n==0]=np.nan
    return np.column_stack([f,j]+[base[:,sm.NAMES.index(m)] for m in METRICS[2:]])

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
