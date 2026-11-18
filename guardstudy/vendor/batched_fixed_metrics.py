"""Vectorized integer-frequency evaluation; exact existing tie/bin/invalid rules."""
import numpy as np

METRICS=('rule_F1','IoU_ECE10','exact_error_AUROC','F1_AURC',
         'F1_loss_capture_review20','confidence_F1_Spearman',
         'supported_predicted_fraction','supported_prediction_count')

def grouped(values,weights,measure=None):
    order=np.argsort(values,kind='stable');v=values[order]
    starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
    counts=np.add.reduceat(weights[:,order],starts,axis=1)
    if measure is not None:
        totals=np.add.reduceat(weights[:,order]*measure[order][None,:],starts,axis=1)
    else:totals=None
    inv=np.empty(len(v),dtype=int)
    inv[order]=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(v)]))
    return counts,totals,inv

def calculate_many(data,weights):
    a=np.asarray(data,dtype=float);w=np.asarray(weights)
    if a.ndim!=2 or a.shape[1]!=7 or w.ndim!=2 or w.shape[1]!=len(a):raise ValueError('Shape mismatch')
    if np.any(w<0)or np.any(w!=np.floor(w)):raise ValueError('Integer weights required')
    out=np.full((len(w),8),np.nan);total=w.sum(axis=1);present=total>0
    out[present,0]=np.sum(w*(a[:,0]*a[:,2])[None,:],axis=1)[present]/total[present]
    for col,dst in [(5,6),(6,7)]:
        known=np.isfinite(a[:,col]);ok=present&~np.any(w[:,~known]>0,axis=1)
        numerator=np.sum(w*np.where(known,a[:,col],0.)[None,:],axis=1)
        out[ok,dst]=numerator[ok]/total[ok]
    keep=a[:,0]==1;a=a[keep];w=w[:,keep]
    if not len(a):return out
    if not np.isfinite(a[:,:5]).all()or np.any((a[:,:5]<0)|(a[:,:5]>1)):raise ValueError('Invalid unit core value')
    q,f,j,error=a[:,1],a[:,2],a[:,3],1-a[:,4]
    n=w.sum(axis=1);ok=n>0;safe=np.where(ok,n,1.)
    qc,losses,qix=grouped(q,w,1-f)
    fc,_,fix=grouped(f,w)
    qr=(np.cumsum(qc,axis=1)-(qc-1)/2)[:,qix]
    fr=(np.cumsum(fc,axis=1)-(fc-1)/2)[:,fix]
    qr-=np.sum(w*qr,axis=1)[:,None]/safe[:,None]
    fr-=np.sum(w*fr,axis=1)[:,None]/safe[:,None]
    vx=np.sum(w*qr*qr,axis=1);vy=np.sum(w*fr*fr,axis=1)
    defined=ok&(vx>0)&(vy>0)
    out[defined,5]=np.sum(w*qr*fr,axis=1)[defined]/np.sqrt(vx[defined]*vy[defined])
    rc,errors,_=grouped(1-q,w,error)
    p=errors.sum(axis=1);neg=n-p;defined=ok&(p>0)&(neg>0)
    ranks=np.cumsum(rc,axis=1)-(rc-1)/2
    numerator=np.sum(errors*ranks,axis=1)-p*(p+1)/2
    out[defined,2]=numerator[defined]/(p[defined]*neg[defined])
    loss=losses.sum(axis=1);defined=ok&(loss>0)
    take=np.clip(.2*n[:,None]-(np.cumsum(qc,axis=1)-qc),0,qc)
    ml=np.divide(losses,qc,out=np.zeros_like(losses,dtype=float),where=qc>0)
    out[defined,4]=np.sum(take*ml,axis=1)[defined]/loss[defined]
    counts=qc[:,::-1].astype(np.int64);ll=losses[:,::-1]
    preceding=np.cumsum(counts,axis=1)-counts;priorloss=np.cumsum(ll,axis=1)-ll
    harmonic=np.r_[0.,np.cumsum(1/np.arange(1,int(n.max())+1,dtype=float))]
    meanloss=np.divide(ll,counts,out=np.zeros_like(ll,dtype=float),where=counts>0)
    terms=meanloss*counts+(priorloss-meanloss*preceding)*(harmonic[preceding+counts]-harmonic[preceding])
    out[ok,3]=np.sum(terms,axis=1)[ok]/n[ok]
    bins=np.searchsorted(np.array([i*.1 for i in range(1,10)]),q,side='right')
    _,deviation,_=grouped(bins,w,q-j)
    out[ok,1]=np.abs(deviation).sum(axis=1)[ok]/n[ok]
    return out

def evaluate_many(arrays,cluster_weights,swaps=None,baseline=False):
    low=[];high=[];seen=set()
    for (seed,domain),(a,b,ix) in arrays.items():
        w=cluster_weights[:,ix]
        if swaps is None:
            aa=a;bb=b;aw=bw=w
        else:
            s=swaps[:,ix]
            aa=bb=np.concatenate([a,b],axis=0)
            aw=np.concatenate([w*(~s),w*s],axis=1)
            bw=np.concatenate([w*s,w*(~s)],axis=1)
        if not baseline or domain not in seen:
            low.append(calculate_many(aa,aw));seen.add(domain)
        high.append(calculate_many(bb,bw))
    # Plain mean deliberately propagates undefined domain/seed entries.
    return np.mean(np.stack(high),axis=0)-np.mean(np.stack(low),axis=0)
