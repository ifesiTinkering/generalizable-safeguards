"""Exact integer-frequency versions of the six prospectively fixed metrics.

Weights represent repeated whole conversations, never fractional pseudo-draws.
The numerical contract is the existing reporting implementation, including its
floating ECE edges and 1-q ROC tie groups. No production source is imported.
"""
import numpy as np

NAMES = ('confidence_F1_Spearman', 'exact_error_AUROC',
         'F1_loss_capture_review20', 'F1_AURC', 'IoU_ECE10', 'rule_F1')

def groups(x, w):
    u, ix = np.unique(x, return_inverse=True)
    return u, ix, np.bincount(ix, weights=w, minlength=len(u))

def ranks(x, w):
    _, ix, counts = groups(x, w)
    return (np.cumsum(counts) - (counts-1)/2)[ix]

def correlation(x, y, w):
    a, b = ranks(x,w), ranks(y,w)
    n = w.sum(); a=a-np.sum(w*a)/n; b=b-np.sum(w*b)/n
    aa=np.sum(w*a*a); bb=np.sum(w*b*b)
    return float(np.sum(w*a*b)/np.sqrt(aa*bb)) if aa>0 and bb>0 else None

def calculate(data, weights=None):
    # columns: valid, q, F1, IoU, exact. Invalid q is stored as zero only in
    # this numeric container, and is removed before any confidence calculation.
    a=np.asarray(data, dtype=float)
    w=np.ones(len(a),dtype=np.int64) if weights is None else np.asarray(weights)
    if a.ndim!=2 or a.shape[1]!=5 or len(w)!=len(a): raise ValueError('Shape mismatch')
    if np.any(w<0) or np.any(w!=np.floor(w)): raise ValueError('Integer counts required')
    out={k:None for k in NAMES}; total=w.sum()
    if not total: return out
    out['rule_F1']=float(np.sum(w*a[:,0]*a[:,2])/total)
    selected=(w>0)&(a[:,0]==1); a=a[selected]; w=w[selected]
    if not len(a): return out
    q,f,j,e=a[:,1],a[:,2],a[:,3],1-a[:,4]; n=w.sum()
    if not np.isfinite(a).all() or np.any((a<0)|(a>1)): raise ValueError('Invalid finite unit values')
    out['confidence_F1_Spearman']=correlation(q,f,w)
    # Preserve canonical binary_ranking's exact 1-q grouping.
    risk,ix,counts=groups(1-q,w); errors=np.bincount(ix,weights=w*e,minlength=len(risk))
    p=errors.sum(); neg=n-p
    if p>0 and neg>0:
        rank=np.cumsum(counts)-(counts-1)/2
        out['exact_error_AUROC']=float((np.sum(errors*rank)-p*(p+1)/2)/(p*neg))
    qs,ix,counts=groups(q,w); losses=np.bincount(ix,weights=w*(1-f),minlength=len(qs))
    loss=losses.sum()
    if loss>0:
        take=np.clip(.2*n-(np.cumsum(counts)-counts),0,counts)
        out['F1_loss_capture_review20']=float(np.sum(take*losses/counts)/loss)
    # Sum exactly the uniform-tie expected risk at every integer rank using
    # H_(c+m)-H_c. No expansion of bootstrap copies is necessary.
    counts=counts[::-1].astype(np.int64); losses=losses[::-1]
    preceding=np.cumsum(counts)-counts; priorloss=np.cumsum(losses)-losses
    h=np.concatenate(([0.],np.cumsum(1/np.arange(1,int(n)+1,dtype=float))))
    meanloss=losses/counts
    out['F1_AURC']=float(np.sum(meanloss*counts+(priorloss-meanloss*preceding)*(h[preceding+counts]-h[preceding]))/n)
    bins=np.searchsorted(np.array([i*.1 for i in range(1,10)]),q,side='right')
    out['IoU_ECE10']=float(np.abs(np.bincount(bins,weights=w*(q-j),minlength=10)).sum()/n)
    return out

def array(records):
    return np.asarray([[int(r['valid']),r['confidence'] if r['valid'] else 0.,
                        r['f1'],r['iou'],r['exact']] for r in records],dtype=float)
