"""Integer-frequency version of the canonical remaining14 reporting metrics."""
from pathlib import Path
import sys
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'ID400_calibration_v1'))
from batched_fixed_metrics import calculate_many,grouped,METRICS as BASE
NAMES=('rule_precision','rule_recall','exact_set_accuracy','parseability','confidence_mean','confidence_SD',
       'IoU_ECE10','exact_error_AUROC','exact_error_AP','F1_AURC','F1_loss_capture_review20',
       'confidence_F1_Spearman','supported_predicted_fraction','supported_prediction_count')

def array(records):
    out=[]
    for r in records:
        valid=r['valid'];gold=set(map(str,r['gold_rules']));pred=set(map(str,r['predicted_rules']))if valid else set()
        tp=len(pred&gold);empty=not(pred or gold)
        known=r.get('semantic_unavailability')is None and r.get('evidence_fraction')is not None and r.get('evidence_all')is not None
        frac=r['evidence_fraction']if known else None
        count=(frac*r['predicted_count']if valid else 0.)if known else None
        out.append([int(valid),r['confidence']if valid else 0.,r['f1'],r['iou'],r['exact'],frac,count,
                    tp/len(pred)if valid and pred else float(empty)if valid else 0.,
                    tp/len(gold)if valid and gold else float(empty)if valid else 0.])
    return np.asarray(out,dtype=float)

def calculate(data,weights):
    x=np.asarray(data);w=np.asarray(weights);base=calculate_many(x[:,:7],w)
    result={m:base[:,BASE.index(m)]for m in NAMES if m in BASE}
    n=w.sum(axis=1);ok=n>0;safe=np.where(ok,n,1)
    for name,values in [('rule_precision',x[:,7]),('rule_recall',x[:,8]),('exact_set_accuracy',x[:,0]*x[:,4]),('parseability',x[:,0])]:
        result[name]=np.where(ok,(w@values)/safe,np.nan)
    keep=x[:,0]==1;v=x[keep];wv=w[:,keep];nv=wv.sum(axis=1);has=nv>0;sv=np.where(has,nv,1)
    result['confidence_mean']=np.full(len(w),np.nan);result['confidence_SD']=np.full(len(w),np.nan);result['exact_error_AP']=np.full(len(w),np.nan)
    if len(v):
        q=v[:,1];mean=(wv@q)/sv
        var=np.sum(wv*(q[None,:]-mean[:,None])**2,axis=1)/sv
        result['confidence_mean'][has]=mean[has];result['confidence_SD'][has]=np.sqrt(np.maximum(0,var[has]))
        # Descending error score equals ascending confidence; each tie enters as a whole.
        counts,errors,_=grouped(q,wv,1-v[:,4]);cumerrors=np.cumsum(errors,axis=1);cumcount=np.cumsum(counts,axis=1)
        precision=np.divide(cumerrors,cumcount,out=np.zeros_like(cumerrors),where=cumcount>0)
        positives=errors.sum(axis=1);defined=has&(positives>0)
        result['exact_error_AP'][defined]=(np.sum(errors*precision,axis=1)/np.where(positives>0,positives,1))[defined]
    return np.column_stack([result[name]for name in NAMES])
