"""Reporting-only metrics; never changes model rewards, parsing or inference."""
import bisect
import collections
import math

F1_THRESHOLDS=(.5,.8,1.)

def mean(xs):return sum(xs)/len(xs) if xs else None

def ece10(qs, targets):
    """Match RLCR np.linspace/digitize convention including floating edges."""
    if len(qs)!=len(targets):raise ValueError('Confidence/target population mismatch')
    edges=[i*.1 for i in range(1,10)]
    buckets=[[] for _ in range(10)]
    for q,y in zip(qs,targets):
        if not all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in (q,y)):
            raise ValueError('Finite confidence/target in [0,1] required')
        buckets[bisect.bisect_right(edges,q)].append((q,y))
    out=[];total=0.
    for i,values in enumerate(buckets):
        q=mean([x for x,_ in values]);y=mean([x for _,x in values])
        if values:total+=len(values)*abs(q-y)
        out.append({'bin':i,'lower':0. if i==0 else edges[i-1], 'upper':1. if i==9 else edges[i],
                    'upper_inclusive':i==9,'n':len(values),'mean_confidence':q,'mean_target':y})
    return {'value':total/len(qs) if qs else None,'n':len(qs),'bins':out}


def binary_ranking(qs, errors):
    """Exact tie groups; error-positive ROC and two distinct PR summaries."""
    if len(qs)!=len(errors) or any(type(e) is not bool for e in errors):raise ValueError('Binary labels required')
    if any(type(q) not in (float,int) or not math.isfinite(q) or not 0<=q<=1 for q in qs):raise ValueError('Invalid confidence')
    n=len(qs);p=sum(errors);negative=n-p
    result={'n':n,'errors':p,'non_errors':negative,'error_prevalence':p/n if n else None,
            'auroc':None,'average_precision':None,'pr_auc_trapezoid':None,
            'degenerate':p in (0,n),'undefined_reason':None,'pr_points':[]}
    if not n:result['undefined_reason']='empty_population';return result
    if not p:result['undefined_reason']='no_errors';return result
    groups=collections.defaultdict(list)
    for q,e in zip(qs,errors):groups[1-q].append(e)
    seen=found=0;previous_recall=0.;previous_precision=1.;ap=auc=0.
    points=[{'recall':0.,'precision':1.,'error_score_threshold':None}]
    for score in sorted(groups,reverse=True):
        labels=groups[score];seen+=len(labels);found+=sum(labels)
        recall=found/p;precision=found/seen;dr=recall-previous_recall
        ap+=dr*precision;auc+=dr*(previous_precision+precision)/2
        points.append({'recall':recall,'precision':precision,'error_score_threshold':score})
        previous_recall,previous_precision=recall,precision
    result.update(average_precision=ap,pr_auc_trapezoid=auc,pr_points=points)
    if not negative:result['undefined_reason']='all_errors';return result
    seen=0;rank_sum=0.
    for score in sorted(groups):
        labels=groups[score];rank_sum+=sum(labels)*(seen+(len(labels)+1)/2);seen+=len(labels)
    result['auroc']=(rank_sum-p*(p+1)/2)/(p*negative)
    return result


def confidence_spread(records):
    """Descriptive valid-only confidence distribution; population SD, no inference."""
    qs=[r['confidence'] for r in records if r['valid']]
    n=len(qs);average=mean(qs)
    counts=collections.Counter(qs)
    edges=[i*.1 for i in range(1,10)];hist=[0]*10
    for q in qs:
        if type(q) not in (float,int) or not math.isfinite(q) or not 0<=q<=1:
            raise ValueError('Invalid valid-output confidence')
        hist[bisect.bisect_right(edges,q)]+=1
    return {'n_all':len(records),'n_valid':n,'n_invalid':len(records)-n,
            'mean':average,'sd_population':math.sqrt(sum((q-average)**2 for q in qs)/n) if n else None,
            'fraction_ge_0_9':sum(q>=.9 for q in qs)/n if n else None,
            'fraction_eq_1':sum(q==1. for q in qs)/n if n else None,
            'unique_values':len(counts),'value_counts':[{'confidence':q,'n':counts[q]} for q in sorted(counts)],
            'histogram':[{'lower':0. if i==0 else edges[i-1],'upper':1. if i==9 else edges[i],
                          'upper_inclusive':i==9,'n':count} for i,count in enumerate(hist)]}


def f1_dcg(qs, gains):
    """Linear F1 gain, log2(rank+1) discount, expected uniform confidence ties."""
    if len(qs)!=len(gains):raise ValueError('Confidence/F1 population mismatch')
    if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in qs+gains):
        raise ValueError('Finite confidence/F1 in [0,1] required')
    n=len(gains);groups=collections.defaultdict(list)
    for q,g in zip(qs,gains):groups[q].append(g)
    expected=[]
    for q in sorted(groups,reverse=True):expected.extend([mean(groups[q])]*len(groups[q]))
    ideal=sorted(gains,reverse=True)
    def at(k):
        dcg=sum(g/math.log2(i+2) for i,g in enumerate(expected[:k]))
        idcg=sum(g/math.log2(i+2) for i,g in enumerate(ideal[:k]))
        return {'k':k,'dcg':dcg if n else None,'ideal_dcg':idcg if n else None,
                'ndcg':dcg/idcg if idcg>0 else None,
                'undefined_reason':'empty_population' if not n else 'zero_ideal_gain' if not idcg else None}
    return {'n_valid':n,'gain':'linear_F1','discount':'1/log2(rank+1), ranks start1',
            'ties':'expected uniform permutation within equal confidence groups, including cutoff-straddling ties',
            'full':at(n),'top20percent':at(math.ceil(.2*n))}


def enrich(cell,records):
    valid=[r for r in records if r['valid']]
    qs=[r['confidence'] for r in valid];fs=[r['quality_f1'] for r in valid];ss=[r['partial_score'] for r in valid]
    cell['f1_confidence_ranked_dcg']=f1_dcg(qs,fs)
    cell['confidence_spread_valid']=confidence_spread(records)
    cell['confidence_by_quality_band_valid']={}
    for name,field in [('F1','quality_f1'),('S','partial_score')]:
        bands={}
        for label,predicate in [('weak_le_0_2',lambda y:y<=.2),('middle_gt_0_2_lt_0_8',lambda y:.2<y<.8),('strong_ge_0_8',lambda y:y>=.8)]:
            subset=[r for r in valid if predicate(r[field])]
            bands[label]=confidence_spread(subset)
            bands[label]['mean_quality']=mean([r[field] for r in subset])
            bands[label]['fraction_of_valid']=len(subset)/len(valid) if valid else None
        cell['confidence_by_quality_band_valid'][name]=bands
    cell['exact_ece10_rlcr']=ece10(qs,[float(r['exact_correct']) for r in valid])
    cell['f1_ece10_rlcr']=ece10(qs,fs);cell['S_ece10_rlcr']=ece10(qs,ss)
    cell['exact_error_ranking']=binary_ranking(qs,[not bool(r['exact_correct']) for r in valid])
    cell['f1_error_ranking']={str(t):binary_ranking(qs,[f<t for f in fs]) for t in F1_THRESHOLDS}
    precision=[];recall=[];tp=fp=fn=0
    for r in records:
        if not r['valid']:
            precision.append(0.);recall.append(0.);fn+=r['_gold_count'];continue
        t,p,n=r['TP'],r['FP'],r['FN'];tp+=t;fp+=p;fn+=n
        empty=(t+p+n)==0
        precision.append(t/(t+p) if t+p else float(empty))
        recall.append(t/(t+n) if t+n else float(empty))
    cell.update(rule_precision_macro_all=mean(precision),rule_recall_macro_all=mean(recall),
                rule_precision_micro=tp/(tp+fp) if tp+fp else None,
                rule_recall_micro=tp/(tp+fn) if tp+fn else None,
                rule_f1_micro=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                rule_micro_counts={'TP':tp,'FP':fp,'FN':fn},
                missing_confidence_n=len(records)-len(valid))


def extend_summarize(frozen_summarize):
    def wrapped(records):
        out=frozen_summarize(records,thresholds=F1_THRESHOLDS)
        enrich(out['all'],records)
        for name,flag in [('gold_positive',True),('gold_negative',False)]:
            enrich(out[name],[r for r in records if r['gold_positive']==flag])
        for domain,scopes in out['by_domain'].items():
            subset=[r for r in records if r['domain']==domain]
            enrich(scopes['all'],subset)
            for name,flag in [('gold_positive',True),('gold_negative',False)]:
                enrich(scopes[name],[r for r in subset if r['gold_positive']==flag])
        return out
    return wrapped
