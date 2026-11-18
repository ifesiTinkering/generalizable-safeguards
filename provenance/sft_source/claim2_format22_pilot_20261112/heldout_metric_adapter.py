"""Pure saved-output v8 validation metrics; no judge/client/calls or state writes.

S remains historical floored signed correctness, F1 is per-response set F1.
All arms have identical validation semantic requirements. Incomplete packets
must never reach patience or checkpoint-pin updates.
"""
import math
from reward_pipeline import parse,turns,grounded,reward_values
from validation_metric_matrix import summarize,qualities
from validation_metrics_extension import enrich,mean,binary_ranking
from validation_protocol import STOP_DIRECTIONS

STOP_NAMES=tuple(STOP_DIRECTIONS)
def close(a,b):return type(a)in(int,float) and math.isfinite(a) and abs(a-b)<=1e-12

def direction(name):
 if name in STOP_DIRECTIONS:return STOP_DIRECTIONS[name]
 if name in ('malformed_fraction',):return 'min'
 if any(s in name for s in ('.MSE_valid','.ECE10_valid','.AURC_valid')):return 'min'
 if name in ('valid_rate','grounded_evidence_coverage_valid','all_quotes_grounded_fraction_valid','F1_calibration_raw','exact_calibration_raw','mean_validation_reward') or name.startswith('rule_') or name.startswith('F1_dcg_') or name.startswith('F1_ndcg_'):return 'max'
 if any(s in name for s in ('.mean_all_invalid_zero','.mean_valid','.loss_capture20_valid','.AUROC_valid','.AP_valid','.AUPRC_trapezoid_valid')):return 'max'
 raise ValueError('No declared direction for '+name)

def _normalize(row,r):
 gold=set(row['ground_truth_rule_ids']);base={'row_id':r['row_id'],'valid':False,'confidence':None,'partial_score':0.,'quality_f1':0.,'exact_correct':0.,'_gold_count':len(gold),'TP':0,'FP':0,'FN':len(gold),'semantic_all':0.,'semantic_fraction':0.,'coverage':None,'all_grounded':None}
 missing=[]
 if r.get('judge_unavailable') or r.get('provider_unknown') or r.get('incomplete'):missing.append('unresolved_provider_or_judge:'+r['row_id'])
 if not isinstance(r.get('text'),str) or type(r.get('clean_termination')) is not bool or type(r.get('valid')) is not bool:
  missing.append('missing_completed_output_fields:'+r['row_id']);return base,missing
 # A completed but malformed response is a real zero, not an unresolved call.
 try:
  if not r['clean_termination']:raise ValueError('unclean')
  p=parse(r['text'],[str(x['rule_id']) for x in row['all_policy_rules']])
 except (ValueError,TypeError,KeyError):
  if r.get('valid'):raise ValueError('Saved valid flag disagrees with four-tag parse')
  return base,missing
 if r.get('valid') is not True:raise ValueError('Saved invalid flag disagrees with four-tag parse')
 pred=set(p['violated_rule_ids']);tp=len(pred&gold);fp=len(pred-gold);fn=len(gold-pred);c=float(pred==gold);f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.;s=max(0.,(tp-fp)/len(gold)) if gold else c
 for key,value in [('f1',f1),('legacy_partial',s),('exact',c),('confidence',p['confidence'])]:
  if not close(r.get(key),value):raise ValueError('Saved '+key+' disagrees with parsed output')
 source=turns(row['conversation']);local={x['rule_id']:grounded(source,x) for x in p['rule_assessments']};judge=r.get('judge');supported={i:False for i in pred}
 if any(local.values()) and judge is None:missing.append('missing_semantic_judgment:'+r['row_id'])
 if judge is not None:
  vs=judge.get('rules')
  if not isinstance(vs,list) or len(vs)!=len(pred) or {v.get('rule_id') for v in vs}!=pred:raise ValueError('Semantic coverage mismatch')
  for v in vs:
   if type(v.get('claim_supported'))is not bool or type(v.get('evidence_supported'))is not bool:raise ValueError('Semantic verdict requires boolean fields')
   supported[v['rule_id']]=v['claim_supported'] and v['evidence_supported'] and local[v['rule_id']]
 aggregate=float(bool(pred) and all(supported.values()));fraction=sum(supported.values())/len(pred) if pred else 0.
 if not missing:
  for key,value in [('evidence_all',aggregate),('evidence_fraction',fraction)]:
   if not close(r.get(key),value):raise ValueError('Saved '+key+' disagrees with grounded semantic verdict')
 return dict(base,valid=True,confidence=p['confidence'],partial_score=s,quality_f1=f1,exact_correct=c,TP=tp,FP=fp,FN=fn,semantic_all=None if missing else aggregate,semantic_fraction=None if missing else fraction,coverage=sum(local.values())/len(pred) if pred else 1.,all_grounded=all(local.values())),missing

def _undefined_reason(name,records,extra):
 valid=[r for r in records if r['valid']]
 if not valid:return 'no_parseable_outputs'
 if name=='no_violation_accuracy':return 'no_gold_empty_rows'
 if name.startswith('rule_'):return 'zero_micro_denominator'
 if 'ndcg' in name:return 'zero_ideal_F1_gain'
 if '.loss_capture20_' in name:return 'zero_total_loss'
 if any(k in name for k in ('AUROC','AP_valid','AUPRC')) or name.startswith('exact_error_'):
  prefix=name.split('.')[0] if '.' in name else 'C'
  if prefix not in ('C','F1','precision','recall','S'):prefix='C'
  threshold=1.
  if '.error_lt_' in name:threshold=float(name.split('.error_lt_')[1].split('.AU')[0].split('.AP')[0])
  errors=sum(qualities(r)[prefix]<threshold for r in valid)
  return 'no_error_positives' if errors==0 else 'no_correct_negatives' if errors==len(valid) else 'ranking_population_undefined'
 return 'mathematically_undefined_population'

def metric_values(records,rows):
 n=len(rows)
 if n<1:raise ValueError("Empty reporting population")
 rows=list(rows.values()) if isinstance(rows,dict) else list(rows);by={r['row_id']:r for r in rows}
 if len(rows)!=n or len(by)!=n or len(records)!=n or len({r['row_id'] for r in records})!=n or {r['row_id'] for r in records}!=set(by):raise ValueError('Exact unique reporting population required')
 rr=[];missing=[]
 for r in records:
  value,issues=_normalize(by[r['row_id']],r);rr.append(value);missing.extend(issues)
 metrics=summarize(rr);extra={};enrich(extra,rr)
 for key in ('rule_precision_micro','rule_recall_micro','rule_f1_micro'):metrics[key]=extra[key]
 for scope in ('full','top20percent'):
  for key in ('dcg','ndcg'):metrics['F1_'+key+'_'+scope]=extra['f1_confidence_ranked_dcg'][scope][key]
 historical={k:v for k,v in metrics.items() if k not in ('n_all','n_valid','n_invalid')}
 if len(historical)!=77:raise ValueError('Historical77 metric roster changed')
 valid=[r for r in rr if r['valid']];goldempty=[r for r in rr if r['_gold_count']==0];rank=extra['exact_error_ranking']
 stops={'exact_set_accuracy':metrics['C.mean_all_invalid_zero'],'macro_precision':metrics['precision.mean_all_invalid_zero'],'macro_recall':metrics['recall.mean_all_invalid_zero'],'macro_f1':metrics['F1.mean_all_invalid_zero'],
 'false_positives_per_parseable_example':mean([r['FP'] for r in valid]),'true_positives_per_parseable_example':mean([r['TP'] for r in valid]),'no_violation_accuracy':mean([r['exact_correct'] for r in goldempty]),
 'exact_error_auroc':rank['auroc'],'exact_error_average_precision':rank['average_precision'],'exact_error_auprc':rank['pr_auc_trapezoid'],'exact_correctness_ece_10_equal_width':metrics['C.ECE10_valid'],'exact_correctness_brier':metrics['C.MSE_valid'],
 'semantic_aggregate_evidence':None if missing else mean([r['semantic_all'] for r in rr]),'semantic_per_rule_evidence':None if missing else mean([r['semantic_fraction'] for r in rr])}
 if set(stops)!=set(STOP_NAMES):raise ValueError('Stop metric roster mismatch')
 tracked=dict(historical,**stops,malformed_fraction=1-len(valid)/len(rr),grounded_evidence_coverage_valid=mean([r['coverage'] for r in valid]),all_quotes_grounded_fraction_valid=mean([float(r['all_grounded']) for r in valid]),
 F1_calibration_raw=mean([reward_values(r['quality_f1'],r['confidence'])['B'] if r['valid'] else -.6 for r in rr]),exact_calibration_raw=mean([reward_values(r['exact_correct'],r['confidence'])['B'] if r['valid'] else -.6 for r in rr]))
 # Optional actual validation reward is explicitly arm-specific, not a shared stop metric.
 if all(type(r.get('reward')) in (int,float) and math.isfinite(r['reward']) for r in records):tracked['mean_validation_reward']=mean([r['reward'] for r in records])
 undefined={k:_undefined_reason(k,rr,extra) for k,v in tracked.items() if v is None and not(k.startswith('semantic_') and missing)}
 return {'complete':not missing,'stop_metrics':stops,'tracked_metrics':tracked,'directions':{k:direction(k) for k in tracked},'undefined_reasons':undefined,'incomplete_reasons':sorted(set(missing)),
 'historical77_metric_names':sorted(historical),'populations':{'all':n,'parseable':len(valid),'malformed':n-len(valid),'gold_empty':len(goldempty),'exact_errors_parseable':rank['errors'],'exact_correct_parseable':rank['non_errors']},'reporting_semantics':{'partial_score':'historical floored signed S; never F1','quality_f1':'per-response set F1','semantic_validation':'same judge and grounding across allABCD; empty/malformed zero; missing judge incomplete'},'provider_calls':0}
def summarize_validation(rows,records):return metric_values(records,rows)
