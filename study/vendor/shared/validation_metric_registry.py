"""Finite all-metric v11 registry. Aliases remain reported/pinned, stop once."""
from validation_metric_matrix import summarize
CORE_DIRECTIONS={
 'exact_set_accuracy':'max','macro_precision':'max','macro_recall':'max','macro_f1':'max',
 'false_positives_per_parseable_example':'min','true_positives_per_parseable_example':'max',
 'no_violation_accuracy':'max','exact_error_auroc':'max','exact_error_average_precision':'max',
 'exact_error_auprc':'max','exact_correctness_ece_10_equal_width':'min','exact_correctness_brier':'min',
 'semantic_aggregate_evidence':'max','semantic_per_rule_evidence':'max','jaccard_iou':'max'}
ALIASES={
 'C.mean_all_invalid_zero':'exact_set_accuracy','F1.mean_all_invalid_zero':'macro_f1',
 'precision.mean_all_invalid_zero':'macro_precision','recall.mean_all_invalid_zero':'macro_recall',
 'C.exact_error.AUROC_valid':'exact_error_auroc','C.exact_error.AP_valid':'exact_error_average_precision',
 'C.exact_error.AUPRC_trapezoid_valid':'exact_error_auprc','C.ECE10_valid':'exact_correctness_ece_10_equal_width',
 'C.MSE_valid':'exact_correctness_brier','IoU.mean_all_invalid_zero':'jaccard_iou',
 'malformed_fraction':'valid_rate'}
ALIAS_TRANSFORMS={'malformed_fraction':'one_minus'}
def direction(name):
 if name in CORE_DIRECTIONS:return CORE_DIRECTIONS[name]
 if name=='malformed_fraction':return 'min'
 if any(x in name for x in ('.MSE_valid','.ECE10_valid','.AURC_valid')):return 'min'
 if name in ('valid_rate','grounded_evidence_coverage_valid','all_quotes_grounded_fraction_valid','F1_calibration_raw','exact_calibration_raw','IoU_calibration_raw','mean_validation_reward') or name.startswith(('rule_','F1_dcg_','F1_ndcg_')):return 'max'
 if any(x in name for x in ('.mean_all_invalid_zero','.mean_valid','.loss_capture20_valid','.AUROC_valid','.AP_valid','.AUPRC_trapezoid_valid')):return 'max'
 raise ValueError('No declared metric direction: '+name)
MATRIX_NAMES=set(summarize([]))-{'n_all','n_valid','n_invalid'}
APPENDIX_NAMES={'rule_precision_micro','rule_recall_micro','rule_f1_micro'}|{'F1_'+kind+'_'+scope for kind in ('dcg','ndcg') for scope in ('full','top20percent')}
EXTRA_NAMES={'malformed_fraction','grounded_evidence_coverage_valid','all_quotes_grounded_fraction_valid','F1_calibration_raw','exact_calibration_raw','IoU_calibration_raw','mean_validation_reward'}
TRACKED_DIRECTIONS={k:direction(k) for k in sorted(MATRIX_NAMES|APPENDIX_NAMES|set(CORE_DIRECTIONS)|EXTRA_NAMES)}
STOP_DIRECTIONS={k:v for k,v in TRACKED_DIRECTIONS.items() if k not in ALIASES}
HISTORICAL77_NAMES=sorted((MATRIX_NAMES|APPENDIX_NAMES)-{k for k in MATRIX_NAMES if k.startswith('IoU.')})
assert len(HISTORICAL77_NAMES)==77
assert set(ALIASES)<=set(TRACKED_DIRECTIONS) and set(ALIASES.values())<=set(STOP_DIRECTIONS)
def canonical_values(tracked):
 if set(tracked)!=set(TRACKED_DIRECTIONS):raise ValueError('Finite tracked metric registry mismatch')
 for alias,canonical in ALIASES.items():
  a,b=tracked[alias],tracked[canonical]
  if (a is None)!=(b is None):raise ValueError('Alias undefined mismatch: '+alias)
  if a is not None:
   expected=1-b if ALIAS_TRANSFORMS.get(alias)=='one_minus' else b
   if abs(a-expected)>1e-12:raise ValueError('Alias value mismatch: '+alias)
 return {k:tracked[k] for k in STOP_DIRECTIONS}
