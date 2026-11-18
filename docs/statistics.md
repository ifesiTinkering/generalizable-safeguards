# Metrics and statistical results

## Fixed reporting metrics

| Group | Metrics | Direction |
|---|---|---|
| Correctness | Set F1, precision, recall, Jaccard similarity, exact set accuracy, parseable-answer fraction | Higher is better |
| Confidence scores | Mean confidence, confidence standard deviation | Descriptive; neither direction is inherently better |
| Calibration | Expected calibration error in ten bins, against Jaccard answer quality | Lower is better |
| Confidence usefulness | Error-detection AUROC and average precision; confidence-F1 Spearman correlation; F1 loss found by reviewing the lowest-confidence 20% | Higher is better |
| Confidence-based selection | Area under the risk-coverage curve, using F1 loss | Lower is better |
| Evidence support | Mean per-answer fraction of predicted rules with a nonzero semantic evidence score | Higher is better |

The saved observations retain supported-rule count as an additional diagnostic. Main exports use the fixed fifteen-metric set above. Metrics use fractions, not percentages, unless a paper table explicitly says percentage points.

Malformed answers receive zero correctness. Confidence ranking uses parseable answers with valid confidence. An undefined correlation or classification metric remains missing. Confidence ties are handled explicitly, including ties at the human-review boundary. Empty correct rule sets count as correct predictions; they earn zero evidence support.

The calibration reward optimizes squared error against its chosen correctness target. The main calibration tables instead report expected calibration error against Jaccard quality, alongside ranking and decision-use metrics. A lower confidence error does not establish a universal improvement in answer selection or human-review usefulness.

## Aggregation

The broad domain panels calculate each metric within a domain and checkpoint. They average domains equally, then average eligible paired training seeds within the same model. The two model families are reported separately.

Every comparison records its actual paired seeds. Qwen's complete seed universe does not imply that all nine seeds are eligible for every arm comparison. GPT's seeds are the subset 2, 4, and 5. A shared SFT baseline remains one checkpoint, not one independent baseline per RL seed.

## P-values and intervals

- The main comparisons pair answers by conversation. All answers and policy views from one conversation move together during resampling or arm-label exchange. This avoids treating repeated answers as independent questions.
- The saved analyses report a raw two-sided permutation p-value and a Benjamini-Hochberg adjusted p-value. The latter controls the false discovery rate over the declared family. The JSON field `BH_q` is this adjusted value. Holm-adjusted values are also retained.
- The paper uses an adjusted threshold of 0.05 where stated. A point estimate, a percentile interval, and a multiple-testing-adjusted p-value answer different questions. We do not switch methods to obtain a desired significance label.
- Ordinary intervals are percentile bootstrap intervals. Undefined bootstrap or null statistics are handled conservatively. The fixed plans specify the availability thresholds.
- Inference is conditional on the evaluated checkpoints. It describes variation over held-out conversations. Three GPT seeds provide limited evidence about variation over new training seeds.

The exact resampling counts, random seeds, hypothesis families, and amendments belong in `provenance/plans/`. They are preserved there for reproducibility. The readable result tables focus on the estimate, interval, and adjusted p-value.

## Missing evidence grades

A missing semantic grade is not a failed evidence judgment. The saved analysis assigns the unknown value its possible lower and upper values. It reports the resulting bounds on an evidence difference. Its conservative test asks whether the result survives those missing values.

Evidence bounds express incomplete identification. A 95% sampling interval expresses sampling uncertainty. Both are labeled separately in CSV and JSON exports. `evidence_adjusted_p_upper` is an upper bound on an adjusted p-value; it is not an ordinary exact p-value.

The ordinary evidence comparison and the support-only sensitivity analysis use separate frozen multiple-testing families. The latter includes Qwen's set-F1 rewards and retains unavailable slots at a p-value of one. Their adjustments are computed separately.

## Recalculation coverage

- `verify --full` recomputes metric values and comparison means from the committed observations.
- `infer` recomputes the original six-endpoint broad-panel calibration family, including its unavailable slots and original random-number sequence.
- `recompute` recalculates the out-of-distribution evidence family and the conservative matched-correctness family. `reproduce` preserves the saved statistical results for in-domain calibration/SFT controls and policy-level tests. Their original scripts remain in the provenance directory. It does not silently substitute a different test for those results.
- The policy-level adjacent-transition family was a secondary follow-up after full-span results had been inspected. Its frozen plan records that chronology.
