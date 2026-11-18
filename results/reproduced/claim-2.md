# Claim 2: Evidence rewards can improve support for predicted rules

[All results](review.html) · [Current paper](paper/report.pdf) · [Reproduction instructions](../../docs/reproduction.md)

A rewards correctness. B adds calibration. C and D combine calibration with all-or-nothing and per-rule evidence credit, respectively. C and D also halve the correctness coefficient relative to B.

These are selected saved comparisons. Each heading gives the model, dataset, domain group, correctness reward, and thinking mode. Full settings and the fixed metric set are in the CSV files below.

An adjusted p-value below 0.05 marks statistical support under the recorded test. A dash means that a value is unavailable or was not tested. Values use a 0–1 scale. Lower calibration error and lower selection loss are better.

Evidence bounds give the possible effect of missing semantic grades. The sampling interval is shown separately. A p-value marked ≤ is a conservative upper bound.

## Qwen3.5-9B; SafePyramid (in-distribution dataset); out-of-distribution domains; binary correctness; thinking on

Comparison: **D versus B**. The difference is **D minus B**.

| Metric | B | D | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.671 | 0.678 | 0.007 | 0.528 | -0.002 to 0.016 |
| Jaccard similarity | 0.583 | 0.591 | 0.007 | — | — |
| Precision | 0.700 | 0.712 | 0.012 | — | — |
| Recall | 0.665 | 0.667 | 0.001 | — | — |
| Exact rule-set accuracy | 0.209 | 0.213 | 0.004 | — | — |
| Parseable answers | 0.890 | 0.883 | -0.007 | — | — |
| Mean confidence | 0.768 | 0.723 | -0.045 | — | — |
| Confidence standard deviation | 0.137 | 0.115 | -0.022 | — | — |
| Expected calibration error (10 bins) | 0.144 | 0.108 | -0.035 | 0.005 | -0.042 to -0.024 |
| Error detection: AUROC | 0.770 | 0.750 | -0.020 | 0.593 | -0.037 to -0.004 |
| Error detection: average precision | 0.890 | 0.872 | -0.018 | — | — |
| F1 loss under confidence-based selection: AURC | 0.164 | 0.155 | -0.009 | 0.166 | -0.016 to -0.002 |
| F1 loss found by 20% review | 0.303 | 0.290 | -0.013 | 0.224 | -0.024 to -0.002 |
| Confidence–F1 rank correlation | 0.436 | 0.406 | -0.030 | 0.593 | -0.058 to -0.003 |
| Mean supported rule fraction per answer | 0.361 to 0.365 (bounds) | 0.511 to 0.515 (bounds) | 0.146 to 0.154 (bounds) | ≤ 0.004 | 0.129 to 0.167 |

## Qwen3.5-9B; SafePyramid (in-distribution dataset); out-of-distribution domains; Jaccard; thinking on

Comparison: **D versus B**. The difference is **D minus B**.

| Metric | B | D | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.695 | 0.696 | 0.001 | 1.000 | -0.006 to 0.008 |
| Jaccard similarity | 0.607 | 0.607 | -0.001 | — | — |
| Precision | 0.703 | 0.712 | 0.009 | — | — |
| Recall | 0.708 | 0.702 | -0.005 | — | — |
| Exact rule-set accuracy | 0.216 | 0.209 | -0.007 | — | — |
| Parseable answers | 0.891 | 0.900 | 0.009 | — | — |
| Mean confidence | 0.792 | 0.785 | -0.007 | — | — |
| Confidence standard deviation | 0.083 | 0.082 | -0.000 | — | — |
| Expected calibration error (10 bins) | 0.113 | 0.116 | 0.003 | 1.000 | -0.003 to 0.008 |
| Error detection: AUROC | 0.732 | 0.737 | 0.004 | 1.000 | -0.013 to 0.022 |
| Error detection: average precision | 0.865 | 0.871 | 0.007 | — | — |
| F1 loss under confidence-based selection: AURC | 0.149 | 0.157 | 0.007 | 0.043 | 0.002 to 0.012 |
| F1 loss found by 20% review | 0.294 | 0.279 | -0.015 | 0.029 | -0.025 to -0.006 |
| Confidence–F1 rank correlation | 0.394 | 0.373 | -0.020 | 0.487 | -0.045 to 0.006 |
| Mean supported rule fraction per answer | 0.438 to 0.443 (bounds) | 0.482 to 0.487 (bounds) | 0.039 to 0.049 (bounds) | ≤ 0.004 | 0.025 to 0.062 |

## GPT-OSS-120B; SafePyramid (in-distribution dataset); out-of-distribution domains; binary correctness; thinking on

Comparison: **D versus B**. The difference is **D minus B**.

| Metric | B | D | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.629 | 0.640 | 0.011 | 0.043 | 0.003 to 0.019 |
| Jaccard similarity | 0.522 | 0.532 | 0.010 | — | — |
| Precision | 0.702 | 0.707 | 0.005 | — | — |
| Recall | 0.609 | 0.622 | 0.013 | — | — |
| Exact rule-set accuracy | 0.159 | 0.160 | 0.001 | — | — |
| Parseable answers | 0.991 | 0.986 | -0.005 | — | — |
| Mean confidence | 0.513 | 0.676 | 0.163 | — | — |
| Confidence standard deviation | 0.128 | 0.087 | -0.042 | — | — |
| Expected calibration error (10 bins) | 0.076 | 0.201 | 0.125 | 0.005 | 0.103 to 0.139 |
| Error detection: AUROC | 0.771 | 0.746 | -0.025 | 1.000 | -0.056 to 0.002 |
| Error detection: average precision | 0.928 | 0.916 | -0.012 | — | — |
| F1 loss under confidence-based selection: AURC | 0.270 | 0.266 | -0.004 | 1.000 | -0.017 to 0.009 |
| F1 loss found by 20% review | 0.250 | 0.252 | 0.001 | 1.000 | -0.013 to 0.017 |
| Confidence–F1 rank correlation | 0.348 | 0.326 | -0.022 | 1.000 | -0.065 to 0.020 |
| Mean supported rule fraction per answer | 0.430 | 0.459 | 0.029 to 0.029 (bounds) | ≤ 0.004 | 0.017 to 0.041 |

## GPT-OSS-120B; SafePyramid (in-distribution dataset); out-of-distribution domains; Jaccard; thinking on

Comparison: **D versus B**. The difference is **D minus B**.

| Metric | B | D | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.652 | 0.644 | -0.008 | 0.447 | -0.017 to 0.001 |
| Jaccard similarity | 0.540 | 0.533 | -0.007 | — | — |
| Precision | 0.698 | 0.704 | 0.005 | — | — |
| Recall | 0.649 | 0.630 | -0.019 | — | — |
| Exact rule-set accuracy | 0.149 | 0.153 | 0.004 | — | — |
| Parseable answers | 0.987 | 0.983 | -0.005 | — | — |
| Mean confidence | 0.832 | 0.844 | 0.012 | — | — |
| Confidence standard deviation | 0.067 | 0.060 | -0.007 | — | — |
| Expected calibration error (10 bins) | 0.285 | 0.302 | 0.016 | 0.009 | 0.008 to 0.025 |
| Error detection: AUROC | 0.734 | 0.731 | -0.003 | 1.000 | -0.040 to 0.032 |
| Error detection: average precision | 0.925 | 0.921 | -0.003 | — | — |
| F1 loss under confidence-based selection: AURC | 0.265 | 0.275 | 0.011 | 0.508 | -0.003 to 0.025 |
| F1 loss found by 20% review | 0.262 | 0.258 | -0.003 | 1.000 | -0.017 to 0.011 |
| Confidence–F1 rank correlation | 0.336 | 0.311 | -0.024 | 0.939 | -0.070 to 0.019 |
| Mean supported rule fraction per answer | 0.429 | 0.452 | 0.023 to 0.023 (bounds) | ≤ 0.004 | 0.011 to 0.035 |

## Complete exports

- [evidence.csv](tables/evidence.csv)
