# Claim 1: Calibration can improve confidence accuracy

[All results](review.html) · [Current paper](paper/report.pdf) · [Reproduction instructions](../../docs/reproduction.md)

A rewards correctness. B adds calibration. C and D combine calibration with all-or-nothing and per-rule evidence credit, respectively. C and D also halve the correctness coefficient relative to B.

These are selected saved comparisons. Each heading gives the model, dataset, domain group, correctness reward, and thinking mode. Full settings and the fixed metric set are in the CSV files below.

An adjusted p-value below 0.05 marks statistical support under the recorded test. A dash means that a value is unavailable or was not tested. Values use a 0–1 scale. Lower calibration error and lower selection loss are better.

## Qwen3.5-9B; SafePyramid (in-distribution dataset); out-of-distribution domains; Jaccard; thinking on

Comparison: **B versus A**. The difference is **B minus A**.

| Metric | A | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.695 | 0.695 | 0.000 | 1.000 | -0.007 to 0.007 |
| Jaccard similarity | 0.604 | 0.607 | 0.003 | — | — |
| Precision | 0.710 | 0.703 | -0.007 | — | — |
| Recall | 0.701 | 0.708 | 0.006 | — | — |
| Exact rule-set accuracy | 0.212 | 0.216 | 0.003 | — | — |
| Parseable answers | 0.901 | 0.891 | -0.011 | — | — |
| Mean confidence | 0.855 | 0.792 | -0.063 | — | — |
| Confidence standard deviation | 0.079 | 0.083 | 0.004 | — | — |
| Expected calibration error (10 bins) | 0.193 | 0.113 | -0.080 | 0.004 | -0.086 to -0.072 |
| Error detection: AUROC | 0.733 | 0.732 | -0.001 | 1.000 | -0.018 to 0.015 |
| Error detection: average precision | 0.866 | 0.865 | -0.002 | — | — |
| F1 loss under confidence-based selection: AURC | 0.166 | 0.149 | -0.017 | 0.051 | -0.022 to -0.011 |
| F1 loss found by 20% review | 0.269 | 0.294 | 0.026 | 0.197 | 0.016 to 0.037 |
| Confidence–F1 rank correlation | 0.352 | 0.394 | 0.041 | 0.744 | 0.016 to 0.065 |
| Mean supported rule fraction per answer | 0.384 to 0.388 (bounds) | 0.438 to 0.443 (bounds) | 0.050 to 0.059 (bounds) | — | — |

## Qwen3.5-9B; DynaBench (out-of-distribution dataset); out-of-distribution domains; Jaccard; thinking on

Comparison: **B versus A**. The difference is **B minus A**.

| Metric | A | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.752 | 0.756 | 0.003 | 0.841 | -0.003 to 0.009 |
| Jaccard similarity | 0.745 | 0.749 | 0.003 | — | — |
| Precision | 0.763 | 0.766 | 0.003 | — | — |
| Recall | 0.749 | 0.753 | 0.004 | — | — |
| Exact rule-set accuracy | 0.722 | 0.725 | 0.004 | — | — |
| Parseable answers | 0.973 | 0.972 | -0.001 | — | — |
| Mean confidence | 0.896 | 0.834 | -0.062 | — | — |
| Confidence standard deviation | 0.087 | 0.117 | 0.030 | — | — |
| Expected calibration error (10 bins) | 0.133 | 0.076 | -0.056 | 0.004 | -0.061 to -0.045 |
| Error detection: AUROC | 0.660 | 0.678 | 0.018 | 0.730 | 0.006 to 0.031 |
| Error detection: average precision | 0.382 | 0.392 | 0.009 | — | — |
| F1 loss under confidence-based selection: AURC | 0.157 | 0.143 | -0.013 | 0.432 | -0.021 to -0.006 |
| F1 loss found by 20% review | 0.351 | 0.357 | 0.006 | 1.000 | -0.011 to 0.024 |
| Confidence–F1 rank correlation | 0.254 | 0.281 | 0.027 | 0.755 | 0.007 to 0.047 |
| Mean supported rule fraction per answer | 0.273 to 0.274 (bounds) | 0.279 to 0.280 (bounds) | 0.004 to 0.007 (bounds) | — | — |

## GPT-OSS-120B; SafePyramid (in-distribution dataset); out-of-distribution domains; Jaccard; thinking on

Comparison: **B versus A**. The difference is **B minus A**.

| Metric | A | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.641 | 0.652 | 0.011 | 0.104 | 0.002 to 0.019 |
| Jaccard similarity | 0.533 | 0.540 | 0.008 | — | — |
| Precision | 0.700 | 0.698 | -0.002 | — | — |
| Recall | 0.629 | 0.649 | 0.020 | — | — |
| Exact rule-set accuracy | 0.160 | 0.149 | -0.011 | — | — |
| Parseable answers | 0.987 | 0.987 | -0.000 | — | — |
| Mean confidence | 0.895 | 0.832 | -0.063 | — | — |
| Confidence standard deviation | 0.057 | 0.067 | 0.010 | — | — |
| Expected calibration error (10 bins) | 0.356 | 0.285 | -0.071 | 0.004 | -0.079 to -0.062 |
| Error detection: AUROC | 0.688 | 0.734 | 0.046 | 0.361 | 0.013 to 0.078 |
| Error detection: average precision | 0.905 | 0.925 | 0.020 | — | — |
| F1 loss under confidence-based selection: AURC | 0.293 | 0.265 | -0.029 | 0.021 | -0.042 to -0.015 |
| F1 loss found by 20% review | 0.248 | 0.262 | 0.014 | 0.587 | -0.002 to 0.030 |
| Confidence–F1 rank correlation | 0.256 | 0.336 | 0.079 | 0.148 | 0.031 to 0.127 |
| Mean supported rule fraction per answer | 0.433 | 0.429 | -0.004 to -0.004 (bounds) | — | — |

## GPT-OSS-120B; DynaBench (out-of-distribution dataset); out-of-distribution domains; Jaccard; thinking on

Comparison: **B versus A**. The difference is **B minus A**.

| Metric | A | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.737 | 0.745 | 0.008 | 0.426 | -0.001 to 0.018 |
| Jaccard similarity | 0.730 | 0.739 | 0.008 | — | — |
| Precision | 0.749 | 0.756 | 0.008 | — | — |
| Recall | 0.732 | 0.740 | 0.008 | — | — |
| Exact rule-set accuracy | 0.709 | 0.718 | 0.009 | — | — |
| Parseable answers | 0.959 | 0.963 | 0.004 | — | — |
| Mean confidence | 0.933 | 0.902 | -0.031 | — | — |
| Confidence standard deviation | 0.051 | 0.066 | 0.015 | — | — |
| Expected calibration error (10 bins) | 0.175 | 0.140 | -0.036 | 0.004 | -0.044 to -0.025 |
| Error detection: AUROC | 0.578 | 0.598 | 0.020 | 0.836 | -0.005 to 0.046 |
| Error detection: average precision | 0.324 | 0.350 | 0.026 | — | — |
| F1 loss under confidence-based selection: AURC | 0.204 | 0.194 | -0.010 | 0.899 | -0.025 to 0.004 |
| F1 loss found by 20% review | 0.260 | 0.310 | 0.050 | 0.053 | 0.017 to 0.082 |
| Confidence–F1 rank correlation | 0.114 | 0.144 | 0.030 | 0.836 | -0.009 to 0.071 |
| Mean supported rule fraction per answer | 0.231 to 0.232 (bounds) | 0.237 | 0.004 to 0.005 (bounds) | — | — |

## Qwen3.5-9B; SafePyramid (in-distribution dataset); in-distribution domains; Jaccard; thinking on

Comparison: **B versus A**. The difference is **B minus A**.

| Metric | A | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.764 | 0.773 | 0.008 | 1.000 | -0.009 to 0.026 |
| Jaccard similarity | 0.694 | 0.701 | 0.007 | — | — |
| Precision | 0.776 | 0.780 | 0.004 | — | — |
| Recall | 0.767 | 0.780 | 0.014 | — | — |
| Exact rule-set accuracy | 0.331 | 0.335 | 0.004 | — | — |
| Parseable answers | 0.906 | 0.922 | 0.016 | — | — |
| Mean confidence | 0.875 | 0.815 | -0.060 | — | — |
| Confidence standard deviation | 0.067 | 0.083 | 0.016 | — | — |
| Expected calibration error (10 bins) | 0.117 | 0.077 | -0.041 | 0.020 | -0.047 to -0.019 |
| Error detection: AUROC | 0.754 | 0.767 | 0.013 | 1.000 | -0.017 to 0.041 |
| Error detection: average precision | 0.801 | 0.813 | 0.012 | — | — |
| F1 loss under confidence-based selection: AURC | 0.097 | 0.093 | -0.004 | 1.000 | -0.012 to 0.004 |
| F1 loss found by 20% review | 0.309 | 0.335 | 0.027 | 0.882 | 0.002 to 0.051 |
| Confidence–F1 rank correlation | 0.441 | 0.485 | 0.044 | 1.000 | -0.004 to 0.090 |
| Mean supported rule fraction per answer | — | — | — | — | — |

## Qwen3.5-9B; DynaBench (out-of-distribution dataset); in-distribution domains; Jaccard; thinking on

Comparison: **B versus A**. The difference is **B minus A**.

| Metric | A | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.805 | 0.805 | -0.000 | 1.000 | -0.014 to 0.014 |
| Jaccard similarity | 0.799 | 0.799 | -0.000 | — | — |
| Precision | 0.815 | 0.815 | 0.000 | — | — |
| Recall | 0.801 | 0.800 | -0.000 | — | — |
| Exact rule-set accuracy | 0.782 | 0.781 | -0.001 | — | — |
| Parseable answers | 0.967 | 0.963 | -0.005 | — | — |
| Mean confidence | 0.904 | 0.851 | -0.054 | — | — |
| Confidence standard deviation | 0.086 | 0.113 | 0.027 | — | — |
| Expected calibration error (10 bins) | 0.109 | 0.096 | -0.014 | 0.542 | -0.022 to 0.013 |
| Error detection: AUROC | 0.730 | 0.736 | 0.006 | — | — |
| Error detection: average precision | 0.442 | 0.420 | -0.021 | — | — |
| F1 loss under confidence-based selection: AURC | 0.103 | 0.098 | -0.006 | 1.000 | -0.016 to 0.005 |
| F1 loss found by 20% review | 0.482 | 0.460 | -0.022 | — | — |
| Confidence–F1 rank correlation | 0.295 | 0.295 | -0.000 | — | — |
| Mean supported rule fraction per answer | 0.305 | 0.317 | 0.012 | 0.542 | -0.002 to 0.026 |

## GPT-OSS-120B; SafePyramid (in-distribution dataset); in-distribution domains; Jaccard; thinking on

Comparison: **B versus A**. The difference is **B minus A**.

| Metric | A | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.722 | 0.750 | 0.028 | 0.123 | 0.009 to 0.048 |
| Jaccard similarity | 0.629 | 0.656 | 0.027 | — | — |
| Precision | 0.767 | 0.780 | 0.013 | — | — |
| Recall | 0.708 | 0.751 | 0.043 | — | — |
| Exact rule-set accuracy | 0.257 | 0.272 | 0.015 | — | — |
| Parseable answers | 0.974 | 0.991 | 0.017 | — | — |
| Mean confidence | 0.906 | 0.846 | -0.060 | — | — |
| Confidence standard deviation | 0.056 | 0.066 | 0.010 | — | — |
| Expected calibration error (10 bins) | 0.261 | 0.187 | -0.074 | 0.020 | -0.092 to -0.054 |
| Error detection: AUROC | 0.700 | 0.751 | 0.052 | 0.720 | -0.017 to 0.112 |
| Error detection: average precision | 0.857 | 0.877 | 0.020 | — | — |
| F1 loss under confidence-based selection: AURC | 0.195 | 0.163 | -0.032 | 0.314 | -0.054 to -0.012 |
| F1 loss found by 20% review | 0.279 | 0.302 | 0.023 | 1.000 | -0.018 to 0.062 |
| Confidence–F1 rank correlation | 0.340 | 0.438 | 0.098 | 0.583 | 0.003 to 0.198 |
| Mean supported rule fraction per answer | — | 0.436 | — | — | — |

## GPT-OSS-120B; DynaBench (out-of-distribution dataset); in-distribution domains; Jaccard; thinking on

Comparison: **B versus A**. The difference is **B minus A**.

| Metric | A | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.789 | 0.785 | -0.004 | 1.000 | -0.033 to 0.022 |
| Jaccard similarity | 0.783 | 0.778 | -0.005 | — | — |
| Precision | 0.799 | 0.797 | -0.002 | — | — |
| Recall | 0.785 | 0.780 | -0.005 | — | — |
| Exact rule-set accuracy | 0.765 | 0.760 | -0.006 | — | — |
| Parseable answers | 0.960 | 0.957 | -0.002 | — | — |
| Mean confidence | 0.936 | 0.908 | -0.027 | — | — |
| Confidence standard deviation | 0.045 | 0.062 | 0.017 | — | — |
| Expected calibration error (10 bins) | 0.136 | 0.121 | -0.015 | 0.899 | -0.032 to 0.016 |
| Error detection: AUROC | 0.675 | 0.644 | -0.031 | — | — |
| Error detection: average precision | 0.398 | 0.378 | -0.019 | — | — |
| F1 loss under confidence-based selection: AURC | 0.129 | 0.151 | 0.022 | 0.778 | -0.011 to 0.053 |
| F1 loss found by 20% review | 0.401 | 0.392 | -0.009 | — | — |
| Confidence–F1 rank correlation | 0.189 | 0.161 | -0.027 | — | — |
| Mean supported rule fraction per answer | 0.240 | 0.246 | 0.006 | 1.000 | -0.023 to 0.033 |

## Qwen3.5-9B; SafePyramid (in-distribution dataset); out-of-distribution domains; Jaccard; thinking on

Comparison: **B versus SFT**. The difference is **B minus SFT**.

| Metric | SFT | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.588 | 0.695 | 0.106 | 0.003 | 0.086 to 0.127 |
| Expected calibration error (10 bins) | 0.218 | 0.113 | -0.105 | 0.003 | -0.120 to -0.089 |
| F1 loss under confidence-based selection: AURC | 0.184 | 0.149 | -0.035 | 0.003 | -0.052 to -0.018 |

## Qwen3.5-9B; DynaBench (out-of-distribution dataset); out-of-distribution domains; Jaccard; thinking on

Comparison: **B versus SFT**. The difference is **B minus SFT**.

| Metric | SFT | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.735 | 0.756 | 0.021 | 0.032 | 0.006 to 0.036 |
| Expected calibration error (10 bins) | 0.132 | 0.076 | -0.056 | 0.003 | -0.068 to -0.039 |
| F1 loss under confidence-based selection: AURC | 0.168 | 0.143 | -0.024 | 0.014 | -0.040 to -0.008 |

## GPT-OSS-120B; SafePyramid (in-distribution dataset); out-of-distribution domains; Jaccard; thinking on

Comparison: **B versus SFT**. The difference is **B minus SFT**.

| Metric | SFT | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.619 | 0.652 | 0.033 | 0.003 | 0.020 to 0.045 |
| Expected calibration error (10 bins) | 0.366 | 0.285 | -0.081 | 0.003 | -0.092 to -0.069 |
| F1 loss under confidence-based selection: AURC | 0.320 | 0.265 | -0.055 | 0.003 | -0.078 to -0.032 |

## GPT-OSS-120B; DynaBench (out-of-distribution dataset); out-of-distribution domains; Jaccard; thinking on

Comparison: **B versus SFT**. The difference is **B minus SFT**.

| Metric | SFT | B | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.729 | 0.745 | 0.016 | 0.148 | 0.000 to 0.030 |
| Expected calibration error (10 bins) | 0.185 | 0.140 | -0.045 | 0.003 | -0.058 to -0.029 |
| F1 loss under confidence-based selection: AURC | 0.235 | 0.194 | -0.042 | 0.024 | -0.066 to -0.015 |

## Complete exports

- [calibration.csv](tables/calibration.csv)
- [sft-controls-ood.csv](tables/sft-controls-ood.csv)
- [sft-controls-id.csv](tables/sft-controls-id.csv)
- [policy-levels.csv](tables/policy-levels.csv)
