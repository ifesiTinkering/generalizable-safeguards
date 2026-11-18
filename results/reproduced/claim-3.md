# Claim 3: Binary correctness rewards do not always generalize worse than partial rewards

[All results](review.html) · [Current paper](paper/report.pdf) · [Reproduction instructions](../../docs/reproduction.md)

A rewards correctness. B adds calibration. C and D combine calibration with all-or-nothing and per-rule evidence credit, respectively. C and D also halve the correctness coefficient relative to B.

These are selected saved comparisons. Each heading gives the model, dataset, domain group, correctness reward, and thinking mode. Full settings and the fixed metric set are in the CSV files below.

An adjusted p-value below 0.05 marks statistical support under the recorded test. A dash means that a value is unavailable or was not tested. Values use a 0–1 scale. Lower calibration error and lower selection loss are better.

## Qwen3.5-9B; DynaBench (out-of-distribution dataset); out-of-distribution domains; Jaccard; arm A; thinking off

Comparison: **Binary correctness A versus Jaccard A**. The difference is **Binary correctness A minus Jaccard A**.

| Metric | Jaccard A | Binary correctness A | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.476 | 0.543 | 0.067 | 0.002 | 0.042 to 0.091 |
| Jaccard similarity | 0.469 | 0.537 | 0.068 | 0.002 | 0.044 to 0.092 |
| Precision | 0.476 | 0.549 | 0.073 | 0.002 | 0.048 to 0.096 |
| Recall | 0.484 | 0.544 | 0.060 | 0.002 | 0.034 to 0.084 |
| Exact rule-set accuracy | 0.447 | 0.521 | 0.073 | 0.002 | 0.050 to 0.096 |
| Parseable answers | 0.866 | 0.972 | 0.106 | 0.002 | 0.090 to 0.121 |
| Mean confidence | 0.895 | 0.923 | 0.028 | 0.002 | 0.023 to 0.033 |
| Confidence standard deviation | 0.123 | 0.081 | -0.042 | 0.002 | -0.052 to -0.031 |
| Expected calibration error (10 bins) | 0.367 | 0.374 | 0.007 | 0.754 | -0.016 to 0.030 |
| Error detection: AUROC | 0.596 | 0.576 | -0.020 | 0.123 | -0.041 to 0.002 |
| Error detection: average precision | 0.567 | 0.524 | -0.043 | 0.017 | -0.074 to -0.012 |
| F1 loss under confidence-based selection: AURC | 0.405 | 0.405 | 0.000 | 1.000 | -0.026 to 0.026 |
| F1 loss found by 20% review | 0.244 | 0.242 | -0.002 | 0.922 | -0.019 to 0.013 |
| Confidence–F1 rank correlation | 0.147 | 0.133 | -0.015 | 0.666 | -0.055 to 0.027 |
| Mean supported rule fraction per answer | 0.047 to 0.049 (bounds) | 0.019 to 0.020 (bounds) | -0.029 to -0.028 (bounds) | — | — |

## Qwen3.5-9B; DynaBench (out-of-distribution dataset); out-of-distribution domains; Jaccard; arm C; thinking off

Comparison: **Binary correctness C versus Jaccard C**. The difference is **Binary correctness C minus Jaccard C**.

| Metric | Jaccard C | Binary correctness C | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.494 | 0.523 | 0.029 | 0.002 | 0.017 to 0.041 |
| Jaccard similarity | 0.489 | 0.517 | 0.028 | 0.002 | 0.016 to 0.040 |
| Precision | 0.495 | 0.529 | 0.035 | 0.002 | 0.022 to 0.048 |
| Recall | 0.500 | 0.524 | 0.024 | 0.002 | 0.012 to 0.037 |
| Exact rule-set accuracy | 0.475 | 0.499 | 0.024 | 0.002 | 0.012 to 0.037 |
| Parseable answers | 0.914 | 0.935 | 0.022 | 0.002 | 0.014 to 0.029 |
| Mean confidence | 0.826 | 0.659 | -0.166 | 0.002 | -0.182 to -0.149 |
| Confidence standard deviation | 0.141 | 0.335 | 0.194 | 0.002 | 0.185 to 0.202 |
| Expected calibration error (10 bins) | 0.297 | 0.339 | 0.042 | 0.003 | 0.018 to 0.065 |
| Error detection: AUROC | 0.580 | 0.568 | -0.012 | 0.615 | -0.030 to 0.006 |
| Error detection: average precision | 0.543 | 0.514 | -0.029 | 0.244 | -0.051 to -0.006 |
| F1 loss under confidence-based selection: AURC | 0.417 | 0.407 | -0.010 | 0.440 | -0.023 to 0.003 |
| F1 loss found by 20% review | 0.229 | 0.208 | -0.021 | 0.182 | -0.034 to -0.007 |
| Confidence–F1 rank correlation | 0.117 | 0.100 | -0.017 | 0.719 | -0.050 to 0.018 |
| Mean supported rule fraction per answer | 0.024 to 0.025 (bounds) | 0.043 to 0.044 (bounds) | 0.018 to 0.020 (bounds) | — | — |

## GPT-OSS-120B; DynaBench (out-of-distribution dataset); out-of-distribution domains; Jaccard; arm A; thinking off

Comparison: **Binary correctness A versus Jaccard A**. The difference is **Binary correctness A minus Jaccard A**.

| Metric | Jaccard A | Binary correctness A | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.398 | 0.502 | 0.104 | 0.006 | 0.039 to 0.168 |
| Jaccard similarity | 0.390 | 0.502 | 0.112 | 0.002 | 0.047 to 0.175 |
| Precision | 0.414 | 0.502 | 0.089 | 0.028 | 0.022 to 0.155 |
| Recall | 0.393 | 0.502 | 0.109 | 0.002 | 0.045 to 0.173 |
| Exact rule-set accuracy | 0.369 | 0.501 | 0.132 | 0.002 | 0.068 to 0.196 |
| Parseable answers | 0.785 | 0.997 | 0.212 | 0.002 | 0.186 to 0.238 |
| Mean confidence | 0.882 | 0.930 | 0.048 | 0.002 | 0.039 to 0.056 |
| Confidence standard deviation | 0.113 | 0.105 | -0.008 | 0.531 | -0.025 to 0.006 |
| Expected calibration error (10 bins) | 0.384 | 0.427 | 0.043 | 0.376 | -0.024 to 0.109 |
| Error detection: AUROC | 0.582 | 0.577 | -0.005 | 0.917 | -0.041 to 0.032 |
| Error detection: average precision | 0.627 | 0.600 | -0.027 | 0.679 | -0.100 to 0.044 |
| F1 loss under confidence-based selection: AURC | 0.457 | 0.453 | -0.004 | 1.000 | -0.074 to 0.066 |
| F1 loss found by 20% review | 0.258 | 0.260 | 0.002 | 0.987 | -0.024 to 0.027 |
| Confidence–F1 rank correlation | 0.125 | 0.134 | 0.009 | 0.898 | -0.054 to 0.073 |
| Mean supported rule fraction per answer | 0.127 | 0.003 | -0.124 to -0.124 (bounds) | — | — |

## GPT-OSS-120B; DynaBench (out-of-distribution dataset); out-of-distribution domains; Jaccard; arm C; thinking off

Comparison: **Binary correctness C versus Jaccard C**. The difference is **Binary correctness C minus Jaccard C**.

| Metric | Jaccard C | Binary correctness C | Difference or evidence bounds | Adjusted p | 95% sampling interval |
|---|---:|---:|---:|---:|---:|
| Rule F1 | 0.352 | 0.458 | 0.107 | 0.002 | 0.074 to 0.138 |
| Jaccard similarity | 0.344 | 0.452 | 0.108 | 0.002 | 0.076 to 0.140 |
| Precision | 0.367 | 0.473 | 0.106 | 0.002 | 0.073 to 0.138 |
| Recall | 0.347 | 0.453 | 0.106 | 0.002 | 0.074 to 0.138 |
| Exact rule-set accuracy | 0.323 | 0.436 | 0.113 | 0.002 | 0.081 to 0.145 |
| Parseable answers | 0.747 | 0.823 | 0.076 | 0.002 | 0.059 to 0.093 |
| Mean confidence | 0.860 | 0.569 | -0.290 | 0.002 | -0.299 to -0.282 |
| Confidence standard deviation | 0.123 | 0.158 | 0.035 | 0.002 | 0.028 to 0.042 |
| Expected calibration error (10 bins) | 0.396 | 0.184 | -0.212 | 0.002 | -0.246 to -0.166 |
| Error detection: AUROC | 0.599 | 0.550 | -0.049 | 0.248 | -0.083 to -0.017 |
| Error detection: average precision | 0.673 | 0.521 | -0.151 | 0.002 | -0.194 to -0.107 |
| F1 loss under confidence-based selection: AURC | 0.478 | 0.417 | -0.061 | 0.149 | -0.111 to -0.011 |
| F1 loss found by 20% review | 0.263 | 0.227 | -0.036 | 0.031 | -0.060 to -0.014 |
| Confidence–F1 rank correlation | 0.161 | 0.075 | -0.086 | 0.224 | -0.145 to -0.029 |
| Mean supported rule fraction per answer | 0.124 to 0.124 (bounds) | 0.099 to 0.100 (bounds) | -0.025 to -0.024 (bounds) | — | — |

## Complete exports

- [correctness.csv](tables/correctness.csv)
- [configuration-means.csv](tables/configuration-means.csv)
- [configuration-leaders.csv](tables/configuration-leaders.csv)
