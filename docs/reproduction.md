# Reproduce claims 1, 2, and 3

## 1. Start from a checkout or the review ZIP

Run these commands in the project folder containing `pyproject.toml`. Python 3.12 is the tested offline environment. The commands below target macOS and Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
safeguards verify --full
safeguards reproduce --open
python -m pytest -q
```

`verify --full` checks the frozen scientific code, paper, numerical inputs, and scored observations. It recalculates the metric summaries and paired comparison means and replays the recorded Qwen checkpoint choices.

`reproduce` writes the three claim pages, a combined HTML view, complete CSV exports, and machine-readable comparisons under `results/reproduced/`. The command needs no keys or model access. Use `--output results/my-review` to keep a separate copy. Use `safeguards reproduce` without `--open` on a headless machine.

## 2. Read each claim's tables

Every comparison identifies the model, dataset, domain group, thinking mode, correctness target, reward arms, metric, eligible paired seeds, and statistical result. The complete CSVs retain the fixed fifteen-metric set. The short Markdown and HTML pages select useful comparisons for reading.

### Claim 1: Calibration

- Open [the Claim 1 page](../results/reproduced/claim-1.md) or the calibration section of [the HTML review](../results/reproduced/review.html#calibration).
- Open [calibration.csv](../results/reproduced/tables/calibration.csv) for every exported B/A comparison. `low_value` is A. `high_value` is B. `difference` is B minus A.
- Open [sft-controls-ood.csv](../results/reproduced/tables/sft-controls-ood.csv) and [sft-controls-id.csv](../results/reproduced/tables/sft-controls-id.csv) for the baseline comparisons.
- Open [policy-levels.csv](../results/reproduced/tables/policy-levels.csv) for the fixed-conversation policy-detail analysis nested under this claim.

The paper's calibration tables use `ood_calibration.json.gz`, `ood_latest_C.json`, `qwen_f1_calibration.json.gz`, `id_inference.json.gz`, the SFT-control packets, and `policy_adjacent.json.gz` from `data/statistics/`. [The packet map](../data/statistics/INDEX.md) identifies the authoritative versions.

### Claim 2: Evidence with calibration

- Open [the Claim 2 page](../results/reproduced/claim-2.md) or the evidence section of [the HTML review](../results/reproduced/review.html#evidence).
- Open [evidence.csv](../results/reproduced/tables/evidence.csv) for C/B and D/B across both datasets and available thinking modes. `low_value` is B. `high_value` is C or D. `difference` is the compared arm minus B.
- For missing semantic grades, read `evidence_bounds`, `evidence_interval95`, and `evidence_adjusted_p_upper`. The first is the possible effect allowed by missing judgments. The second is a sampling interval envelope. The last is a conservative upper bound on an adjusted p-value.

The out-of-distribution domain inputs are in `ood_latest_C.json`. The in-distribution domain inputs are in `id_inference.json.gz`. Adding C or D also halves the correctness coefficient. These comparisons estimate the whole reward-recipe change.

### Claim 3: Correctness rewards

- Open [the Claim 3 page](../results/reproduced/claim-3.md) or the correctness section of [the HTML review](../results/reproduced/review.html#correctness).
- Open [correctness.csv](../results/reproduced/tables/correctness.csv) for binary versus Jaccard or set F1 within the same arm. `low_value` is partial-credit correctness. `high_value` is binary correctness. `difference` is binary minus partial.
- Open [configuration-means.csv](../results/reproduced/tables/configuration-means.csv) and [configuration-leaders.csv](../results/reproduced/tables/configuration-leaders.csv) for descriptive rankings. Available seed sets differ in those rankings; their leaders are separate from the paired significance tests.

The inputs are `correctness_points.json` and `correctness_tests.json.gz`. The source statistical packet stores partial minus binary. The table exporter reverses the effect and both interval endpoints together. The p-value is unchanged.

## 3. Recalculate statistics from scored observations

Exporting the saved tables reuses the recorded p-values and intervals. These additional commands rerun the portable statistical analyses:

```bash
# Claim 1: full original broad-panel calibration family.
safeguards infer --output results/recomputed/calibration.json.gz

# Claim 1: separate Qwen set-F1 calibration family.
safeguards recompute --analysis calibration-f1 --output results/recomputed/calibration-f1.json.gz

# Claim 2: recipe tests plus the separate conservative support tests.
safeguards recompute --analysis evidence --workers 2 --output results/recomputed/evidence.json.gz

# Claim 3: matched correctness tests.
safeguards recompute --analysis correctness --workers 2 --output results/recomputed/correctness.json.gz
```

| Command | Recalculated outputs |
|---|---|
| `infer` | `calibration.json.gz`, `calibration.csv` |
| `recompute --analysis calibration-f1` | `calibration-f1.json.gz`, `calibration-f1.csv` |
| `recompute --analysis evidence` | `evidence.json.gz`, `evidence.csv`, `evidence.support.csv` |
| `recompute --analysis correctness` | `correctness.json.gz`, `correctness.csv` |

The evidence JSON retains the ordinary tests in `results` and the conservative support tests in `support_sensitivity`. Their p-values are adjusted within separate frozen families. The support CSV makes the latter results directly readable. The correctness JSON retains its original partial-minus-binary orientation; its CSV uses binary minus partial, matching the paper export.

The broad-panel inference commands are local CPU work. They resample and exchange labels at the conversation level, preserving repeated answers and policy views within a conversation. They make no model requests. A full-family run is needed to reproduce adjusted p-values. For a quicker raw-test check of one calibration scope:

```bash
safeguards infer --family GPT --target iou --source safepyramid \
  --mode TWO_STAGE_ON --output results/recomputed/gpt-calibration-check.json
```

`iou` means Jaccard. `partial` means set F1. `OFF` means thinking off. `TWO_STAGE_ON` means thinking on with separate reasoning and answer stages.

### Statistical reproduction coverage

| Analysis | Portable recalculation | Saved-statistics export |
|---|---|---|
| Broad-panel calibration, binary and Jaccard | `infer` | `reproduce` |
| Broad-panel Qwen set-F1 calibration | `recompute --analysis calibration-f1` | `reproduce` |
| Broad-panel evidence and missing-grade sensitivity | `recompute --analysis evidence` | `reproduce` |
| Matched correctness, including in-distribution domains | `recompute --analysis correctness` | `reproduce` |
| In-distribution domain calibration and evidence | Original scripts retained for inspection | `reproduce` |
| SFT-control tests and policy-level tests | Original scripts retained for inspection | `reproduce` |

The historical scripts in `provenance/original_analysis/` retain their original campaign paths. The public entrypoints above cover the portable analyses. [Statistical plans](../provenance/plans/) and [the metrics guide](statistics.md) specify the original denominators and test families.

Recalculated results go to a separate directory. They do not replace the frozen statistical packets or alter a paper table automatically.

After running all four commands with their default archived observations, compare every recorded inferential endpoint with its saved result:

```bash
python scripts/check_inference_parity.py --directory results/recomputed
```

The check requires the complete original scopes and reports any differing estimates, intervals, p-values, availability flags, or evidence bounds. Use it to verify code changes on the archived data. New model answers can produce different results and should be analyzed as a new experiment.

## 4. Trace a table to the exact paper source

| Claim | Paper source files |
|---|---|
| Calibration | `page2_calibration.tex`, `final_calibration_table.tex`, `final_indomain_table.tex`, `sft_control_compact.tex`, `policy_trend_compact.tex` |
| Evidence with calibration | `page5_evidence.tex`, `final_evidence_table.tex` |
| Correctness rewards | `page6_correctness.tex`, `correctness_matched_fixed15.tex`, `configuration_leaders_compact.tex` |

The exact active LaTeX set is recorded in `provenance/paper_sources.json`. `reproduce` generates readable analysis tables and copies the archived paper. It does not rewrite the paper's hand-formatted LaTeX tables. To compile those exact sources with a suitable LaTeX installation:

```bash
cd paper
latexmk -pdf main.tex
```

## 5. Collect new answers or train new checkpoints

Continue with [sampling and grading](evaluation.md#prepare-local-inputs) to collect new answers, or [fresh RL training](training.md#launch-a-fresh-rl-run) to train a new checkpoint. Both require source artifacts and accessible weights in addition to provider credentials. The archived paper remains fixed while you run a new experiment.
