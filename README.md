# Generalizable Safeguards

How does GRPO reward design affect safeguard generalization? This codebase records the final Qwen3.5-9B and GPT-OSS-120B experiments on correctness, confidence calibration, and evidence quality.

[Repository](https://github.com/ifesiTinkering/generalizable-safeguards) · [Reproduction checks](https://github.com/ifesiTinkering/generalizable-safeguards/actions/workflows/offline.yml)

## Read the results

| Start here | What you will see |
|---|---|
| [Claim 1: Calibration](results/reproduced/claim-1.md) | B versus A; confidence accuracy, usefulness, correctness, and SFT controls |
| [Claim 2: Evidence with calibration](results/reproduced/claim-2.md) | D versus B; evidence support and the other fixed metrics |
| [Claim 3: Correctness rewards](results/reproduced/claim-3.md) | Binary versus Jaccard within the same reward arm |
| [Current paper](paper/report.pdf) | The archived five-page report and its examples |
| [Training-run reference](reports/run-tracking.pdf) | Selected checkpoints, updates, seeds, and validation history |

Each claim page gives the model, dataset, domain group, thinking mode, compared settings, metric values, intervals, and adjusted p-values. Its CSV links include the remaining settings. These pages can be read directly on GitHub. The [HTML review](results/reproduced/review.html) opens in a local browser after checkout or download.

SafePyramid is the in-distribution dataset. DynaBench is the out-of-distribution dataset. Privacy, fraud, and defamation are the in-distribution domains. The evaluation panels and their row counts are in [the evaluation guide](docs/evaluation.md#exact-question-panels).

A rewards correctness. B adds calibration. C and D combine calibration with all-or-nothing and per-rule evidence credit, respectively. The [study design](design.md#models-and-reward-choices) gives the exact equations and coefficients.

## Reproduce all three claim tables

Clone the repository and enter its folder. Python 3.12 is the tested offline environment.

```bash
git clone https://github.com/ifesiTinkering/generalizable-safeguards.git
cd generalizable-safeguards
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
safeguards verify --full
safeguards reproduce --open
```

These commands target macOS and Linux. On a server without a browser, use `safeguards reproduce` and download the generated files.

**This path requires no API keys, GPUs, RunPods, or model requests.** It uses the committed scored observations and statistical results. It writes:

| Claim | Main CSV | Additional outputs |
|---|---|---|
| 1. Calibration | [calibration.csv](results/reproduced/tables/calibration.csv) | SFT controls, policy-level comparisons, and `claim-1.md` |
| 2. Evidence with calibration | [evidence.csv](results/reproduced/tables/evidence.csv) | C/B and D/B across both datasets; `claim-2.md` |
| 3. Correctness rewards | [correctness.csv](results/reproduced/tables/correctness.csv) | Configuration means and descriptive leaders; `claim-3.md` |

All outputs are under `results/reproduced/`. To keep a separate copy, run:

```bash
safeguards reproduce --output results/my-review
python -m pytest -q
```

`verify --full` checks the frozen inputs, recalculates metrics and comparison means, and replays Qwen checkpoint selection. `reproduce` exports the saved p-values and intervals. The generated tables are analysis exports; the paper's exact typeset tables remain in `paper/*.tex`.

## Recalculate the statistical tests

These commands use the scored answers and make no model calls:

```bash
# Claim 1: the original broad-panel calibration family.
safeguards infer --output results/recomputed/calibration.json.gz

# Claim 1: the separate Qwen set-F1 calibration analysis.
safeguards recompute --analysis calibration-f1 --output results/recomputed/calibration-f1.json.gz

# Claim 2: ordinary recipe tests and conservative evidence-support tests.
safeguards recompute --analysis evidence --workers 2 --output results/recomputed/evidence.json.gz

# Claim 3: matched binary/partial comparisons.
safeguards recompute --analysis correctness --workers 2 --output results/recomputed/correctness.json.gz
```

Each command writes JSON and CSV. Evidence also writes `evidence.support.csv` with its separate support-only tests and missing-grade bounds. The in-distribution domain calibration, SFT-control, and policy-level statistics are exported from their saved test packets. Their original scripts are preserved as audit records. The [claim-by-claim walkthrough](docs/reproduction.md) identifies exactly which outputs are recalculated and which are reused.

After all four recalculations, check them against the archived statistical results:

```bash
python scripts/check_inference_parity.py --directory results/recomputed
```

## Run new model experiments

Choose the route you need:

| Route | Required inputs | Instructions |
|---|---|---|
| Re-sample the paper's checkpoints | Tinker key, accessible checkpoint weights, source-row bundle, matching tokenizers; semantic-judge key for evidence | [Sampling and grading](docs/evaluation.md#prepare-local-inputs) |
| Train a fresh RL run | The above training inputs, an accessible selected SFT checkpoint, and an explicit training budget | [Fresh training](docs/training.md#launch-a-fresh-rl-run) |

A Tinker key supplies account access. The original checkpoint weights and source artifacts must also be available to that account. Private bindings, source payloads, and large response archives are excluded from this repository. Their public artifact release locations have not been published.

You can inspect a complete evaluation plan immediately:

```bash
safeguards evaluate --config configs/evaluation/gpt-ood.json
```

After preparing a local configuration, validate it without sending requests:

```bash
safeguards evaluate --config private/gpt-ood.local.json --check-inputs
```

The evaluation guide then shows the explicit `--execute` command, how to export newly scored answers, and how to analyze them. Fresh RL training starts from the selected SFT checkpoint. Historical SFT scripts are retained for inspection.

## Find your way around the code

| Path | Purpose |
|---|---|
| [`design.md`](design.md) | Research question, reward equations, datasets, and study boundaries |
| [`guardstudy/`](guardstudy/) | Data loading, metrics, statistical tests, table exports, sampling, and grading |
| [`study/`](study/) | Fresh RL entrypoint and preserved scientific training code |
| [`configs/`](configs/) | Model-specific training examples and the six evaluation grids |
| [`data/`](data/README.md) | Scored observations, question membership, run roster, and statistical packets |
| [`paper/`](paper/) | Exact current paper sources and PDF |
| [`provenance/`](provenance/ADAPTATIONS.md) | Source hashes, original analysis scripts, plans, and consolidation changes |
| [`tests/`](tests/) | Scientific checks, export checks, and fake-provider concurrency/resume tests |

The [code walkthrough](docs/code-map.md) follows an answer from a model request through grading and into a claim table. The [metrics guide](docs/statistics.md) explains denominators, metric directions, paired seeds, confidence ties, and missing evidence grades.

The final roster contains 101 trained Qwen checkpoints, 24 trained GPT checkpoints, and one SFT baseline per model. Qwen used nine training seeds. GPT used the subset 2, 4, and 5. Each comparison records its eligible paired seeds. Reward effects are estimated within each model.

## Local archives and publication

All 767,080 saved answer records are preserved locally in compressed, indexed archives. See [artifacts and restoration](docs/artifacts.md). The source-data bundle, full answers, private checkpoint bindings, and weights are separate from the small offline review package.

This public repository includes the code, numerical observations, claim tables, and archived paper. Source-code licensing and the separate artifacts are described in [publication details](docs/publication.md). The archived Overleaf paper remains unchanged.
