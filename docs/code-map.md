# Walk through the code

## Reading or reproducing the existing results

```text
data/observations/       Scored answers, row IDs, draws, and conversation groups
         |
guardstudy/data.py       Match cells by target, arm, seed, mode, and domain
         |
guardstudy/metrics.py    Calculate the fixed metrics with their original denominators
         |
guardstudy/reproduce.py  Join comparison means to the frozen statistical packets
         |
results/reproduced/      Claim pages, full CSVs, HTML, and the archived paper
```

- `guardstudy/cli.py` defines the public commands. Offline commands do not create provider clients.
- `guardstudy/io.py` resolves the project root from the installed source location. Commands can run from another working directory. JSON writes are atomic; gzip outputs have a fixed timestamp.
- `guardstudy/data.py` loads the evaluation cells. A cell is one checkpoint and domain in one thinking mode. It retains the question, draw, and conversation identifiers needed for paired comparisons.
- `guardstudy/metrics.py` defines the reporting names and directions. Its preserved kernels handle malformed answers, tied confidence, and undefined metrics. An unavailable evidence grade stays unknown.
- `guardstudy/reproduce.py` writes the three claim pages and complete tables. `claim_highlights()` chooses examples for display. It does not choose the hypothesis family or recalculate p-values.
- `guardstudy/verify.py` checks source/data integrity and recalculates metric and checkpoint-selection results.

## Recalculating statistical tests

- `guardstudy/inference.py` implements the original conversation-paired calibration analysis. The resampling unit is the conversation. The same conversation's policy views and repeated answers move together. Original unavailable test slots remain in the declared family.
- `guardstudy/recompute.py` exposes the evidence and correctness analyses. Ordinary evidence tests and the conservative support tests have separate adjustments. The correctness JSON keeps the source orientation; its CSV writer matches the paper's binary-minus-partial orientation.
- `guardstudy/vendor/` contains the preserved numerical kernels. Their scientific behavior is retained rather than rewritten during presentation cleanup.
- `data/statistics/INDEX.md` identifies the current statistical packets. `provenance/plans/` records their families and analysis chronology.

## Collecting new answers

```text
configs/evaluation/     Checkpoints, question panel, modes, draws, worker counts
private/               Canonical rows, tokenizers, checkpoint bindings, credentials outside Git
         |
evaluation.plan()      Count the grid without private files or model calls
evaluation.preflight() Check local inputs, checkpoint hashes, budgets, key presence
         |
NativeTinker           Native prompt framing and separate reasoning/answer stages
         |
grade_file()           Parse, check quotes, and judge semantic evidence support
         |
export_run.export()    Validate the completed grid and export portable scored cells
```

`evaluation.py` keeps sampling and grading concurrent with separate worker pools. Each physical request is journaled before dispatch. Completed raw answers can be reused. A request with an unknown outcome is retained for reconciliation instead of automatically repeated.

`guardstudy/native/` preserves the model-specific token assembly. Thinking-on sampling conditions the answer on the complete reasoning-stage token sequence. Only the final answer is graded.

`export_run.py` checks row membership, draw counts, checkpoint identity, and raw/score hashes before creating a new scored-observation file. New data remain separate from the frozen paper inputs. `infer --qwen-cells` and `--gpt-cells` accept those exported files for a new broad-panel calibration analysis.

## Training a fresh RL run

- `study/config.py` validates the fixed recipe, model-specific adapter rank, data hashes, and local settings.
- `study/model_adapter.py` constructs the model renderer and canonical prompts.
- `study/runner.py` connects the preserved training engine to an isolated fresh model, worker pools, validation, and accounting.
- `study/portable/` adapts campaign-specific queue and admission behavior to supplied local paths.
- `study/vendor/shared/` holds the preserved reward, parser, training update, validation, and stopping logic. `study/vendor/qwen/` and `study/vendor/gpt/` hold model-specific code.
- `guardstudy/selection.py` expresses the validation composite and tie rule used to select the archived Qwen checkpoints.

Fresh RL runs start from an accessible selected SFT checkpoint. The original SFT code is retained under `provenance/sft_source/`. The current entrypoint does not automate that full historical SFT sequence.

## Making changes

Add interfaces, comments, and validation to the portable modules. Preserve the scientific files listed in `provenance/source_manifest.json` and the frozen paper/data inputs. An intentional scientific change should get its own configuration, output directory, and provenance rather than silently replacing the reported study.

Run `python -m pytest -q`, `safeguards verify --full`, and `safeguards reproduce` after changing the portable code. The GitHub workflow runs these offline checks. It has no model credentials or live jobs.
