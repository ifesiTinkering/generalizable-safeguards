# Matched sampling and grading

## Exact question panels

The row descriptors in `data/panels/` define membership. The same row identifiers are used across eligible checkpoints within each panel.

| Dataset | Domain | Positive | Negative | Panel |
|---|---|---:|---:|---|
| SafePyramid, in-distribution dataset | Content moderation | 100 | 10 | Out-of-distribution domains |
| SafePyramid, in-distribution dataset | Specialized advice | 100 | 10 | Out-of-distribution domains |
| SafePyramid, in-distribution dataset | Intellectual property | 98 | 10 | Out-of-distribution domains |
| DynaBench, out-of-distribution dataset | Content moderation | 99 | 99 | Out-of-distribution domains |
| DynaBench, out-of-distribution dataset | Specialized advice | 100 | 100 | Out-of-distribution domains |
| DynaBench, out-of-distribution dataset | Intellectual property | 100 | 100 | Out-of-distribution domains |
| SafePyramid, in-distribution dataset | Privacy, fraud, defamation; each domain | 50 | 10 | In-distribution domains |
| DynaBench, out-of-distribution dataset | Privacy and fraud; each domain | 50 | 40 | In-distribution domains |
| DynaBench, out-of-distribution dataset | Defamation | 19 | 21 | In-distribution domains |

- The out-of-distribution domain panel uses three answers per question in each thinking mode.
- The in-distribution domain panel uses one answer per question with thinking on.
- The policy-level panel uses four fixed conversations in each of seven SafePyramid domains. Each conversation has its L0, L1, and L2 question. It uses one answer per question with thinking on.
- Four questions with unknown physical outcomes were excluded uniformly from the original broad panel. Their identifiers and the frozen exclusion plan are retained in `provenance/plans/ood.json`.
- The repeated-answer extension is outside the current paper and this package's default experiment grid.

## Input and output

Every prompt supplies the policy, its rule inventory, and the conversation. The model emits these sections in order:

```xml
<answer>["1"]</answer>
<evidence>
  <rule id="1">
    <quote turn_id="2" speaker="assistant">Exact text from the conversation.</quote>
    <justification>The quoted text supports the predicted violation.</justification>
  </rule>
</evidence>
<analysis>The answer identifies the relevant rule.</analysis>
<confidence>0.8</confidence>
```

This is a format illustration. The paper preserves labeled examples of actual model output. Parsing and quote checks are in `study/vendor/shared/reward_pipeline.py`. Prompts are identical across arms and correctness targets.

Thinking off uses the model's native final-answer framing. Thinking on first samples up to 10,000 reasoning tokens. The final answer then has a separate allowance of 23,000 tokens. Native closure and answer-prefill tokens are recorded as conditioning. They are not reported as sampled reasoning tokens. Full reasoning is retained; the scorer receives only the final answer.

## Prepare local inputs

The checkout includes the question membership and scored observations. New sampling also needs the original text inputs and accessible weights:

| Local input | Default path | How to supply it |
|---|---|---|
| Canonical conversation and policy rows | `private/rows-ood.json.gz`, `private/rows-id.json.gz`, `private/rows-policy.json.gz` | Restore the supplied source-data bundle using [the artifact guide](artifacts.md#restore-artifact-bundles) |
| Matching tokenizer files | `private/tokenizers/qwen/`, `private/tokenizers/gpt/` | Restore the recorded tokenizers or provide matching local directories |
| Checkpoint sampler bindings | `private/checkpoint-bindings.local.json` | Supply accessible sampler URIs matching the identities in `data/runs/checkpoints.json` |
| Sampling credential | Environment variable `TINKER_API_KEY` | Set it through your shell or secret manager |
| Semantic evidence credential | Environment variable `OPENAI_API_KEY` | Needed when `semantic_grading` is true |

The private artifacts and provider bindings are local release candidates. Their public download locations have not been published. A fresh checkout can reproduce the saved tables immediately; new sampling requires those additional inputs.

Create a local configuration from the model/panel you want:

```bash
mkdir -p private
cp configs/evaluation/gpt-ood.json private/gpt-ood.local.json
```

Edit `private/gpt-ood.local.json`:

- Set `output_directory` to a new directory, such as `results/live/gpt-ood-rerun`.
- Confirm the source-row and tokenizer paths. Relative paths resolve from the project root, including when the configuration is in `private/`.
- Supply the checkpoint-binding file. Each key is a public identifier such as `gpt-jaccard-b-s2`; each value is its accessible `tinker://.../sampler_weights/...` URI. The URI hash must match the roster.
- Enter your verified Tinker input/output prices in `sampling_prices_per_million`, set `sampling_cash_limit_usd`, and set `provider_prices_verified` to true. The templates leave these unset. The sampling limit covers Tinker; semantic-judge usage is tracked separately.
- Choose `sample_workers` and `grade_workers` for your provider limits. Leave `grade_as_ready` enabled to grade answers while sampling continues.

The `.env.example` file lists the required variable names. The CLI reads the process environment; it does not automatically load `.env` files. Keep keys outside the configuration and repository.

## Plan and execute

The six committed evaluation configurations select the complete final checkpoint roster for one model and panel. They refer to local row payloads by hash. `private/checkpoint-bindings.local.json` maps public checkpoint identifiers to accessible sampler URIs.

```bash
safeguards evaluate --config configs/evaluation/qwen-ood.json
safeguards evaluate --config configs/evaluation/gpt-id.json
safeguards evaluate --config configs/evaluation/gpt-policy.json
```

These commands only print the task grid. After preparing the local files and credentials, install the recorded live dependencies in a Python 3.13.2 environment:

```bash
python -m pip install -e . -r requirements-live.txt
safeguards evaluate --config private/gpt-ood.local.json --check-inputs
safeguards evaluate --config private/gpt-ood.local.json --execute
```

`--check-inputs` validates the private files, checkpoint identities, sampling budget, and credential presence without importing the provider SDK or sending requests. `--execute` performs these checks before dispatch. The sampling ledger reserves a maximum output allowance before each request and retains reservations for unknown outcomes. A source-row checksum mismatch fails before sampling.

Sampling workers share the grid across checkpoints, rows, and draws. Worker counts are explicit configuration fields. The orchestration bounds local task materialization and makes independent requests concurrent. The provider's admitted concurrency remains a limit. This interface does not promise unlimited simultaneous provider execution.

With `grade_as_ready` enabled, answers are graded as they return. Grading has a separate worker limit. It can also run separately:

```bash
safeguards grade --config private/gpt-ood.local.json --check-inputs
safeguards grade --config private/gpt-ood.local.json --execute
```

The semantic judge uses the frozen per-rule rubric. It receives the actual predicted rules and evidence. It does not receive gold labels or reported confidence. Prepaid judge usage and Tinker sampling usage are separate accounting quantities.

Each task writes its identity before dispatch, then physical-stage journals, raw answer, assembly record when applicable, and scored result. A completed raw answer is reused. A dispatched request without a known answer is held for reconciliation and is never automatically retried. An unavailable semantic grade remains unknown; it never becomes evidence score zero.

## Analyze a new matched run

```bash
safeguards export-run --config private/gpt-ood.local.json --output results/live/gpt-ood-cells.json.gz
safeguards infer --family GPT --target iou --source safepyramid --mode TWO_STAGE_ON \
  --gpt-cells results/live/gpt-ood-cells.json.gz --output results/recomputed/new-gpt-calibration.json
```

`export-run` checks that the configured grid is complete and writes portable scored cells plus a metric CSV. A rerun never replaces the archived paper automatically. For newly trained checkpoints, supply an explicit `checkpoint_roster` JSON in the evaluation configuration and matching local sampler bindings. Each roster entry must specify the checkpoint identifier, model family, internal correctness target, arm, seed, and sampler-URI SHA256. The default configurations select the paper's fixed roster.

After exporting both models' out-of-distribution domain panels, recalculate Claim 2 with the new scored answers:

```bash
safeguards recompute --analysis evidence --workers 2 \
  --qwen-cells results/live/qwen-ood-cells.json.gz \
  --gpt-cells results/live/gpt-ood-cells.json.gz \
  --output results/recomputed/new-evidence.json.gz
```

Claim 3's fixed family includes both domain groups. Supply all four exported panels:

```bash
safeguards recompute --analysis correctness --workers 2 \
  --qwen-cells results/live/qwen-ood-cells.json.gz \
  --gpt-cells results/live/gpt-ood-cells.json.gz \
  --qwen-id-cells results/live/qwen-id-cells.json.gz \
  --gpt-id-cells results/live/gpt-id-cells.json.gz \
  --output results/recomputed/new-correctness.json.gz
```

These commands retain the frozen experiment panels, eligible seed sets, and hypothesis families. They require a complete matched grid. The CLI rejects a partial override for these full-family analyses, so archived and new observations cannot be silently mixed. The output records the input-file hashes. The saved paper results remain unchanged.
