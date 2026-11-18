# Training and checkpoint selection

## Starting checkpoints

Each model starts reinforcement learning from its selected supervised fine-tuning checkpoint. The SFT curricula differ. The common evaluation panel provides a shared performance reference.

| Model | Thinking | Rule F1 | Precision | Recall | Parseable answers |
|---|---|---:|---:|---:|---:|
| Qwen3.5-9B | Off | 0.516 | 0.492 | 0.568 | 0.871 |
| Qwen3.5-9B | On | 0.662 | 0.670 | 0.667 | 0.869 |
| GPT-OSS-120B | Off | 0.355 | 0.384 | 0.361 | 0.899 |
| GPT-OSS-120B | On | 0.674 | 0.713 | 0.660 | 0.968 |

These values come from `data/statistics/sft_starting.json`. They use the shared out-of-distribution domain panel. They are diagnostic evaluations, not identical SFT selection criteria.

- Qwen3.5-9B continued through its original SFT, an extension on 55 defamation examples, and format-only training. The last Qwen SFT stage trains structural output tokens. It masks rule identifiers, quoted evidence, explanations, and confidence values from the SFT loss. The selected starting checkpoint is `formatonly-epoch04`.
- GPT-OSS-120B used the recorded format-first curriculum and answer-focused continuation. The last GPT SFT stage trains the answer-list tokens with the recorded loss weight of 200. It masks the remaining output tokens. The selected starting checkpoint is the answer-only checkpoint at update 27.
- The exact target builders and SFT scripts are preserved under `provenance/sft_source/`. Their historical orchestration contains campaign-specific paths and is provided for method review. The unified fresh RL runner starts from an accessible selected SFT checkpoint; it does not recreate that SFT ancestry automatically.

## Reinforcement-learning recipe

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen3.5-9B or openai/gpt-oss-120b:peft:131072 |
| Adapter rank | 64 for Qwen; 32 for GPT |
| Prompts per attempt | Three: one from each in-distribution domain |
| Responses per prompt | Ten |
| Sampling temperature | 1.0 |
| Thinking during RL training | Off |
| Final-answer allowance | 23,000 tokens |
| Application context limit | 65,536 tokens |
| Optimizer | Adam, learning rate 0.00001; beta1 0.9; beta2 0.95; epsilon 0.00000001 |
| Validation | The same 90 questions every ten successful updates |
| Training ceiling | 155 successful updates in the final selection protocol |

The update follows the audited Tinker tutorial adaptation: raw rewards are centered within each response group, and the loss uses token-summed importance sampling. It uses no reward-standard-deviation normalization, response-length normalization, reference KL term, or clipped PPO objective. This is the implementation used for the study's GRPO-style group updates.

An attempt samples a batch. A successful update commits an optimizer step. A group with constant rewards gives no relative learning signal. Attempts and queue exposure can therefore advance without a successful update.

The data queue first consumes the SafePyramid rows, then the appended PolicyTraj rows. Exact order, seed namespaces, and cursor behavior live in `study/portable/` and `study/vendor/shared/data/`.

## Selection and stopping

For Qwen, the post-training selection score averages three components on the shared 90-question validation set:

- Correctness averages exact set accuracy, set F1, and Jaccard similarity. Malformed answers contribute zero.
- Calibration averages `1 - (confidence - exact correctness)^2`. Malformed answers contribute zero.
- Evidence averages aggregate and per-rule support. The final selection score averages these three components.

The highest completed score in the 100-155 window wins. The later update breaks a tie. `guardstudy/selection.py` implements the exact composite and tie rule. `data/runs/checkpoint-selection.json.gz` preserves the candidate component scores and replays all 101 Qwen choices.


- Qwen results use the eligible selected checkpoints in the 100-155 update window. The public roster identifies all selected updates. Historical trajectories include changes made during training; their full logs remain in the local archive.
- GPT results use the fixed accepted terminal checkpoints. The six C checkpoints are bound to the exact evaluated weights. Two binary C checkpoints were operationally paused when frozen for the paper. Their later training status does not change this report's checkpoint identity.
- The final stopping helper tracks the recorded validation components over the rolling five-checkpoint window. It requires at least two component failures at the stated 0.03 improvement threshold. Its precise metric directions and observation order are in the frozen `selection_majority_stop.py` source.
- The final recipe was adopted during the Qwen campaign and used from the GPT launch. Fresh runs use the final recipe; exact historical interruption and recovery sequences remain documented as history. A fresh stricter run can stop before the historical Qwen selection window. Exact reproduction of the reported comparisons uses the archived selected checkpoints.

## Launch a fresh RL run

```bash
python -m pip install -e . -r requirements-live.txt
python -m study verify-sources
python -m study validate --config configs/training/qwen.example.json
```

Use a separate Python 3.13.2 environment for the pinned live dependencies. Create a local configuration:

```bash
mkdir -p private
cp configs/training/qwen.example.json private/qwen-training.local.json
```

Edit that file to select `target` (`binary`, `iou` for Jaccard, or `partial` for set F1), reward `arm` (`A`, `B`, `C`, or `D`), and `seed`. Supply canonical training, validation, and supplement JSON files and their hashes. Supply an accessible SFT full-checkpoint URI and a matching local tokenizer. Set a new empty `output_directory`, an explicit sampling/training budget and storage reserve, and confirm the provider prices and checkpoint-storage settings in the template. Set `TINKER_API_KEY` and `OPENAI_API_KEY` in the environment.

Set `semantic_api_prepaid` to true when the semantic API is prepaid. In that mode, API usage is recorded separately and known judge calls settle at zero cash. The prepaid allowance never silently falls back to a cash-funded API call.

```bash
python -m study validate --config private/qwen-training.local.json --check-data
python -m study train --config private/qwen-training.local.json \
  --execute --accept-paid-operations
```

The runner creates a new isolated RL model and an empty output directory. It does not adopt an existing run or delete provider checkpoints. The preserved journals record unknown physical outcomes for explicit recovery review. Change the local output directory before launching another seed or arm.

Use the same commands for GPT with its configuration. Run the two model families in separate processes because the scientific adapters install model-specific modules.

## Read the run records

`data/runs/checkpoints.json` is the paper roster. `data/runs/training-history.json.gz` and `reports/training-history.csv.gz` contain the recovered validation records through each reported checkpoint. `reports/run-tracking.pdf` gives a compact overview. Provider identifiers and original state files remain in the ignored local archive.
