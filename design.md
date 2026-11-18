# Study design

## Question

How do correctness, calibration, and evidence reward choices affect safeguard performance on out-of-distribution domains and datasets?

Confidence matters when a team uses model scores to decide which decisions need human review. Thinking-on and thinking-off evaluations check whether reward effects carry over to both inference modes.

The model receives a conversation and a policy. It predicts the violated rule identifiers, cites evidence, explains its answer, and reports confidence. Every reward arm uses the same output prompt.

## Models and reward choices

| Setting | Qwen3.5-9B | GPT-OSS-120B |
|---|---|---|
| Adapter rank | 64 | 32 |
| Training seeds | 0, 1, 2, 3, 4, 5, 7, 36, 42 | Subset 2, 4, 5 |
| Correctness targets | Exact set, Jaccard, set F1 | Exact set, Jaccard |
| Reward arms | A, B, C, D | A, B, C, D |
| Report checkpoint selection | Best eligible validation checkpoint in updates 100-155 | Fixed accepted terminal checkpoints; six evaluated C checkpoints |

Let `T` be the chosen correctness score and `q` be reported confidence. Let `E_all` be one when every predicted rule has supporting evidence. Let `E_fraction` be the fraction of predicted rules with supporting evidence.

| Arm | Reward |
|---|---|
| A | `T` |
| B | `T - 0.5 (q - T)^2` |
| C | `0.5 T + 0.5 E_all - 0.5 (q - T)^2` |
| D | `0.5 T + 0.5 E_fraction - 0.5 (q - T)^2` |

- An exactly correct empty prediction earns correctness credit. Empty predictions earn no evidence bonus.
- A malformed final answer earns -0.6.
- B versus A isolates the calibration term. C or D versus B changes the evidence recipe and halves the correctness coefficient. This comparison measures the whole recipe change.
- C versus D compares aggregate and fractional evidence credit at the same coefficient.

## Data

SafePyramid is the in-distribution dataset. DynaBench is the out-of-distribution dataset. Privacy, fraud, and defamation are the in-distribution domains.

| Training source | Privacy: positive / negative | Fraud: positive / negative | Defamation: positive / negative |
|---|---:|---:|---:|
| SafePyramid | 130 / 10 | 130 / 10 | 130 / 10 |
| PolicyTraj supplement | 359 / 187 | 370 / 187 | 77 / 30 |

SafePyramid negatives were constructed for the study. The supplement was appended to the training queues. Checkpoints stopped at different points and therefore consumed different amounts of the supplement. The run records preserve that exposure. Source-row and conversation identifiers support checks for overlap between training, validation, and evaluation.

## Evaluation panels

| Panel | Dataset and domains | Questions | Answers per question | Thinking |
|---|---|---:|---:|---|
| Out-of-distribution domains | SafePyramid: content moderation, specialized advice, intellectual property | 328 | 3 | On and off |
| Out-of-distribution domains | DynaBench: the same three domains | 598 | 3 | On and off |
| In-distribution domains | SafePyramid: privacy, fraud, defamation | 180 | 1 | On |
| In-distribution domains | DynaBench: privacy, fraud, defamation | 220 | 1 | On |
| Policy levels | SafePyramid: four conversations in each of seven domains; L0, L1, L2 for each conversation | 84 policy-specific questions | 1 | On |

The policy-level panel adds discrimination to the six domains above. All of its questions have at least one ground-truth violation. It fixes each conversation and changes its policy level. It supports the policy-detail analysis nested under the calibration findings.

## Code boundaries

```text
study/                 Historical scientific training modules and fresh-run adapter
  vendor/shared/       Rewards, parsing, evidence grading, validation, update
  vendor/qwen/         Qwen-specific settings
  vendor/gpt/          GPT-specific settings
  vendor/operations/   Preserved training recovery and selection helpers
  portable/            Portable queue and admission integration

guardstudy/            Shared user-facing package
  metrics.py           Sixteen recorded metrics; fifteen printed metrics
  data.py              Matched checkpoint, domain, and question access
  inference.py         Portable conversation-paired calibration inference
  reproduce.py         Tables, CSV exports, and local reading view
  evaluation.py        Native sampling and separate/streaming grading
  native/              Qwen and GPT output framing and two-stage conditioning
  archives.py          Deterministic compressed answer shards

data/                   Committed scored data, row descriptors, run roster, tests
paper/                  Exact frozen current paper and LaTeX sources
private/                Local source payloads, provider bindings, original answer archives
provenance/             Source hashes, statistical plans, and migration checks
```

## Reproduction boundaries

- Offline reproduction uses committed scored observations. It does not need a provider account or a running RunPod.
- Recomputed calibration inference uses the same conversation clusters, metric definitions, seeds, random-number sequence, and multiple-testing family as the saved analysis.
- The evidence and correctness inference commands recompute their declared families from the scored observations. The original source and plans are retained. In-domain calibration/SFT tests and policy-level tests are replayed from their frozen packets.
- A fresh training run uses the final documented recipe. It will not recreate every historical recovery, mid-training protocol change, or provider interruption. The original run records identify the checkpoints used in the paper.
- Live sampling preserves every reasoning token used to condition the final answer. Only the final answer is scored. Unknown provider outcomes are retained for reconciliation and are never automatically resubmitted.
