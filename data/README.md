# Saved study data

| Directory | Contents |
|---|---|
| `panels/` | Exact question membership, labels, gold rule identifiers, and conversation clusters |
| `runs/` | Final checkpoint roster, recorded validation history, candidate selection scores, and training row counts |
| `observations/` | Compressed scored answer cells for every study checkpoint and panel |
| `statistics/` | Original numerical test packets; see `statistics/INDEX.md` for the authoritative sources |

Each observation cell identifies the correctness target, reward arm, seed, thinking mode, dataset/domain, and public checkpoint identifier. Its row identifiers, draw identifiers, and conversation identifiers align with its value rows. The value columns are:

| Column | Meaning |
|---|---|
| `valid` | One for a parseable final answer; zero for a known malformed answer |
| `confidence` | Reported confidence for valid answers |
| `f1`, `iou`, `exact` | Rule-set correctness scores |
| `evidence_fraction` | Per-answer fraction of predicted rules with semantic evidence support |
| `supported_count` | Number of supported predicted rules |
| `precision`, `recall` | Per-answer rule-set precision and recall |

`null` evidence values represent unavailable grades. They are preserved as unknown values. They are not zeros. Known malformed answers contribute zero correctness. Confidence metrics use valid answers. A correct empty rule set earns correctness credit and zero evidence credit.

The arrays support the paper's numerical metrics without requiring a provider account or the original text datasets. Actual answer text and available reasoning text are retained in the separate local response archives. Source conversations and policies are in the local source-data artifact. Those larger files have their own manifests and publication decisions.
