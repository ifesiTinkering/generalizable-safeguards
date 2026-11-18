# Statistical packet map

The public command uses the following authoritative packets. Original source filenames and hashes are preserved in the provenance records.

| Purpose | Current packet |
|---|---|
| Out-of-distribution domain means, evidence tests, and missing-grade sensitivity, with the fixed later C checkpoints | ood_latest_C.json |
| Out-of-distribution domain B/A calibration tests | ood_calibration.json.gz |
| Qwen set-F1 calibration secondary tests | qwen_f1_calibration.json.gz |
| In-distribution domain means and tests for all reward arms | id_inference.json.gz |
| Readable in-distribution domain calibration tables | id_reader.json |
| Baseline SFT diagnostic performance | sft_starting.json |
| Out-of-distribution domain SFT controls | ood_sft_points.json.gz, ood_sft_inference.json.gz |
| Matched binary/partial comparisons | correctness_points.json, correctness_tests.json.gz |
| Missing semantic evidence in correctness comparisons | correctness_evidence.json |
| Policy-level adjacent/full-span tests used in the current paper | policy_adjacent.json.gz |

The other packets retain the supporting analyses and their original chronology. Older OOD evidence packets predate the later C checkpoint refresh. They do not override ood_latest_C.json. The earlier policy families remain supporting records; the current table uses the corrected adjacent/full-span family. No command selects a packet by its modification time.
