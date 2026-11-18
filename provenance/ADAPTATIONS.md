# Consolidation changes

The scientific source files listed in `source_manifest.json` are unchanged. The current paper and its source files are unchanged.

- The fresh-run adapter sets Qwen3.5-9B's adapter rank to 64 and GPT-OSS-120B's rank to 32. The earlier portable scaffold used 32 for both. Recorded training states establish these model-specific ranks.
- The shared sampling interface uses each model's native output framing. Thinking-on sampling preserves the complete first-stage token sequence when conditioning the final-answer stage. The final answer receives its own 23,000-token allowance.
- Sampling and evidence grading have separate worker pools. Durable task records prevent automatic replay of unknown provider outcomes. The concurrency and resume tests use a fake provider and spend no model credits.
- Prepaid semantic grading is tracked separately from Tinker sampling. Exhausting a prepaid grant stops admission; it does not fall back to a cash balance.
- Portable analysis extracts the original metric and inference kernels. Its full-family replay checks reproduce the saved estimates, intervals, and adjusted p-values. The separate support-only evidence family remains separate from the ordinary comparison family.
- Public metadata uses checkpoint identity hashes instead of private provider bindings. `metadata_redactions.json` records the affected file hashes. Original metadata and full bindings remain under `private/`.
- Canonical question payloads agree across the recovered prompt caches. The exact rendered conversations have no overlap with the recovered training, SFT source-pool, or validation conversations. See `conversation_overlap_audit.json` for the checked sources and panels.

The fresh-run training adapter applies the final stopping recipe from the start. Historical campaigns included protocol changes and recoveries. The recorded checkpoint roster and validation histories identify the exact weights used in the paper; a fresh run is not presented as a bitwise reconstruction of those campaigns.

## Reader workflow pass

The README now links to generated claim pages. Evidence support tests have a separate CSV export. Correctness recalculation CSVs use the paper’s binary-minus-partial direction. New offline preflight checks run before live execution, and complete new-data overrides are required for evidence/correctness inference. The public modules add comments explaining these boundaries. Archived numerical inputs are bound by `analysis_inputs.json`; the scientific values and paper remain unchanged.

The reader pass found that the new review exporter had failed to join evidence-support results whose older packet omitted the regime field. The join now assigns those known OOD scopes explicitly. Regression coverage checks every affected saved support test. Arm bounds are shown when exact means are unavailable. The original evidence results and paper were unchanged.
