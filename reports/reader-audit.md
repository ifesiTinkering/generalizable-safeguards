# Reader workflow review

The review starts with a fresh source checkout and no private artifact directory.

- [x] A fresh Git clone passes the tests and regenerates the tables from outside the checkout.
- [x] All four README inference commands match the archived results through the parity-check script.
- [x] The README links directly to each claim's readable results, the paper, and training-run reference.
- [x] One offline command regenerates the three claim pages, HTML, and full CSVs.
- [x] The walkthrough identifies the comparison direction and numerical input for each claim.
- [x] Saved-statistics exports and statistical recalculation are described separately.
- [x] Evidence's separate conservative support tests have a readable CSV export.
- [x] Every broad-panel support test joins to its comparison row despite the historical packet's missing regime field. The regression test covers all saved binary/Jaccard C/B and D/B scopes.
- [x] Recalculated correctness CSVs use the same binary-minus-partial direction as the paper exports.
- [x] Frozen numerical packets, panels, and checkpoint metadata receive integrity checks.
- [x] Public modules explain data flow, paired observations, missing evidence, and inference boundaries in comments and docstrings.
- [x] Live configurations expose unset sampling prices and limits instead of implying a ready-to-spend budget.
- [x] Input preflight checks data, checkpoint identity, tokenizers, budgets, and credential presence without provider clients.
- [x] Saved raw answers carry checkpoint identity hashes; changed weights cannot silently reuse old answers.
- [x] New full-family evidence/correctness analyses require complete input overrides instead of mixing archived and new observations.
- [x] New RL runs have explicit starting-checkpoint, data, budget, environment, and output-directory instructions.
- [x] The README describes private artifacts and accessible weights as prerequisites for new model experiments.

The archived paper, scientific source files, observations, and statistical results remain unchanged. Historical SFT orchestration and the in-domain/SFT/policy-level statistical scripts remain inspection records with their original paths. Their current portable-recalculation coverage is stated in docs/reproduction.md.
