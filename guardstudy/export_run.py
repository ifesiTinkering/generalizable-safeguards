"""Convert a completed sampling/grading run into portable scored cells."""

from collections import defaultdict

from . import evaluation
from .io import ROOT, read_json, sha256, write_json
from .metrics import COLUMNS, summarize
from .reproduce import csv_write


def export(cfg, output):
    rows = evaluation.source_rows(cfg)
    chosen = {r["checkpoint_id"]: r for r in evaluation.selected(cfg)}
    panel = read_json(ROOT / f"data/panels/{cfg['panel']}.json")
    descriptors = {r["row_id"]: r for r in panel.get("rows", [])}
    if cfg["panel"] == "policy":
        descriptors = {r["row_id"]: r for t in panel["triplets"] for r in t["levels"]}
    from study import bootstrap

    bootstrap.install(next(iter(chosen.values()))["model_family"].lower())
    from reward_pipeline import parse

    buckets = defaultdict(list)
    seen = set()
    spec = evaluation.plan(cfg)
    if read_json(evaluation.resolve_path(cfg["output_directory"]) / "configuration.json") != cfg:
        raise ValueError("Source run configuration differs")
    for path in (evaluation.resolve_path(cfg["output_directory"]) / "records").glob(
        "*/raw.json.gz"
    ):
        raw = read_json(path)
        cp = chosen[raw["checkpoint_id"]]
        if raw.get("checkpoint_identity_sha256") != cp["checkpoint_identity_sha256"]:
            raise ValueError("Raw answer checkpoint identity differs from the selected roster")
        rid = raw["row_id"]
        descriptor = descriptors[rid]
        if (
            raw["mode"] not in spec["modes"]
            or type(raw["draw"]) is not int
            or not 0 <= raw["draw"] < spec["draws_per_question"]
        ):
            raise ValueError("Task lies outside the configured evaluation grid")
        key = (cp["checkpoint_id"], rid, raw["mode"], raw["draw"])
        if key in seen:
            raise ValueError("Duplicate answer task")
        seen.add(key)
        scores = read_json(path.with_name("scored.json.gz"))
        if scores["raw_sha256"] != sha256(path):
            raise ValueError("Raw answer changed after grading")
        score = scores["scores"]
        valid = score["valid"]
        gold = set(rows[rid]["ground_truth_rule_ids"])
        pred = (
            set(
                parse(raw["text"], [str(r["rule_id"]) for r in rows[rid]["all_policy_rules"]])[
                    "violated_rule_ids"
                ]
            )
            if valid
            else set()
        )
        tp = len(gold & pred)
        empty = not (gold or pred)
        frac = score.get("evidence_fraction")
        vector = [
            int(valid),
            score.get("confidence", 0.0) if valid else 0.0,
            score.get("f1", 0.0) if valid else 0.0,
            score.get("iou", 0.0) if valid else 0.0,
            score["exact"] if valid else 0.0,
            frac,
            frac * len(pred) if frac is not None else None,
            tp / len(pred) if valid and pred else float(valid and empty),
            tp / len(gold) if valid and gold else float(valid and empty),
        ]
        target = cp["internal_target"]
        arm = None if target == "SFT" else cp["arm"]
        seed = None if target == "SFT" else cp["seed"]
        cell = descriptor["source"] + ":" + descriptor["domain"]
        ident = (target, arm, seed, raw["mode"], cell)
        buckets[ident].append(
            (
                rid,
                raw["draw"],
                descriptor["conversation_hash"],
                vector,
                len(pred),
                cp["checkpoint_id"],
            )
        )
    if len(seen) != spec["answer_slots"]:
        raise ValueError("The selected sampling/grading grid is incomplete")
    cells = []
    table = []
    for ident, answers in sorted(buckets.items(), key=lambda x: str(x[0])):
        answers.sort(key=lambda r: (r[0], r[1]))
        c = dict(
            identity=dict(zip(["target", "arm", "seed", "mode", "cell"], ident)),
            checkpoint_id=answers[0][5],
            row_ids=[r[0] for r in answers],
            draws=[r[1] for r in answers],
            conversation_ids=[r[2] for r in answers],
            values=[r[3] for r in answers],
            predicted_counts=[r[4] for r in answers],
        )
        cells.append(c)
        table.append(
            dict(
                checkpoint_id=c["checkpoint_id"],
                **c["identity"],
                answers=len(answers),
                **summarize(c["values"]),
            )
        )
    output = evaluation.resolve_path(output)
    write_json(
        output,
        dict(
            schema="scored-observation-cells-v1",
            panel=cfg["panel"],
            model_family=spec["family"],
            columns=COLUMNS,
            cells=cells,
            records=len(seen),
            source_configuration_sha256=sha256(
                evaluation.resolve_path(cfg["output_directory"]) / "configuration.json"
            ),
            provider_calls=0,
        ),
    )
    csv_write(output.with_suffix(".metrics.csv"), table)
    return output
