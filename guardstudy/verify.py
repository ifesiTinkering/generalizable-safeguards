"""Integrity, coverage, metric parity, and publication checks."""

from study.config import verify_sources

from .data import paired_points
from .io import ROOT, read_json, sha256
from .metrics import METRICS, summarize


def verify(full=False):
    results = {"provider_calls": 0, "frozen_scientific_files": verify_sources()}
    results["frozen_analysis_inputs"] = verify_analysis_inputs(ROOT)
    inv = read_json(ROOT / "provenance/observation_inventory.json")
    total = 0
    for name, item in inv.items():
        path = ROOT / "data/observations" / (name + ".json.gz")
        if sha256(path) != item["sha256"]:
            raise ValueError("Changed observation artifact: " + name)
        total += item["records"]
    results["scored_observations"] = total
    pm = read_json(ROOT / "provenance/paper_sources.json")
    for item in pm["active_files"]:
        if sha256(ROOT / "paper" / item["name"]) != item["sha256"]:
            raise ValueError("Changed paper source: " + item["name"])
    results["paper_pdf_sha256"] = sha256(ROOT / "paper/report.pdf")
    if (
        results["paper_pdf_sha256"]
        != "02b1f331b649907140093f314eb607dc061c3d3cbf4de580068a3cb1ec2dcd38"
    ):
        raise ValueError("Changed frozen paper PDF")
    results["checkpoint_count"] = len(read_json(ROOT / "data/runs/checkpoints.json")["checkpoints"])
    if full:
        refs = read_json(ROOT / "data/observations/REFERENCE_CELL_METRICS.json.gz")
        checks = 0
        maximum = 0.0
        for name in ("ood-qwen", "ood-gpt", "ood-qwen-sft"):
            for c in read_json(ROOT / f"data/observations/{name}.json.gz")["cells"]:
                i = c["identity"]
                key = (
                    name
                    + "|"
                    + "|".join(str(i[k]) for k in ("target", "arm", "seed", "mode", "cell"))
                )
                r = refs.get(key)
                if r is None:
                    raise ValueError("Missing reference cell " + key)
                m = summarize(c["values"])
                for metric in METRICS:
                    if metric not in r:
                        continue
                    a, b = m[metric], r[metric]
                    if a is None or b is None:
                        if a != b:
                            raise ValueError("Changed missing metric " + key + " " + metric)
                    else:
                        error = abs(a - b)
                        maximum = max(maximum, error)
                        if error > 1e-9:
                            raise ValueError("Metric parity failed " + key + " " + metric)
                    checks += 1
        # Match the paper's actual paired denominators, including nonuniform Qwen availability.
        packet = read_json(ROOT / "data/statistics/ood_latest_C.json")
        pointchecks = 0
        for r in packet["ood_all_fixed24_points"]:
            high, low = r["contrast"].split("-")
            arms, _delta = paired_points(
                "ood",
                r["family"],
                r["target"],
                r["source"],
                r["mode"],
                low,
                high,
                r["paired_seeds"],
            )
            for arm, ms in arms.items():
                for metric, v in ms.items():
                    ref = r["arms"][arm].get(metric)
                    if v is not None and ref is not None:
                        if abs(v - ref) > 1e-9:
                            raise ValueError(
                                "Comparison point mismatch "
                                + str((r["family"], r["target"], r["contrast"], metric))
                            )
                        pointchecks += 1
        results.update(
            cell_metric_checks=checks,
            max_absolute_error=maximum,
            comparison_point_checks=pointchecks,
        )
    if full:
        results.update(verify_secondary_points())
    return results


def verify_analysis_inputs(root):
    """Bind exported means and p-values to the same archived input snapshot."""
    manifest = read_json(root / "provenance/analysis_inputs.json")
    for item in manifest["files"]:
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError("Changed archived analysis input: " + item["path"])
    return len(manifest["files"])


def verify_secondary_points():
    from .data import cells
    from .metrics import strict_mean
    from .selection import choose

    checks = 0

    def compare(actual, expected):
        nonlocal checks
        for metric in METRICS:
            if metric not in expected:
                continue
            a, b = actual[metric], expected[metric]
            checks += 1
            if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
                raise ValueError("Secondary metric parity failed: " + metric)

    for row in read_json(ROOT / "data/statistics/id_inference.json.gz")["results"]:
        for arm, group in row["full24"].items():
            if arm == "difference":
                continue
            idx = cells("id", row["family"])
            seeds = [None] if arm == "SFT" else row["paired_seeds"]
            target = "SFT" if arm == "SFT" else row["target"]
            name = None if arm == "SFT" else arm
            values = [
                summarize(
                    idx[target, name, seed, "TWO_STAGE_ON", row["source"] + ":" + domain]["values"]
                )
                for seed in seeds
                for domain in ("privacy", "fraud", "defamation")
            ]
            compare({m: strict_mean(v[m] for v in values) for m in METRICS}, group["mean"])
    for row in read_json(ROOT / "data/statistics/policy_metrics.json.gz")["per_checkpoint"]:
        idx = cells("policy", row["family"])
        wanted = set(row["row_ids"])
        values = []
        for key, cell in idx.items():
            if (
                key[0] == row["target"]
                and (key[1] or "baseline") == (row["arm"] or "baseline")
                and key[2] == row["seed"]
            ):
                values.extend(v for rid, v in zip(cell["row_ids"], cell["values"]) if rid in wanted)
        if len(values) != 28:
            raise ValueError("Wrong policy-level row coverage")
        compare(summarize(values), row["metrics"])
    selections = read_json(ROOT / "data/runs/checkpoint-selection.json.gz")["rows"]
    for row in selections:
        if choose(row["candidates"])["update"] != row["reported_update"]:
            raise ValueError("Checkpoint selection replay changed")
    return dict(
        in_domain_and_policy_metric_checks=checks, checkpoint_selection_replays=len(selections)
    )
