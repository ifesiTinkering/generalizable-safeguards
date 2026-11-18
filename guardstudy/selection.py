"""The recorded post-training checkpoint selection score; no model operations."""


def composite(records):
    if len(records) != 90 or len({r["row_id"] for r in records}) != 90:
        raise ValueError("Exactly ninety unique validation questions are required")
    if any(
        r.get("judge_unavailable") or r.get("provider_unknown") or r.get("incomplete")
        for r in records
    ):
        raise ValueError("Unresolved validation outcome")
    valid = [r for r in records if r["valid"]]
    correctness = sum(r["exact"] + r["f1"] + r["iou"] for r in valid) / 270
    calibration = sum(1 - (r["confidence"] - r["exact"]) ** 2 for r in valid) / 90
    evidence = sum(r["evidence_all"] + r["evidence_fraction"] for r in records) / 180
    return dict(
        score=(correctness + calibration + evidence) / 3,
        correctness=correctness,
        calibration=calibration,
        evidence=evidence,
        n=90,
        n_valid=len(valid),
    )


def choose(candidates, minimum=100, maximum=155):
    complete = [
        r for r in candidates if r.get("status") == "complete" and minimum <= r["update"] <= maximum
    ]
    if not complete:
        raise ValueError("No eligible completed validation candidate")
    return max(complete, key=lambda r: (r["score"], r["update"]))
