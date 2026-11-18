"""Portable entrypoints for the paper's evidence and correctness inference."""

import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .data import OVERRIDES, cells, set_override
from .inference import ENDPOINTS, adjust, calibration_scope
from .io import ROOT, read_json, write_json
from .vendor import correctness_inference as ci
from .vendor import evidence_inference as ei

EVIDENCE_ENDPOINTS = ENDPOINTS + ("supported_predicted_fraction", "all_predicted_rules_supported")


def evidence_cells(family):
    """Represent unknown support as [0, 1], preserving empty predictions at zero."""
    out = {}
    for key, c in cells("ood", family).items():
        lo = []
        hi = []
        for value, count in zip(c["values"], c["predicted_counts"]):
            frac = value[5]
            all_supported = float(frac == 1.0 and count > 0) if frac is not None else None
            a, b = ei.values(count, frac, all_supported)
            lo.append(a)
            hi.append(b)
        out[(family,) + key] = dict(c, lower=lo, upper=hi)
    return out


def evidence_scope(spec):
    family, target, source, mode, regime, contrast = spec
    high, low = contrast.split("-")
    row = calibration_scope(
        family,
        target,
        source,
        mode,
        regime,
        paired_seeds=ei.seeds(family, target, contrast),
        arms=(low, high),
    )
    bounded = None
    if row["complete"]:
        bounded = ei.sensitivity(evidence_cells(family), family, target, source, mode, contrast)
    for m in EVIDENCE_ENDPOINTS[-2:]:
        t = bounded["metrics"][m] if bounded else None
        known = (
            t is not None
            and abs(t["identification_bounds"][1] - t["identification_bounds"][0]) < 1e-15
        )
        row["metrics"][m] = dict(
            difference=t["identification_bounds"][0] if known else None,
            percentile95=t["bootstrap95_envelope"] if known else None,
            raw_p=t["conservative_permutation_p_upper"] if known else 1.0,
            available=known,
        )
    return row, bounded


def evidence_all(output, workers=2):
    """Recompute the ordinary recipe tests and the separate support-only family.

    The two families have different hypotheses and separate p-value adjustments.
    Keeping ``support_sensitivity`` separate prevents an evidence bound from being
    mistaken for an ordinary point estimate or percentile interval.
    """
    specs = list(
        itertools.product(
            ("Qwen", "GPT"),
            ("binary", "iou"),
            ("safepyramid", "dynabench"),
            ("OFF", "TWO_STAGE_ON"),
            ("ID", "OOD"),
            ("C-B", "D-B"),
        )
    )
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_workers, initargs=(OVERRIDES,)
    ) as pool:
        result = list(pool.map(evidence_scope, specs))
    ordinary = adjust([r for r, b in result], EVIDENCE_ENDPOINTS)
    support = [b for r, b in result if b]
    extras = list(
        itertools.product(
            ("Qwen",),
            ("partial",),
            ("safepyramid", "dynabench"),
            ("OFF", "TWO_STAGE_ON"),
            ("C-B", "D-B"),
        )
    )
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_workers, initargs=(OVERRIDES,)
    ) as pool:
        support.extend(pool.map(support_scope, extras))
    adjust_support(support)
    data = dict(
        schema="portable-evidence-inference-v1",
        family_size=512,
        results=ordinary,
        support_sensitivity=support,
        provider_calls=0,
    )
    write_json(output, data)
    return data


def correctness_work(row):
    family, target, arm, source, regime, mode = [
        row[k] for k in ["family", "high_target", "arm", "source", "regime", "mode"]
    ]
    idx = cells("id" if regime == "ID" else "ood", family)
    ss = row["paired_seeds"]
    domains = (
        ["privacy", "fraud", "defamation"]
        if regime == "ID"
        else ["content_moderation", "specialized_advice", "intellectual_property"]
    )
    required = [
        (t, arm, s, mode, source + ":" + d)
        for t, s, d in itertools.product(("binary", target), ss, domains)
    ]
    convs = sorted({x for k in required for x in idx[k]["conversation_ids"]})
    ci0 = {x: i for i, x in enumerate(convs)}
    arrays = []
    for seed, domain in itertools.product(ss, domains):
        a, b = [idx[t, arm, seed, mode, source + ":" + domain] for t in ("binary", target)]
        for field in ["row_ids", "draws", "conversation_ids"]:
            if a[field] != b[field]:
                raise ValueError("Paired correctness tasks differ")
        arrays.append(
            dict(
                seed=seed,
                domain=domain,
                a=a["values"],
                b=b["values"],
                ix=[ci0[c] for c in a["conversation_ids"]],
            )
        )
    return dict(
        family=family,
        high_target=target,
        arm=arm,
        source=source,
        regime=regime,
        mode=mode,
        paired_seeds=ss,
        conversations=len(convs),
        arrays=arrays,
    )


def correctness_scope(path):
    with np.errstate(all="ignore"):
        row = ci.run(str(path))
    for t in row["tests"].values():
        t["available"] = bool(
            t["available"]
            and t["defined_bootstrap"] >= 0.95 * 1999
            and t["defined_null"] >= 0.95 * 1999
        )
        if not t["available"]:
            t["raw_p"] = 1.0
    return row


def correctness_all(output, workers=2):
    """Rerun the frozen within-arm partial-versus-binary comparisons.

    The JSON retains the source orientation (partial minus binary). The CSV writer
    reverses both the effect and interval to match the paper's binary-minus-partial
    presentation. The eligible paired seeds come from the frozen test plan.
    """
    folder = Path(output).parent / "correctness-work"
    folder.mkdir(parents=True, exist_ok=True)
    work = []
    for i, row in enumerate(
        read_json(ROOT / "data/statistics/correctness_tests.json.gz")["results"]
    ):
        p = folder / f"work_{i:03d}.json.gz"
        write_json(p, correctness_work(row))
        work.append(p)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_workers, initargs=(OVERRIDES,)
    ) as pool:
        rows = list(pool.map(correctness_scope, work))
    slots = [t for r in rows for t in r["tests"].values()]
    assert len(slots) == 1008
    # Share the exact adjustment implementation while retaining the source's test mapping.
    adapted = [{"metrics": r["tests"]} for r in rows]
    adjust(adapted, ci.METRICS)
    data = dict(
        schema="portable-correctness-conservative-v2",
        family_size=1008,
        results=rows,
        provider_calls=0,
    )
    write_json(output, data)
    return data


def initialize_workers(overrides):
    np.seterr(all="ignore")
    for (panel, family), path in overrides.items():
        set_override(panel, family, path)


def calibration_f1_all(output, workers=2):
    rows = [
        calibration_scope("Qwen", "partial", source, mode, regime)
        for source, mode, regime in itertools.product(
            ("safepyramid", "dynabench"), ("OFF", "TWO_STAGE_ON"), ("ID", "OOD")
        )
    ]
    data = dict(
        schema="portable-qwen-f1-calibration-v1",
        family_size=48,
        results=adjust(rows),
        provider_calls=0,
    )
    write_json(output, data)
    return data


def write_test_csv(data, output):
    """Export tests with explicit effect directions and separate evidence bounds.

    Evidence's conservative support results receive their own ``.support.csv``;
    they use a different adjustment family from the ordinary recipe tests.
    """
    from .reproduce import csv_write

    rows = data.get("results", [data])
    out = []
    for row in rows:
        for metric, test in row.get("tests", row.get("metrics", {})).items():
            item = {
                k: v
                for k, v in row.items()
                if k not in ("tests", "metrics", "arms", "point", "missing_cells")
                and not isinstance(v, dict)
            }
            item.update(
                metric=metric,
                reference_value=test.get("binary"),
                compared_value=test.get("high"),
                difference=test.get("difference"),
                interval95=test.get("percentile95"),
                raw_p=test.get("raw_p"),
                adjusted_p=test.get("BH_q"),
                available=test.get("available"),
            )
            if data.get("schema") == "portable-correctness-conservative-v2":
                # Reverse the source orientation exactly as the paper export does.
                item.update(
                    reference_value=test.get("high"),
                    compared_value=test.get("binary"),
                    contrast="binary-minus-" + row["high_target"],
                    difference=-test["difference"] if test.get("difference") is not None else None,
                    interval95=[-v for v in reversed(test["percentile95"])]
                    if test.get("percentile95")
                    else None,
                )
            item["effect_direction"] = item.get("contrast")
            out.append(item)
    path = Path(output)
    stem = path.name.removesuffix(".gz").removesuffix(".json")
    csv_write(path.with_name(stem + ".csv"), out)
    if "support_sensitivity" in data:
        support = []
        for row in data["support_sensitivity"]:
            for metric, test in row["metrics"].items():
                support.append(
                    dict(
                        **{k: v for k, v in row.items() if k != "metrics"},
                        metric=metric,
                        effect_direction=row["contrast"],
                        evidence_bounds=test.get("identification_bounds"),
                        interval95=test.get("bootstrap95_envelope"),
                        raw_p_upper=test.get("conservative_permutation_p_upper"),
                        adjusted_p_upper=test.get("secondary_BH512_upper"),
                        p_adjustment="Benjamini–Hochberg; separate support-only family",
                        family_size=512,
                    )
                )
        csv_write(path.with_name(stem + ".support.csv"), support)


def support_scope(spec):
    family, target, source, mode, contrast = spec
    return ei.sensitivity(evidence_cells(family), family, target, source, mode, contrast)


def adjust_support(rows):
    # This is the distinct, frozen support-only secondary family: three metrics
    # over all five model/target combinations, with the remaining slots at p=1.
    slots = [r["metrics"][m] for r in rows for m in ei.NAMES]
    if len(slots) != 120:
        raise ValueError("The full declared support family is required")
    adapted = []
    for t in slots:
        adapted.append({"metrics": {"p": {"raw_p": t["conservative_permutation_p_upper"]}}})
    adapted.extend({"metrics": {"p": {"raw_p": 1.0}}} for _ in range(512 - len(slots)))
    adjust(adapted, ("p",))
    for t, a in zip(slots, adapted):
        t.update(
            secondary_BH512_upper=a["metrics"]["p"]["BH_q"],
            secondary_Holm512_upper=a["metrics"]["p"]["Holm_p"],
        )
