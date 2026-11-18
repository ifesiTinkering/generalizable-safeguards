"""Build the claim tables from the archived, scored study.

Data flow: observations -> metric checks; statistical packets -> comparisons ->
CSV, Markdown, and HTML. Exporting a table reuses the frozen p-values and
intervals. The ``infer`` and ``recompute`` commands rerun the statistical tests.
No function in this module samples or grades a model answer.
"""

import csv
import html
import json
import os
import shutil
from pathlib import Path

from .data import cell_summaries, paired_points
from .io import ROOT, read_json, write_json
from .metrics import LABELS, LOWER_IS_BETTER, PRINTED_METRICS

S = ROOT / "data/statistics"
SCOPE = ("family", "target", "source", "mode", "regime", "contrast")


def key(r):
    return tuple(r.get(k) for k in SCOPE)


def load(name):
    return read_json(S / name)


def rows(name):
    return load(name)["results"]


def comparisons():
    """Return long-form claim rows with explicit paired seeds and effect direction.

    ``high_value - low_value`` defines ``difference`` in every ordinary row.
    Evidence rows additionally retain bounds for unavailable semantic judgments.
    The original correctness packet has the opposite orientation, so both its
    effect and confidence interval are reversed together below; p-values stay fixed.
    """
    output = []
    latest = load("ood_latest_C.json")
    f1 = rows("qwen_f1_calibration.json.gz")
    for r in f1:
        if r.get("complete"):
            arms, delta = paired_points(
                "ood", r["family"], "partial", r["source"], r["mode"], "A", "B", r["paired_seeds"]
            )
            latest["ood_all_fixed24_points"].append(dict(r, arms=arms, difference=delta))
    ordinary = {
        key(r): r for r in rows("ood_calibration.json.gz") + latest["ood_all_recipe_tests"] + f1
    }
    # The support-only packet predates the explicit regime field. Its rows all
    # describe OOD domains; normalize that metadata before joining the packets.
    bounds = {key(dict(r, regime="OOD")): r for r in latest["ood_all_support_sensitivity"]}
    for r in latest["ood_all_fixed24_points"]:
        high, low = r["contrast"].split("-")
        for m in PRINTED_METRICS:
            test = ordinary.get(key(r), {}).get("metrics", {}).get(m, {})
            bound = bounds.get(key(r), {}).get("metrics", {}).get(m, {})
            identified = r.get("identification_bounds", {})
            point_bounds = identified.get("difference", {}).get(m)
            o = {k: r.get(k) for k in SCOPE}
            o.update(
                panel="ood",
                metric=m,
                low=low,
                high=high,
                low_value=r["arms"][low].get(m),
                high_value=r["arms"][high].get(m),
                difference=r["difference"].get(m),
                seeds=r["paired_seeds"],
                raw_p=test.get("raw_p"),
                adjusted_p=test.get("BH_q"),
                interval95=test.get("percentile95"),
                available=test.get("available", False),
                inference="frozen conversation-paired tests",
                p_adjustment="Benjamini–Hochberg",
                family_size=(48 if r["target"] == "partial" else 192)
                if r["contrast"] == "B-A"
                else 512,
                evidence_bounds=bound.get("identification_bounds")
                or ([point_bounds["lower"], point_bounds["upper"]] if point_bounds else None),
                low_value_bounds=identified.get("arms", {}).get(low, {}).get(m),
                high_value_bounds=identified.get("arms", {}).get(high, {}).get(m),
                evidence_interval95=bound.get("bootstrap95_envelope"),
                evidence_adjusted_p_upper=bound.get("secondary_BH512_upper"),
            )
            if bound:
                o["raw_p"] = bound.get("conservative_permutation_p_upper")
                o["adjusted_p"] = bound.get("secondary_BH512_upper")
                o["interval95"] = bound.get("bootstrap95_envelope")
                o["available"] = o["adjusted_p"] is not None
                o["inference"] = "conservative missing-evidence bounds"
            output.append(o)
    sft_tests = {key(r): r for r in rows("ood_sft_inference.json.gz")}
    for r in rows("ood_sft_points.json.gz"):
        if not r.get("complete"):
            continue
        high = r["contrast"].split("_")[0]
        for m in PRINTED_METRICS:
            test = sft_tests.get(key(r), {}).get("metrics", {}).get(m, {})
            bound = (
                r.get("identification_bounds", {}).get("fixed24", {}).get("RL_minus_SFT", {}).get(m)
            )
            o = {k: r.get(k) for k in SCOPE}
            o.update(
                panel="ood-sft",
                metric=m,
                low="SFT",
                high=high,
                low_value=r["fixed24"]["SFT"].get(m),
                high_value=r["fixed24"]["RL"].get(m),
                difference=r["fixed24"]["RL_minus_SFT"].get(m),
                seeds=r["RL_seeds"],
                raw_p=test.get("raw_p"),
                adjusted_p=test.get("BH_q"),
                interval95=test.get("percentile95"),
                available=test.get("available", False),
                family_size=384,
                p_adjustment="Benjamini–Hochberg",
                evidence_bounds=[bound["lower"], bound["upper"]] if bound else None,
                inference="frozen centered conversation bootstrap; one SFT baseline",
            )
            output.append(o)
    for r in rows("id_inference.json.gz"):
        if not r.get("complete"):
            continue
        high, low = r["contrast"].split("-") if "-" in r["contrast"] else (None, None)
        # The packet names its internal arms A/B even for other recipe contrasts.
        full = r["full24"]
        a = full[low]["mean"]
        b = full[high]["mean"]
        for m in PRINTED_METRICS:
            test = r["metrics"].get(m, {})
            o = {k: r.get(k) for k in SCOPE}
            o.update(
                panel="id",
                regime="ID",
                metric=m,
                low=low,
                high=high,
                low_value=a.get(m),
                high_value=b.get(m),
                difference=full["difference"].get(m),
                seeds=r["paired_seeds"],
                raw_p=test.get("raw_p"),
                adjusted_p=test.get("BH_q"),
                interval95=test.get("percentile95"),
                available=test.get("available", False),
                inference="frozen conversation-paired tests",
                p_adjustment="Benjamini–Hochberg",
                family_size=320,
            )
            output.append(o)
    for r in rows("correctness_points.json"):
        for m in PRINTED_METRICS:
            t = r["tests"].get(m, {})
            o = {k: r.get(k) for k in SCOPE}
            o.update(
                panel="correctness",
                target=r["high_target"],
                contrast="binary-minus-" + r["high_target"],
                arm=r["arm"],
                metric=m,
                low=r["high_target"] + " " + r["arm"],
                high="binary " + r["arm"],
                low_value=t.get("high", {}).get("point")
                if isinstance(t.get("high"), dict)
                else t.get("high"),
                high_value=t.get("binary", {}).get("point")
                if isinstance(t.get("binary"), dict)
                else t.get("binary"),
                # Paper orientation: binary minus partial, within the same arm.
                difference=-t["difference"] if t.get("difference") is not None else None,
                seeds=r["paired_seeds"],
                raw_p=t.get("raw_p"),
                adjusted_p=t.get("BH_q"),
                interval95=[-v for v in reversed(t["percentile95"])]
                if t.get("percentile95")
                else None,
                available=t.get("available", False),
                p_adjustment="Benjamini–Hochberg",
                family_size=1008,
                inference="frozen conversation-paired tests",
                evidence_bounds=[-v for v in reversed(t["difference_bounds"])]
                if t.get("difference_bounds")
                else None,
                low_value_bounds=t.get("high", {}).get("bounds")
                if isinstance(t.get("high"), dict)
                else None,
                high_value_bounds=t.get("binary", {}).get("bounds")
                if isinstance(t.get("binary"), dict)
                else None,
            )
            output.append(o)
    for r in rows("policy_adjacent.json.gz"):
        o = dict(r["scope"])
        t = r["test"]
        o.update(
            panel="policy",
            source="safepyramid",
            regime="mixed_domains",
            difference=t.get("difference"),
            raw_p=t.get("raw_p"),
            adjusted_p=t.get("BH_q"),
            interval95=t.get("percentile95"),
            available=t.get("available"),
            levels=r["levels"],
            seeds=r["scope"]["paired_seeds"],
            p_adjustment="Benjamini–Hochberg",
            family_size=960,
            inference="frozen stratified conversation-paired tests",
        )
        output.append(o)
    return output


def csv_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        return
    fields = list(dict.fromkeys(k for r in data for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in data:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                    for k, v in r.items()
                }
            )


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def displayed_value(row, side):
    """Show a known arm mean or labeled bounds, never an invented mean."""
    value = row.get(side + "_value")
    if value is not None:
        return fmt(value)
    bounds = row.get(side + "_value_bounds")
    if bounds:
        if isinstance(bounds, dict):
            bounds = [bounds["lower"], bounds["upper"]]
        return " to ".join(fmt(v) for v in bounds) + " (bounds)"
    return "—"


def recipe_label(value):
    return (
        str(value)
        .replace("iou", "Jaccard")
        .replace("partial", "Set F1")
        .replace("binary", "Binary correctness")
    )


def scope_text(r):
    ds = (
        "SafePyramid (in-distribution dataset)"
        if r["source"] == "safepyramid"
        else "DynaBench (out-of-distribution dataset)"
    )
    model = "Qwen3.5-9B" if r["family"] == "Qwen" else "GPT-OSS-120B"
    dom = "in-distribution domains" if r.get("regime") == "ID" else "out-of-distribution domains"
    target = {
        "iou": "Jaccard",
        "partial": "set F1",
        "binary": "binary correctness",
    }.get(r["target"], r["target"])
    arm = "; arm " + r["arm"] if r.get("arm") else ""
    return f"{model}; {ds}; {dom}; {target}{arm}; thinking {'off' if r['mode'] == 'OFF' else 'on'}"


def claim_highlights(comparisons):
    """Select the documented examples; complete settings stay in the CSV exports.

    These predicates select display scopes, not hypotheses for p-value adjustment.
    Inference families were computed before this presentation step.
    """
    c1 = [
        r
        for r in comparisons
        if r["panel"] in ("ood", "id")
        and r.get("contrast") == "B-A"
        and r["target"] == "iou"
        and r["mode"] == "TWO_STAGE_ON"
    ]
    c1_sft = [
        r
        for r in comparisons
        if r["panel"] == "ood-sft"
        and r["contrast"] == "B_minus_SFT"
        and r["target"] == "iou"
        and r["mode"] == "TWO_STAGE_ON"
        and r["metric"] in ("rule_F1", "IoU_ECE10", "F1_AURC")
    ]
    c2 = [
        r
        for r in comparisons
        if r["panel"] == "ood"
        and r.get("contrast") == "D-B"
        and r["source"] == "safepyramid"
        and r["mode"] == "TWO_STAGE_ON"
    ]
    c3 = [
        r
        for r in comparisons
        if r["panel"] == "correctness"
        and r.get("source") == "dynabench"
        and r.get("regime") == "OOD"
        and r["mode"] == "OFF"
        and r["target"] == "iou"
        and r["arm"] in ("A", "C")
    ]
    return {1: c1, 2: c2, 3: c3, "sft": c1_sft}


def write_claim_pages(out, values):
    """Write GitHub-readable result pages using the same rows as the HTML view."""
    highlights = claim_highlights(values)
    titles = {
        1: "Calibration can improve confidence accuracy",
        2: "Evidence rewards can improve support for predicted rules",
        3: "Binary correctness rewards do not always generalize worse than partial rewards",
    }
    for number, title in titles.items():
        rows = highlights[number] + (highlights["sft"] if number == 1 else [])
        sections = {}
        # Include the contrast so C/B and D/B can never share an ambiguous header.
        for row in rows:
            sections.setdefault((scope_text(row), row["low"], row["high"]), []).append(row)
        instructions = Path(os.path.relpath(ROOT / "docs/reproduction.md", out)).as_posix()
        text = [
            f"# Claim {number}: {title}\n",
            "[All results](review.html) · [Current paper](paper/report.pdf) · "
            f"[Reproduction instructions]({instructions})\n",
            "A rewards correctness. B adds calibration. C and D combine calibration with "
            "all-or-nothing and per-rule evidence credit, respectively. C and D also halve "
            "the correctness coefficient relative to B.\n",
            "These are selected saved comparisons. Each heading gives the model, dataset, "
            "domain group, correctness reward, and thinking mode. Full settings and the "
            "fixed metric set are in the CSV files below.\n",
            "An adjusted p-value below 0.05 marks statistical support under the recorded "
            "test. A dash means that a value is unavailable or was not tested. "
            "Values use a 0–1 scale. Lower calibration error and lower selection loss are better.\n",
        ]
        if number == 2:
            text.append(
                "Evidence bounds give the possible effect of missing semantic grades. "
                "The sampling interval is shown separately. A p-value marked ≤ is a "
                "conservative upper bound.\n"
            )
        for (scope, low, high), group in sections.items():
            low, high = recipe_label(low), recipe_label(high)
            text.extend(
                [
                    f"## {scope}\n",
                    f"Comparison: **{high} versus {low}**. "
                    f"The difference is **{high} minus {low}**.\n",
                    f"| Metric | {low} | {high} | Difference or evidence bounds | Adjusted p | 95% sampling interval |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in group:
                change = fmt(row.get("difference"))
                if row.get("evidence_bounds"):
                    change = " to ".join(fmt(v) for v in row["evidence_bounds"]) + " (bounds)"
                interval = row.get("interval95")
                p = ("≤ " if row.get("evidence_adjusted_p_upper") is not None else "") + fmt(
                    row.get("adjusted_p") if row.get("available") else None
                )
                fields = [
                    LABELS[row["metric"]],
                    displayed_value(row, "low"),
                    displayed_value(row, "high"),
                    change,
                    p,
                    " to ".join(fmt(v) for v in interval) if interval else "—",
                ]
                text.append("| " + " | ".join(fields) + " |")
            text.append("")
        exports = {
            1: ("calibration", "sft-controls-ood", "sft-controls-id", "policy-levels"),
            2: ("evidence",),
            3: ("correctness", "configuration-means", "configuration-leaders"),
        }
        text.append("## Complete exports\n")
        text.extend(f"- [{name}.csv](tables/{name}.csv)" for name in exports[number])
        (out / f"claim-{number}.md").write_text("\n".join(text) + "\n")


def render_review(out, comparisons):
    def table(rs):
        groups = {}
        for r in rs:
            groups.setdefault((scope_text(r), r.get("low"), r.get("high")), []).append(r)
        pieces = []
        for (title, _low, _high), group in groups.items():
            low = html.escape(recipe_label(group[0].get("low", "Reference")))
            high = html.escape(recipe_label(group[0].get("high", "Compared recipe")))
            pieces.append(
                "<h3>"
                + html.escape(title)
                + f"</h3><table><tr><th>Metric</th><th>{low}</th><th>{high}</th><th>Compared minus reference</th><th>Adjusted p</th><th>95% interval</th></tr>"
            )
            for r in group:
                interval = r.get("interval95")
                diff = fmt(r.get("difference"))
                if r.get("evidence_bounds"):
                    diff = " to ".join(fmt(v) for v in r["evidence_bounds"]) + " (bounds)"
                p = r.get("adjusted_p")
                sig = p is not None and p < 0.05 and r.get("available")
                css = ""
                if sig:
                    if r["metric"] in ("confidence_mean", "confidence_SD"):
                        css = "descriptive"
                    else:
                        change = r.get("difference")
                        if change is None and r.get("evidence_bounds"):
                            change = r["evidence_bounds"][0]
                        if change is not None:
                            improved = change < 0 if r["metric"] in LOWER_IS_BETTER else change > 0
                            css = "significant" if improved else "deterioration"
                pieces.append(
                    "<tr"
                    + (f' class="{css}"' if css else "")
                    + ">"
                    + "".join(
                        "<td>" + html.escape(v) + "</td>"
                        for v in [
                            LABELS[r["metric"]],
                            displayed_value(r, "low"),
                            displayed_value(r, "high"),
                            diff,
                            ("≤ " if r.get("evidence_adjusted_p_upper") is not None else "")
                            + fmt(p if r.get("available") else None),
                            " to ".join(fmt(v) for v in interval) if interval else "—",
                        ]
                    )
                    + "</tr>"
                )
            pieces.append("</table>")
        return "".join(pieces)

    highlights = claim_highlights(comparisons)
    c1, c1_sft, c2, c3 = [highlights[k] for k in (1, "sft", 2, 3)]
    text = """<!doctype html><html><meta charset="utf-8"><title>Generalizable Safeguards: saved results</title><style>body{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#172435}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:14px;margin-bottom:32px}th,td{text-align:right;border-bottom:1px solid #ddd;padding:8px}th:first-child,td:first-child{text-align:left}th{background:#eef3fa}.significant{background:#e6f4ef}.deterioration{background:#fceded}.descriptive{background:#edf1f8}nav a{margin-right:20px}p{max-width:950px}h2{margin-top:48px}</style><h1>Generalizable Safeguards</h1><p>This local review reproduces the saved paper's comparisons. Statistics are conditional on the evaluated checkpoints. Green marks statistically supported improvements. Red marks statistically supported deterioration. Blue marks significant descriptive shifts. These colors use the adjusted threshold of 0.05. Lower expected calibration error and lower selection loss are better. Fractions use a 0–1 scale. The CSV exports include every setting, not only the tables below.</p><p>Evidence bounds show the range allowed by missing semantic grades. They are different from the sampling interval in the last column. A blank p-value means that endpoint was descriptive or unavailable.</p><nav><a href="#calibration">Calibration</a><a href="#evidence">Evidence</a><a href="#correctness">Correctness</a><a href="paper/report.pdf">Frozen paper</a><a href="tables/">All tables</a></nav>"""
    text += (
        '<h2 id="calibration">Claim 1: Calibration can improve confidence accuracy</h2><p>These tables compare Jaccard B with Jaccard A, with thinking on. The domain and dataset scope appears above each table.</p>'
        + table(c1)
        + "<h3>SFT baseline controls</h3><p>These compact controls show the three endpoints in the paper. The SFT-control CSV retains the full fixed metric set.</p>"
        + table(c1_sft)
    )
    text += (
        '<h2 id="evidence">Claim 2: Evidence rewards can improve support for predicted rules</h2><p>These tables compare D with B on out-of-distribution domains in the in-distribution SafePyramid dataset, with thinking on. The complete C/B and D/B comparisons for both datasets are in evidence.csv.</p>'
        + table(c2)
    )
    text += (
        '<h2 id="correctness">Claim 3: Binary correctness rewards do not always generalize worse than partial rewards</h2><p>These tables show binary minus Jaccard within the same model and arm. The examples use out-of-distribution DynaBench domains with thinking off. Other settings and the descriptive configuration leaders are exported separately.</p>'
        + table(c3)
    )
    text += "</html>"
    (out / "review.html").write_text(text)


def reproduce(output, recalculate=True):
    """Regenerate readable claim pages and CSVs without provider access.

    The frozen LaTeX files are copied as paper artifacts; this function generates
    new analysis tables, not replacement LaTeX for the typeset paper.
    """
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tab = out / "tables"
    values = comparisons()
    write_json(out / "comparisons.json.gz", values)
    for name, predicate in {
        "calibration": lambda r: r["panel"] in ("ood", "id") and r.get("contrast") == "B-A",
        "evidence": lambda r: r["panel"] in ("ood", "id") and r.get("contrast") in ("C-B", "D-B"),
        "correctness": lambda r: r["panel"] == "correctness",
        "policy-levels": lambda r: r["panel"] == "policy",
        "sft-controls-ood": lambda r: r["panel"] == "ood-sft",
        "sft-controls-id": lambda r: r["panel"] == "id" and "SFT" in r.get("contrast", ""),
    }.items():
        csv_write(tab / (name + ".csv"), [r for r in values if predicate(r)])
    csv_write(
        tab / "sft-starting.csv",
        [
            dict(family=r["family"], mode=r["mode"], **r["metrics"])
            for r in load("sft_starting.json")["rows"]
        ],
    )
    if recalculate:
        cs = []
        for panel in ("ood", "id", "policy"):
            for family in ("Qwen", "GPT"):
                cs.extend(
                    dict((k, v) for k, v in r.items() if k != "metrics") | r["metrics"]
                    for r in cell_summaries(panel, family)
                )
        csv_write(tab / "recalculated-cell-metrics.csv", cs)
        configuration_tables(tab, cs)
    shutil.copytree(ROOT / "paper", out / "paper", dirs_exist_ok=True)
    render_review(out, values)
    write_claim_pages(out, values)
    write_json(
        out / "reproduction.json",
        dict(
            provider_calls=0,
            comparison_rows=len(values),
            statistics="Replayed from frozen packets; infer/recompute rerun the documented statistical families",
            cell_metrics="Recalculated from scored observations" if recalculate else "Skipped",
            paper="Exact archived source and PDF; Overleaf unchanged",
        ),
    )
    return out


def configuration_tables(tab, cs):
    from collections import defaultdict

    from .metrics import strict_mean

    groups = defaultdict(list)
    for c in cs:
        if c["panel"] in ("ood", "id") and c["target"] != "SFT":
            groups[
                tuple(c[k] for k in ("panel", "family", "source", "mode", "target", "arm"))
            ].append(c)
    configs = []
    for k, records in sorted(groups.items()):
        seedrows = defaultdict(list)
        for r in records:
            seedrows[r["seed"]].append(r)
        if any(len(rs) != 3 for rs in seedrows.values()):
            raise ValueError("Incomplete descriptive domain group")
        row = dict(zip(("panel", "family", "source", "mode", "target", "arm"), k))
        row["seeds"] = sorted(seedrows)
        for m in PRINTED_METRICS:
            row[m] = strict_mean(strict_mean(r[m] for r in rs) for rs in seedrows.values())
        configs.append(row)
    csv_write(tab / "configuration-means.csv", configs)
    scopes = defaultdict(list)
    for r in configs:
        scopes[tuple(r[k] for k in ("panel", "family", "source", "mode"))].append(r)
    leaders = []
    for k, rs in sorted(scopes.items()):
        for m in ("rule_F1", "IoU_ECE10"):
            known = [r for r in rs if r[m] is not None]
            if not known:
                continue
            best = (min if m == "IoU_ECE10" else max)(known, key=lambda r: r[m])
            leaders.append(
                dict(
                    zip(("panel", "family", "source", "mode"), k),
                    metric=m,
                    target=best["target"],
                    arm=best["arm"],
                    value=best[m],
                    seeds=best["seeds"],
                    inference="Descriptive ranking; available seed sets differ",
                )
            )
    csv_write(tab / "configuration-leaders.csv", leaders)
