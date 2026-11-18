"""Compare default-input statistical recalculations with the archived study.

Run after the four README recalculation commands. This is an offline acceptance
check for the original observations, not a comparison for newly sampled data.
"""

import argparse
import math

from guardstudy.io import ROOT, read_json, write_json
from guardstudy.verify import verify_analysis_inputs

SCOPE = ("family", "target", "high_target", "arm", "source", "mode", "regime", "contrast")
TEST_FIELDS = ("difference", "percentile95", "raw_p", "BH_q", "Holm_p", "available")
SUPPORT_FIELDS = (
    "identification_bounds",
    "bootstrap95_envelope",
    "conservative_permutation_p_upper",
    "secondary_BH512_upper",
    "secondary_Holm512_upper",
)


def check(directory):
    """Require every saved scope and compare each reported inferential endpoint."""
    verify_analysis_inputs(ROOT)
    differences = []
    checks = 0

    def same(a, b):
        if a is None or b is None:
            return a is b
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        if isinstance(a, list):
            return (
                isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            )
        return math.isclose(a, b, abs_tol=1e-11, rel_tol=1e-11)

    def compare(label, actual, expected, fields=TEST_FIELDS):
        nonlocal checks

        def key(row):
            return tuple(row.get(k) for k in SCOPE)

        reference = {key(r): r for r in expected}
        if (
            len(reference) != len(expected)
            or len(actual) != len(expected)
            or {key(r) for r in actual} != set(reference)
        ):
            raise ValueError("Different analysis scopes: " + label)
        for row in actual:
            original = reference[key(row)]
            current = row.get("metrics", row.get("tests"))
            saved = original.get("metrics", original.get("tests"))
            if set(current) != set(saved):
                raise ValueError("Different endpoints: " + label)
            for metric in saved:
                for field in fields:
                    checks += 1
                    if not same(current[metric].get(field), saved[metric].get(field)):
                        differences.append(
                            dict(analysis=label, scope=list(key(row)), metric=metric, field=field)
                        )

    directory = directory.resolve()
    stats = ROOT / "data/statistics"
    for name, packet in (
        ("calibration", "ood_calibration.json.gz"),
        ("calibration-f1", "qwen_f1_calibration.json.gz"),
        ("correctness", "correctness_tests.json.gz"),
    ):
        compare(
            name,
            read_json(directory / f"{name}.json.gz")["results"],
            read_json(stats / packet)["results"],
        )
    evidence = read_json(directory / "evidence.json.gz")
    original = read_json(stats / "ood_latest_C.json")
    compare("evidence-recipe", evidence["results"], original["ood_all_recipe_tests"])
    compare(
        "evidence-support",
        evidence["support_sensitivity"],
        original["ood_all_support_sensitivity"],
        SUPPORT_FIELDS,
    )
    return dict(provider_calls=0, checks=checks, differences=differences)


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/recomputed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.directory)
    if args.output:
        write_json(args.output, result)
    import json

    print(json.dumps(result, indent=2))
    if result["differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
