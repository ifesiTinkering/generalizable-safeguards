"""Public entrypoint. All commands are offline unless execute is explicit."""

import argparse
import json

from .io import ROOT, read_json, sha256, write_json


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Reproduce and inspect the final safeguard reward study"
    )
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="Check frozen source, paper and scored artifact hashes")
    v.add_argument("--full", action="store_true")
    r = sub.add_parser("reproduce", help="Export claim tables and a readable local report")
    r.add_argument("--output", default=str(ROOT / "results/reproduced"))
    r.add_argument("--skip-cell-metrics", action="store_true")
    r.add_argument(
        "--open", action="store_true", help="Open the generated local HTML in your browser"
    )
    sub.add_parser("runs", help="List the final report checkpoint roster")
    i = sub.add_parser(
        "infer", help="Recompute the frozen OOD calibration tests from scored answers"
    )
    i.add_argument("--output", required=True)
    i.add_argument("--family", choices=["Qwen", "GPT"])
    i.add_argument("--target", choices=["binary", "iou", "partial"])
    i.add_argument("--source", choices=["safepyramid", "dynabench"])
    i.add_argument("--mode", choices=["OFF", "TWO_STAGE_ON"])
    e = sub.add_parser("evaluate", help="Plan or execute matched sampling")
    e.add_argument("--config", required=True)
    e.add_argument("--execute", action="store_true")
    e.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate local inputs and credentials without model calls",
    )
    g = sub.add_parser("grade", help="Score saved model answers")
    g.add_argument("--config", required=True)
    g.add_argument("--execute", action="store_true")
    g.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate local grading inputs without model calls",
    )
    a = sub.add_parser("archive", help="Make checksum-verified response archive shards")
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--max-mib", type=int, default=40)
    a = sub.add_parser(
        "answers", help="Read actual saved model outputs from restored response shards"
    )
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--panel", choices=["ood", "id", "policy"], required=True)
    a.add_argument("--mode", choices=["OFF", "TWO_STAGE_ON"], required=True)
    a.add_argument("--row-id")
    a.add_argument("--limit", type=int, default=2)
    rec = sub.add_parser(
        "recompute",
        help="Recalculate a complete evidence, correctness, or set-F1 calibration family",
    )
    rec.add_argument(
        "--analysis", choices=["evidence", "correctness", "calibration-f1"], required=True
    )
    rec.add_argument("--output", required=True)
    rec.add_argument("--workers", type=int, default=2)
    rec.add_argument("--qwen-cells", help="New Qwen out-of-distribution domain cells")
    rec.add_argument("--gpt-cells", help="New GPT out-of-distribution domain cells")
    rec.add_argument("--qwen-id-cells", help="New Qwen in-distribution domain cells")
    rec.add_argument("--gpt-id-cells", help="New GPT in-distribution domain cells")
    exp = sub.add_parser(
        "export-run", help="Convert a completed live run to scored cells and metric CSV"
    )
    exp.add_argument("--config", required=True)
    exp.add_argument("--output", required=True)
    i.add_argument("--qwen-cells")
    i.add_argument("--gpt-cells")
    args = p.parse_args(argv)
    if args.command == "recompute":
        if args.workers < 1:
            p.error("Positive worker count required")
        import os

        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(name, "1")
        # A fresh full-family analysis must not silently mix new and archived data.
        from .data import set_override

        bindings = {
            "qwen_cells": ("ood", "Qwen"),
            "gpt_cells": ("ood", "GPT"),
            "qwen_id_cells": ("id", "Qwen"),
            "gpt_id_cells": ("id", "GPT"),
        }
        supplied = {name for name in bindings if getattr(args, name)}
        required = {
            "evidence": {"qwen_cells", "gpt_cells"},
            "correctness": set(bindings),
            "calibration-f1": {"qwen_cells"},
        }[args.analysis]
        if supplied and supplied != required:
            p.error(
                "For new data, supply exactly: "
                + ", ".join("--" + k.replace("_", "-") for k in sorted(required))
            )
        for name in supplied:
            set_override(*bindings[name], getattr(args, name))
        from .recompute import calibration_f1_all, correctness_all, evidence_all

        result = {
            "evidence": evidence_all,
            "correctness": correctness_all,
            "calibration-f1": calibration_f1_all,
        }[args.analysis](args.output, args.workers)
        result["input_cell_bindings"] = {
            name: {"path": getattr(args, name), "sha256": sha256(getattr(args, name))}
            for name in sorted(supplied)
        }
        write_json(args.output, result)
        from .recompute import write_test_csv

        write_test_csv(result, args.output)
        print(args.output)
        return
    if args.command == "answers":
        if args.limit < 1:
            p.error("Positive answer limit required")
        from .answers import find

        print(
            json.dumps(
                find(args.checkpoint, args.panel, args.mode, args.row_id, args.limit),
                indent=2,
            )
        )
        return
    if args.command == "verify":
        from .verify import verify

        print(json.dumps(verify(args.full), indent=2))
        return
    if args.command == "reproduce":
        from .reproduce import reproduce

        page = reproduce(args.output, not args.skip_cell_metrics) / "review.html"
        print(page)
        if args.open:
            import webbrowser

            webbrowser.open(page.as_uri())
        return
    if args.command == "runs":
        print(json.dumps(read_json(ROOT / "data/runs/checkpoints.json"), indent=2))
        return
    if args.command == "infer":
        from .data import set_override
        from .inference import calibration_all, calibration_scope

        for family, path in [("Qwen", args.qwen_cells), ("GPT", args.gpt_cells)]:
            if path:
                set_override("ood", family, path)
        scope = [args.family, args.target, args.source, args.mode]
        if any(scope) and not all(scope):
            p.error("Supply all four scope selectors, or none for the full fixed family")
        result = calibration_scope(*scope) if all(scope) else calibration_all()
        if all(scope):
            result["inference_note"] = (
                "Single scope: raw p-values only. Full-family run computes the original adjustment."
            )
        result["input_cell_bindings"] = {
            family: {
                "path": path,
                "sha256": sha256(path),
            }
            for family, path in [("Qwen", args.qwen_cells), ("GPT", args.gpt_cells)]
            if path
        }
        write_json(args.output, result)
        from .recompute import write_test_csv

        write_test_csv(result, args.output)
        print(args.output)
        return
    if args.command in ("evaluate", "grade"):
        from .evaluation import execute_grade, execute_sample, plan, preflight

        cfg = read_json(args.config)
        if args.check_inputs and args.execute:
            p.error("Choose --check-inputs or --execute")
        if args.check_inputs:
            try:
                print(json.dumps(preflight(cfg, args.command), indent=2))
            except (ValueError, KeyError, FileNotFoundError) as exc:
                p.error(str(exc))
            return
        if not args.execute:
            print(json.dumps(plan(cfg, args.command), indent=2))
            return
        try:
            preflight(cfg, args.command)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            p.error(str(exc))
        import asyncio

        asyncio.run(execute_sample(cfg) if args.command == "evaluate" else execute_grade(cfg))
        return
    if args.command == "archive":
        from .archives import shard

        print(json.dumps(shard(args.input, args.output, args.max_mib), indent=2))
