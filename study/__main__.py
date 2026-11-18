"""Default commands are offline. Training requires two explicit live flags."""

import argparse
import json

from .config import load_external_rows, read_config, verify_sources


def main():
    p = argparse.ArgumentParser(description="Portable policy-reward study; no providers by default")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-sources")
    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--check-data", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--execute", action="store_true")
    train.add_argument("--accept-paid-operations", action="store_true")
    args = p.parse_args()
    count = verify_sources()
    if args.command == "verify-sources":
        print(json.dumps({"verified_frozen_files": count, "provider_calls": 0}))
        return
    cfg = read_config(args.config)
    if args.command == "validate":
        report = {
            "model": cfg["model"],
            "target": cfg["target"],
            "arm": cfg["arm"],
            "seed": cfg["seed"],
            "lora_rank": cfg["lora_rank"],
            "max_updates": cfg["max_updates"],
            "verified_frozen_files": count,
            "provider_calls": 0,
        }
        if args.check_data:
            rows, _ = load_external_rows(cfg)
            report["dataset_rows"] = {k: len(v) for k, v in rows.items()}
        print(json.dumps(report, indent=2))
        return
    if not (args.execute and args.accept_paid_operations):
        p.error("train requires --execute --accept-paid-operations; no provider was constructed")
    from .runner import run

    run(cfg)


if __name__ == "__main__":
    main()
