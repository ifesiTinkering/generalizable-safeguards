"""Portable external-input contract. Validation never constructs provider clients."""

import hashlib
import json
import math
from pathlib import Path

MODEL = {"qwen": "Qwen/Qwen3.5-9B", "gpt": "openai/gpt-oss-120b:peft:131072"}
FIXED = {
    "max_updates": 155,
    "batch_size": 3,
    "samples_per_prompt": 10,
    "rows_per_domain": 1,
    "learning_rate": 1e-5,
    "adam": {"beta1": 0.9, "beta2": 0.95, "eps": 1e-8},
    "max_tokens": 23000,
    "context_limit": 65536,
    "validation_every_updates": 10,
    "min_delta": 0.03,
    "row_queue_schema": "claim2-safepyramid-policytraj-append-v1",
    "max_domain_replacements": 140,
    "algorithm": "official_Tinker_centered_raw_token_sum_importance_sampling_beta0",
    "reference_kl_coefficient": 0.0,
    "reward_std_normalization": False,
    "response_length_normalization": False,
    "stopping_rule_id": "selection_three_majority_rolling_five_v1",
    "temperature": 1.0,
}
ADAPTER_RANK = {"qwen": 64, "gpt": 32}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if cfg.get("model_family") not in MODEL:
        raise ValueError("model_family must be qwen or gpt")
    if cfg.get("target") not in ("binary", "partial", "iou"):
        raise ValueError("target must be binary, partial (set F1), or iou")
    if cfg.get("arm") not in ("A", "B", "C", "D"):
        raise ValueError("arm must be A, B, C, or D")
    if type(cfg.get("seed")) is not int or cfg["seed"] < 0:
        raise ValueError("Nonnegative integer seed required")
    for name, value in FIXED.items():
        if name in cfg and cfg[name] != value:
            raise ValueError("Fixed scientific setting changed: " + name)
    rank = ADAPTER_RANK[cfg["model_family"]]
    if "lora_rank" in cfg and cfg["lora_rank"] != rank:
        raise ValueError("Adapter rank must match the final model-specific recipe")
    cfg.update(FIXED)
    cfg["model"] = MODEL[cfg["model_family"]]
    cfg["lora_rank"] = rank
    cfg["config_path"] = str(path)
    cfg["config_sha256"] = sha(path)
    return cfg


def load_external_rows(cfg):
    datasets = {}
    bindings = {}
    all_ids = set()
    for kind, count in [("train", 420), ("validation", 90), ("supplement", 1210)]:
        binding = cfg["datasets"][kind]
        p = Path(binding["path"]).expanduser().resolve()
        if sha(p) != binding["sha256"]:
            raise ValueError("Dataset SHA256 mismatch: " + kind)
        rows = json.loads(p.read_text())
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("Exact dataset size required: " + kind)
        for row in rows:
            rid = row["row_id"]
            if not isinstance(rid, str) or not rid or rid in all_ids:
                raise ValueError("Dataset row IDs must be globally unique")
            all_ids.add(rid)
            if not all(
                isinstance(row[k], str) for k in ("conversation", "policy", "policy_framework")
            ):
                raise ValueError(
                    "Supply canonical, pre-normalized source strings; no silent rendering edits"
                )
            ids = [str(r["rule_id"]) for r in row["all_policy_rules"]]
            gold = row["ground_truth_rule_ids"]
            if (
                len(ids) != len(set(ids))
                or not isinstance(gold, list)
                or any(type(x) is not str for x in gold)
                or len(gold) != len(set(gold))
                or not set(gold) <= set(ids)
            ):
                raise ValueError("Invalid rule inventory/gold labels")
            if (kind == "supplement") != (row.get("supplement_source") == "policytraj"):
                raise ValueError("Exact original versus PolicyTraj source marker required")
        datasets[kind] = rows
        bindings[kind] = {"path": str(p), "sha256": binding["sha256"], "rows": count}
    return datasets, bindings


def verify_sources():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "provenance/source_manifest.json").read_text())
    for item in manifest["files"]:
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError("Changed frozen scientific source: " + item["path"])
    return len(manifest["files"])


def live_inputs(cfg):
    if (
        not isinstance(cfg.get("initial_checkpoint"), str)
        or not cfg["initial_checkpoint"].startswith("tinker://")
        or "/weights/" not in cfg["initial_checkpoint"]
        or "<" in cfg["initial_checkpoint"]
    ):
        raise ValueError("An accessible, selected SFT full-checkpoint URI is required")
    if not Path(cfg["tokenizer_path"]).expanduser().is_dir():
        raise ValueError("Local model tokenizer directory required")
    for key in ("cash_limit_usd", "storage_reserve_usd"):
        value = cfg.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Positive explicit " + key + " required")
    if cfg["storage_reserve_usd"] >= cfg["cash_limit_usd"]:
        raise ValueError("Storage reserve must leave inference/training capacity")
    if cfg.get("provider_prices_verified") is not True:
        raise ValueError("Verify provider prices before enabling paid execution")
    if cfg.get("permanent_checkpoint_storage_authorized") is not True:
        raise ValueError(
            "Explicit storage authorization required; no automatic checkpoint deletion"
        )
    for key, default, ceiling in [
        ("sample_workers", 8, 16),
        ("judge_workers", 4, 16),
        ("validation_sample_workers", 4, 8),
        ("validation_judge_workers", 4, 8),
    ]:
        value = cfg.setdefault(key, default)
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError("Invalid bounded " + key)
    return cfg
