"""Check restored local source artifacts without any provider operations."""

from guardstudy.io import ROOT, read_json, sha256
from study.config import load_external_rows, read_config


def main():
    inv = read_json(ROOT / "provenance/dataset_payload_inventory.json")
    for group in ("panels", "training"):
        for name, record in inv[group].items():
            if sha256(ROOT / record["path"]) != record["sha256"]:
                raise ValueError("Changed input " + name)
    for family in ("qwen", "gpt"):
        cfg = read_config(ROOT / f"private/{family}-training.local.json")
        rows, _ = load_external_rows(cfg)
        assert {k: len(v) for k, v in rows.items()} == {
            "train": 420,
            "validation": 90,
            "supplement": 1210,
        }
    print("Verified local panels and training inputs; provider calls: 0")


if __name__ == "__main__":
    main()
