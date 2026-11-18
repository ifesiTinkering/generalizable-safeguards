"""Look up actual saved outputs in the local compressed response shards."""

import gzip
import json

from .io import ROOT, read_json


def find(checkpoint, panel, mode, row_id=None, limit=2):
    root = ROOT / "artifacts/answers"
    manifest = read_json(root / "manifest.json")
    paths = manifest["index"].get("|".join([panel, checkpoint, mode]), [])
    found = []
    for name in paths:
        if "/" in name or ".." in name:
            raise ValueError("Invalid shard path")
        with gzip.open(root / name, "rt") as f:
            for line in f:
                r = json.loads(line)
                i = r["identity"]
                if (
                    i["checkpoint_id"] == checkpoint
                    and i["panel"] == panel
                    and i["mode"] == mode
                    and (row_id is None or i["row_id"] == row_id)
                ):
                    found.append(r)
                    if len(found) >= limit:
                        return found
    return found
