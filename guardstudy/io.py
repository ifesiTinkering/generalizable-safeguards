"""Deterministic, relative-path artifact I/O."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_json(path: str | Path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            value,
            sort_keys=True,
            allow_nan=False,
            indent=None if path.suffix == ".gz" else 2,
        )
        + "\n"
    ).encode()
    payload = gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_id(family: str, target: str, arm, seed) -> str:
    if target.upper() == "SFT":
        return family.lower() + "-sft"
    target = {"partial": "f1", "iou": "jaccard"}.get(target, target)
    return f"{family.lower()}-{target}-{arm.lower()}-s{seed}"
