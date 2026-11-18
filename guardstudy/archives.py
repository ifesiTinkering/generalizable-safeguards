"""Deterministic JSONL/gzip shards suitable for release assets."""

import gzip
import json
from pathlib import Path

from .io import sha256, write_json


def shard(source, destination, max_mib=40):
    if max_mib < 1:
        raise ValueError("Positive shard size required")
    source = Path(source)
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Archive destination must be empty")
    limit = int(max_mib * 1024**2 * 0.95)
    parts = []
    stream = None
    path = None
    n = size = total = 0

    def close():
        if stream is not None:
            stream.close()
            parts.append(
                dict(
                    path=path.name,
                    records=n,
                    bytes=path.stat().st_size,
                    sha256=sha256(path),
                )
            )

    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as f:
        for line in f:
            json.loads(line)  # A corrupt source cannot become a successful archive.
            if len(line) > limit:
                raise ValueError("A single record exceeds the shard limit")
            if stream is None or size + len(line) > limit:
                close()
                path = out / f"answers-{len(parts) + 1:05d}.jsonl.gz"
                stream = gzip.GzipFile(filename=str(path), mode="wb", mtime=0, compresslevel=6)
                n = size = 0
            stream.write(line)
            n += 1
            size += len(line)
            total += 1
    close()
    manifest = dict(
        schema="answer-shards-v1",
        records=total,
        source_sha256=sha256(source),
        parts=parts,
    )
    write_json(out / "manifest.json", manifest)
    return manifest
