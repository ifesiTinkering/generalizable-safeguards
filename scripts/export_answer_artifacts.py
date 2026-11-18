"""Prepare local release candidates from the checksum-verified original export.

No model output text is summarized. Provider pointers become identity hashes;
original unmodified objects remain in the private export. Nothing is published.
"""

import collections
import gzip
import hashlib
import json

from guardstudy.io import ROOT, read_json, sha256, write_json


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, str) and x.startswith("tinker://"):
        return "checkpoint-sha256:" + hashlib.sha256(x.encode()).hexdigest()
    return x


def main():
    out = ROOT / "artifacts/answers"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use an empty artifact destination")
    parts = []
    indices = collections.defaultdict(set)
    seen = set()
    counts = collections.Counter()
    status = collections.Counter()
    stream = None
    path = None
    n = size = 0
    limit = 38 * 1024**2

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

    sources = [
        ROOT / "private/model-answers.jsonl.gz",
        ROOT / "private/model-answers-legacy.jsonl.gz",
    ]
    for source in sources:
        with gzip.open(source, "rt") as f:
            for line in f:
                r = json.loads(line)
                ref = r["reference"]
                identity = {k: ref[k] for k in ["panel", "checkpoint_id", "mode", "row_id", "draw"]}
                key = tuple(identity.values())
                if key in seen:
                    raise ValueError("Duplicate answer identity")
                seen.add(key)
                counts[ref["panel"] + "-" + ref["family"]] += 1
                status[r.get("raw_status", "unknown")] += 1
                if r.get("raw_status") != "verified":
                    raise ValueError("Raw answer unavailable or changed")
                obj = dict(
                    identity=identity,
                    raw=clean(r["raw"]),
                    scored=clean(r.get("scored")),
                    source_raw_sha256=r["raw_sha256"],
                    source_scored_sha256=r.get("scored_sha256"),
                )
                if r.get("final") is not None:
                    obj["original_final_record"] = clean(r["final"])
                wire = (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
                if stream is None or size + len(wire) > limit:
                    close()
                    path = out / f"answers-{len(parts) + 1:05d}.jsonl.gz"
                    stream = gzip.GzipFile(filename=str(path), mode="wb", mtime=0, compresslevel=3)
                    n = size = 0
                stream.write(wire)
                n += 1
                size += len(wire)
                indices[tuple(identity[k] for k in ["panel", "checkpoint_id", "mode"])].add(
                    path.name
                )
    close()
    expected = set()
    for p in (ROOT / "data/observations").glob("*.json.gz"):
        if p.name.startswith("REFERENCE"):
            continue
        panel = p.name.split("-")[0]
        for c in read_json(p)["cells"]:
            expected.update(
                (panel, c["checkpoint_id"], c["identity"]["mode"], rid, draw)
                for rid, draw in zip(c["row_ids"], c["draws"])
            )
    if seen != expected:
        raise ValueError("Answer coverage differs from scored observation membership")
    manifest = dict(
        schema="saved-answer-release-candidate-v1",
        records=len(seen),
        counts=dict(counts),
        parts=parts,
        index={"|".join(k): sorted(v) for k, v in indices.items()},
        provider_calls=0,
        publication_status="Local candidate only; redistribution terms and storage destination remain to be selected",
    )
    write_json(out / "manifest.json", manifest)
    audit = dict(
        records=len(seen),
        counts=dict(counts),
        raw_status=dict(status),
        missing=0,
        unexpected=0,
        source_archives=[
            dict(path=str(p.relative_to(ROOT)), sha256=sha256(p), bytes=p.stat().st_size)
            for p in sources
        ],
        release_candidate_manifest_sha256=sha256(out / "manifest.json"),
        parts=len(parts),
        provider_calls=0,
        remote_mutations=0,
    )
    write_json(ROOT / "provenance/answer_archive_inventory.json", audit)
    print(
        json.dumps(
            {
                "records": len(seen),
                "shards": len(parts),
                "missing": 0,
                "bytes": sum(p["bytes"] for p in parts),
            }
        )
    )


if __name__ == "__main__":
    main()
