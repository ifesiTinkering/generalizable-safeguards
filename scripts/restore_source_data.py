"""Restore an approved source-data artifact with path and checksum checks."""

import argparse
import hashlib
import tarfile

from guardstudy.io import ROOT, read_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    manifest = read_json(args.manifest)
    if sha256(args.archive) != manifest["sha256"]:
        raise ValueError("Archive checksum mismatch")
    expected = {r["path"]: r for r in manifest["files"]}
    # Verify every member before restoring any files.
    for restore in (False, True):
        seen = set()
        with tarfile.open(args.archive, "r:gz") as tar:
            for member in tar:
                if member.name not in expected or not member.isfile() or member.name in seen:
                    raise ValueError("Unexpected artifact member")
                dest = (ROOT / member.name).resolve()
                if not dest.is_relative_to((ROOT / "private").resolve()):
                    raise ValueError("Artifact path outside private directory")
                data = tar.extractfile(member).read()
                record = expected[member.name]
                if (
                    len(data) != record["bytes"]
                    or hashlib.sha256(data).hexdigest() != record["sha256"]
                ):
                    raise ValueError("Member checksum mismatch")
                if dest.exists() and sha256(dest) != record["sha256"]:
                    raise ValueError("Existing file differs; no overwrite")
                if restore and not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                seen.add(member.name)
        if seen != expected.keys():
            raise ValueError("Artifact has missing files")
    print("Restored verified source data; provider calls: 0")


if __name__ == "__main__":
    main()
