"""Build a local source-data release candidate without checkpoint credentials."""

import gzip
import io
import tarfile

from guardstudy.io import ROOT, sha256, write_json


def main():
    inputs = [
        *sorted((ROOT / "private").glob("rows-*.json.gz")),
        *sorted((ROOT / "private/training").glob("*.json")),
        *sorted((ROOT / "private/training").glob("*.json.gz")),
    ]
    out = ROOT / "artifacts/source-data.tar.gz"
    out.parent.mkdir(exist_ok=True)
    manifest = []
    with (
        out.open("wb") as stream,
        gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, compresslevel=3) as gz,
        tarfile.open(fileobj=gz, mode="w|") as archive,
    ):
        for p in inputs:
            b = p.read_bytes()
            name = str(p.relative_to(ROOT))
            info = tarfile.TarInfo(name)
            info.size = len(b)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(b))
            manifest.append(dict(path=name, bytes=len(b), sha256=sha256(p)))
    write_json(
        ROOT / "artifacts/source-data-manifest.json",
        dict(
            archive=out.name,
            sha256=sha256(out),
            files=manifest,
            provider_calls=0,
            publication_status="Local release candidate; not published",
        ),
    )
    print(out)


if __name__ == "__main__":
    main()
