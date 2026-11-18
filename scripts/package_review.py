"""Build an offline review ZIP without private data, environments, or live jobs."""

import zipfile

from guardstudy.io import ROOT, sha256, write_json

DIRECTORIES = (
    "guardstudy",
    "study",
    "configs",
    "data",
    "paper",
    "provenance",
    "reports",
    "docs",
    "scripts",
    "tests",
    ".github",
    "results/reproduced",
)
ROOT_FILES = (
    "README.md",
    "design.md",
    "NOTICE.md",
    "LICENSE_PENDING.md",
    "pyproject.toml",
    "requirements-live.txt",
    ".gitignore",
    ".env.example",
    "Makefile",
)
EXCLUDED = {"__pycache__", ".pytest_cache", ".DS_Store"}


def files():
    result = [ROOT / n for n in ROOT_FILES]
    for name in DIRECTORIES:
        for p in (ROOT / name).rglob("*"):
            if (
                p.is_file()
                and not any(x in EXCLUDED for x in p.parts)
                and p.suffix not in (".pyc", ".aux", ".log", ".out")
                and p.name != "release_manifest.json"
            ):
                result.append(p)
    return sorted(set(result))


def main():
    selected = files()
    manifest = [
        dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha256(p))
        for p in selected
    ]
    write_json(
        ROOT / "provenance/release_manifest.json",
        dict(
            schema="offline-review-release-v1",
            files=manifest,
            excludes=["private/", "artifacts/", ".venv/", "results/live/"],
            provider_calls=0,
        ),
    )
    out = ROOT.parent / "generalizable-safeguards-review.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in selected + [ROOT / "provenance/release_manifest.json"]:
            info = zipfile.ZipInfo(
                "generalizable-safeguards/" + str(p.relative_to(ROOT)),
                date_time=(2026, 11, 17, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())
    print(str(out))
    print(
        "Files:",
        len(selected) + 1,
        "Bytes:",
        out.stat().st_size,
        "SHA256:",
        sha256(out),
    )


if __name__ == "__main__":
    main()
