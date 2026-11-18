"""Audit the exact review-package file set without printing possible secrets."""

import gzip
import re

from package_review import files

from guardstudy.io import ROOT, write_json

PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "api_token": re.compile(rb"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{32,}"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "huggingface_token": re.compile(rb"\bhf_[A-Za-z0-9]{30,}"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "slack_token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{24,}"),
    "literal_credential": re.compile(
        rb"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[=:]\s*['\"][A-Za-z0-9_+/=-]{24,}['\"]"
    ),
    "provider_binding": re.compile(
        rb"tinker://(?!(?:synthetic|test-only|testmodel|test)(?:[/:]|$))[A-Za-z0-9][^\s\"'<>]*"
    ),
}


def main():
    selected = files()
    issues = []
    for path in selected:
        rel = str(path.relative_to(ROOT))
        if path.is_symlink() or any(
            part in {"private", "artifacts", ".venv", ".env"}
            for part in path.relative_to(ROOT).parts
        ):
            issues.append(dict(path=rel, issue="excluded path or symlink"))
        if path.stat().st_size >= 100 * 1024**2:
            issues.append(dict(path=rel, issue="file exceeds ordinary GitHub size limit"))
        if path.suffix == ".pdf":
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            tail = b""
            while chunk := stream.read(1024 * 1024):
                text = tail + chunk
                for name, pattern in PATTERNS.items():
                    if pattern.search(text) and dict(path=rel, issue=name) not in issues:
                        issues.append(dict(path=rel, issue=name))
                tail = text[-512:]
    result = dict(files_checked=len(selected), issues=issues, provider_calls=0)
    write_json(ROOT / "reports/release-audit.json", result)
    print(result)
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
