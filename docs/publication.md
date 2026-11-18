# Public repository and separate artifacts

The public repository is https://github.com/ifesiTinkering/generalizable-safeguards.

## Included on GitHub

- Training, evaluation, analysis, and archive-management code.
- The final checkpoint roster with identity hashes and public run identifiers.
- Numerical scored observations and statistical test packets.
- Readable pages and complete CSV tables for claims 1, 2, and 3.
- The archived paper, its source, and the training-run reference.

The README links directly to the claim pages. Offline reproduction uses the committed numerical observations and needs no provider account.

## Kept local

- API keys, environment files, SSH keys, and private checkpoint URI bindings.
- Canonical source-text bundles and matching tokenizer snapshots.
- Raw model-answer, reasoning-trace, provider-response, and RunPod archives.
- Model weights, live experiment outputs, local environments, and caches.

Those files remain in ignored local directories. Public checkpoint hashes identify the evaluated weights; they do not grant access to them. Source-data and response archives can be distributed separately after their release terms and storage locations have been settled.

## Publication checks

The initial publication uses the explicit file list produced by `scripts/package_review.py`. It starts with new Git history, so older workspace commits cannot carry private files into the repository. `scripts/audit_release.py` scans that file set, including compressed JSON and CSV/HTML exports. The initial publication also checks extracted PDF text and the staged Git tree.

For later changes, run:

```bash
python scripts/audit_release.py
safeguards verify --full
python -m pytest -q
```

Inspect `git diff --cached` and the staged file names before pushing. `.gitignore` excludes the private directories and common credential files. The GitHub workflow runs the offline tests and reproduction checks without model credentials.

## License and attribution

The authors have not selected a source-code redistribution license. The repository is public for inspection and reproduction review. `LICENSE_PENDING.md` records that status. Third-party dependencies, datasets, model weights, and model outputs retain their own terms. `NOTICE.md` records the training-adapter provenance.

The archived Overleaf paper remains unchanged by this publication.
