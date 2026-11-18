# Data, answers, and run archives

## Three artifact layers

| Layer | Location | What it preserves |
|---|---|---|
| Small committed study | `data/`, `paper/`, `provenance/` | Scored observations, question descriptors, checkpoint roster, original tests, and exact paper sources |
| Local reproduction inputs | `private/rows-*.json.gz`, `private/training/` | Canonical conversation, policy, rule inventory, gold labels, and training payloads |
| Original records | `private/model-answers*.jsonl.gz`, `private/*run-records*.tar.gz`, `private/runpod-assets.tar.gz` | Actual answers, decoded traces, original score records, selected checkpoint records, and run states |

The observation inventory is `provenance/observation_inventory.json`. It records each compressed file's checksum and answer count. There are 767,080 scored answer records across the three panels. Some policy-level answers reuse an earlier response, so this is an analysis-record count rather than a count of newly purchased model requests.

## Source payloads

`provenance/dataset_payload_inventory.json` records the prepared row-file hashes. It verifies that rendered scientific inputs agree across the recovered Qwen and GPT prompt caches. The local files preserve the original text; the public row descriptors omit conversation and policy text.

The local training inputs contain:

- 420 original RL rows and 90 validation rows.
- 1,210 appended PolicyTraj rows.
- The original SFT source pool and the selected 55-example SFT continuation subset.

Matching tokenizer files are preserved under `private/tokenizers/`; their hashes are recorded in `provenance/tokenizer_inventory.json`. Provider identifiers are supplied separately in local checkpoint bindings. Access to a provider URI is not implied by knowing its hash. Checkpoint weights live at the provider and are not silently downloaded into this repository.

## Compressed model answers

The read-only RunPod export checks each available recorded raw-file hash. It records the original response object and its original score object together with the question, draw, thinking mode, and public checkpoint identifier. The legacy SFT export also preserves its separate final-output record.

`provenance/answer_archive_inventory.json` records final collection and coverage checks. The original archives are local, ignored by Git. The prepared release candidates are under artifacts/answers/ and are also ignored by Git. Their index supports direct output lookup:

```bash
safeguards answers --checkpoint gpt-jaccard-b-s2 --panel id --mode TWO_STAGE_ON --limit 2
```

This command prints the actual full saved records. It does not summarize the answer. Release-ready compressed shards can be made with:

```bash
safeguards archive --input private/model-answers.jsonl.gz \
  --output private/answer-shards --max-mib 40
```

Each shard is valid gzip-compressed JSONL. Its manifest records its SHA256, byte size, and record count. Sharding validates each JSON record and preserves complete record boundaries. The target directory must be empty, which prevents accidental replacement of an earlier archive.

The available reasoning text is retained in `private/reasoning-traces.jsonl.gz`. The inventory records 413,429 answers with reasoning text and 845 Qwen SFT answers whose source text is unavailable. Their final answers are preserved. Use the same archive command to shard this larger trace file when preparing a public artifact release.

The source-data bundle and answer shards are intended for release assets or external artifact storage after redistribution decisions are complete. Large archives should not be added to normal Git history. Small numerical observations stay in the repository so reading and checking the paper remains convenient.

## Restore artifact bundles

A collaborator can restore a supplied source-data archive into a local `private/` directory, verify its files against the payload inventory, and supply their own checkpoint bindings. `scripts/check_local_inputs.py` verifies those local data files and checks the training JSON schema. It makes no provider calls.

The canonical datasets are cached here for the owner's review. A public release must state their upstream sources and redistribution terms. This package does not claim that a source snapshot alone grants redistribution rights.

Restore a supplied source bundle with:

```bash
python scripts/restore_source_data.py --archive artifacts/source-data.tar.gz \
  --manifest artifacts/source-data-manifest.json
```

The restorer checks the archive and every member before writing files. It accepts only the declared files beneath `private/` and refuses to replace a different existing file. Historical metadata replaces private provider URIs with identity hashes; unmodified copies remain in the local archive.
