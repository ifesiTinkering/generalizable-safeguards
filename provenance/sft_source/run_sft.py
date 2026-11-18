"""Isolated, once-only matched GPT SFT. No RL launch or existing-run writes."""
import fcntl
import hashlib
import json
import os
import random
import sys
import time
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE / 'source_snapshot'))
from protocol import read, save, sha

def curriculum():
    rows = read(HERE / 'sft_source_snapshot/claim2_format_only55_20261112/training_rows.json')
    ids = [r['row_id'] for r in rows]
    if len(ids) != 55 or len(set(ids)) != 55:
        raise RuntimeError('Exactly55 unique format-training rows required')
    return [('format', ids, 10)]

def run_format_curriculum(train, checkpoint, validate):
    """Pure orchestration: baseline first, structure-only updates, no later semantic stage."""
    pair = checkpoint('fresh-base-format-baseline')
    result = validate(0, pair)
    if result['pass']:
        return pair, result
    stage, ids, epochs = curriculum()[0]
    for epoch in range(1, epochs + 1):
        train(stage, ids, epoch - 1)
        if epoch % 2 == 0:
            pair = checkpoint(f'formatonly-epoch{epoch:02d}')
            result = validate(epoch, pair)
            if result['pass']:
                break
    return pair, result

def batches(ids, epoch):
    ordered = sorted(ids)
    random.Random(epoch).shuffle(ordered)
    return [ordered[i:i+4] for i in range(0, len(ordered), 4)]

def required_bindings():
    paths = {HERE / name for name in ('run_sft.py', 'sft_backend.py', 'cost_plan.py', 'rates.json', 'sft_replication_plan.json')}
    for folder in ('source_snapshot', 'sft_source_snapshot'):
        paths.update((HERE / folder).rglob('*.py'))
    paths.update((HERE / 'sft_source_snapshot').rglob('*.json'))
    paths.update(p for p in (HERE / 'tokenizer').iterdir() if p.is_file())
    paths.update(HERE / 'prepared_sft' / name for name in
                 ('format_datums.json', 'validation_prompts.json'))
    return {str(p.relative_to(HERE)) for p in paths}

def validate_format_datums(datums):
    expected = set(curriculum()[0][1])
    if set(datums) != expected:
        raise RuntimeError('Format datum membership mismatch')
    required = {'answer_id','evidence_rule_id','turn_id','quote','justification','analysis','confidence'}
    observed = set()
    for datum in datums.values():
        audit = datum['audit'];spans = audit['masked_spans'];observed.update(x['field'] for x in spans)
        if audit.get('answer_ids_supervised') is not False or audit.get('evidence_rule_ids_supervised') is not False or audit.get('synthetic_payload_supervised') is not False:
            raise RuntimeError('Semantic supervision prohibited')
        rw = audit['response_weights'];offsets = audit['offsets'];start = audit['response_start']
        if len(rw) != len(offsets) or not rw or rw[-1] != 1:
            raise RuntimeError('Invalid structural/EOS alignment')
        expected_weights = ([0.] * start + rw)[1:]
        if datum['weights'] != expected_weights or len(datum['weights']) != len(datum['target_tokens']):
            raise RuntimeError('Prompt/response loss alignment mismatch')
        for weight,(left,right) in zip(rw,offsets):
            if weight not in (0.,1.):raise RuntimeError('Invalid loss weight')
            if any(left < span['end'] and right > span['start'] for span in spans) and weight != 0:
                raise RuntimeError('Semantic payload token has nonzero loss')
    if not required <= observed:
        raise RuntimeError('Missing required semantic mask fields')
    return True

def preflight(m):
    expected = required_bindings()
    if not expected or set(m.get('source_hashes', {})) != expected:
        raise RuntimeError('Exhaustive source/input bindings required')
    for name, digest in m['source_hashes'].items():
        if sha(HERE / name) != digest:
            raise RuntimeError('Source mismatch: ' + name)
    rows = read(HERE / 'sft_source_snapshot/claim2_format_only55_20261112/validation90_rows.json')
    prompts = read(HERE / 'prepared_sft/validation_prompts.json')
    keys = [r['row_id'] for r in rows]
    if len(keys) != 90 or len(set(keys)) != 90 or set(prompts) != set(keys):
        raise RuntimeError('Exactly 90 unique frozen validation rows/prompts required')
    validate_format_datums(read(HERE / 'prepared_sft/format_datums.json'))
    return rows, prompts

def execute(manifest_path):
    manifest_path = Path(manifest_path)
    m = read(manifest_path)
    if not m.get('authorized') or m['model'] != 'openai/gpt-oss-120b:peft:131072':
        raise RuntimeError('Exact GPT SFT authorization required')
    if m.get('admission_status') != 'ready':
        raise RuntimeError('Resource and final artifact admission required before launch')
    rank = m['lora_rank']
    if rank not in (32, 64):
        raise RuntimeError('Unapproved adapter rank')
    if rank == 32 and m.get('rank32_exception_authorized') is not True:
        raise RuntimeError('Rank32 exception must be documented')
    rows, prompts = preflight(m)
    out = Path(m['output_root'])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (out / 'launch.json').open('x') as f:
        json.dump({'pid': os.getpid(), 'unix': time.time(), 'manifest_sha256': sha(manifest_path)}, f)
        f.flush()
        os.fsync(f.fileno())
    directory_fd = os.open(out, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    from evidence_budget import Ledger
    from cohort_budget import CohortMeter
    from sft_backend import SFTBackend
    from live_backend import disable_retries
    from reward_pipeline import score_all
    from transformers import AutoTokenizer
    from tinker_cookbook import renderers
    import tinker
    from tinker.lib.retry_handler import RetryConfig
    ledger = Ledger(out / 'budget.sqlite', cap_usd=m['cash_cap_usd'])
    meter = CohortMeter(ledger, 'gpt_ioud:sft', Decimal(str(m['cash_cap_usd'])), out / 'cohort.json')
    storage = Decimal(str(m['storage_allowance_usd']))
    if storage <= 0:
        raise RuntimeError('Positive funded storage allowance required')
    storage_id = 'gpt_ioud:sft:storage'
    if not meter.reserve(storage_id, storage, {'kind': 'modeled_storage_allowance'}):
        raise RuntimeError('Storage allowance already owned')
    meter.dispatch(storage_id)
    meter.settle(storage_id, storage)
    b = SFTBackend.__new__(SFTBackend)
    b.config = dict(model=m['model'], initial_checkpoint='fresh_base', learning_rate=1e-5,
                    adam={'beta1': .9, 'beta2': .95, 'eps': 1e-8},
                    storage_reserve_confirmed=True, storage_max_pairs_per_job=7,
                    permanent_retention_authorized=True)
    b.outdir = out
    b.journal = out / 'transport_journal'
    b.journal.mkdir()
    b.meter = meter
    b.sdk = tinker
    b.retry_class = RetryConfig
    b.service = tinker.ServiceClient()
    disable_retries(b.service)
    b.client = b.sampler = b.sampling_path = None
    r = renderers.get_renderer('gpt_oss_no_sysprompt', AutoTokenizer.from_pretrained(HERE / 'tokenizer', local_files_only=True))
    b.config['stop_sequences'] = r.get_stop_sequences()
    done = 0
    def initialize():
        b.client = b.service.create_lora_training_client(base_model=m['model'], rank=rank, seed=0)
        disable_retries(b.client)
        return {'model': m['model'], 'rank': rank, 'seed': 0}
    def train(stage, ids, epoch):
        nonlocal done
        if stage != 'format':
            raise RuntimeError('Semantic stages are not authorized by this curriculum')
        ds = read(HERE / 'prepared_sft/format_datums.json')
        for batch in batches(ids, epoch):
            save(out / 'state.json', {'updates': done, 'pending': done+1, 'stage': stage, 'epoch': epoch})
            value = b.supervised_update([ds[k] for k in batch], f'gpt_ioud:sft:update:{done+1}')
            done += 1
            save(out / 'updates' / f'{done:03d}.json', value)
            save(out / 'state.json', {'updates': done, 'pending': None, 'stage': stage, 'epoch': epoch})
    def validate(epoch, pair):
        results = []
        for row in rows:
            key = row['row_id']; prompt = prompts[key]
            seq = b.sample(prompt['tokens'], prompt['seed'], 23000, f'gpt_ioud:sft:validation:{epoch}:{key}')
            raw = r.tokenizer.decode(seq['tokens'], skip_special_tokens=False)
            _, end = r.parse_response(seq['tokens'])
            clean = bool(end.is_clean) and '<|channel|>' not in raw and '<|call|>' not in raw
            text = '<answer>\n' + raw.removesuffix('<|return|>')
            scores = score_all(row, text, 'partial', None, clean)
            valid = bool(scores['valid'])
            item = {'row_id': key, 'valid': valid, 'text': text, 'tokens': seq['tokens'], 'clean': clean,
                    'f1': scores.get('f1', 0.), 'iou': scores.get('iou', 0.), 'exact': scores['exact'],
                    'semantic_evidence_evaluated': False}
            save(out / 'validation' / str(epoch) / (hashlib.sha256(key.encode()).hexdigest()+'.json'), item)
            results.append(item)
        n = sum(x['valid'] for x in results)
        receipt = {'epoch': epoch, 'n': 90, 'valid': n, 'pass': n >= 86, 'checkpoint': pair,
                   'selection_metric': 'full_contract_validity', 'no_F1_selection': True,
                   'student_diagnostics_all90_invalid_zero': {k: sum(x[k] for x in results)/90 for k in ('f1','iou','exact')},
                   'semantic_evidence_evaluated': False}
        save(out / 'validation' / str(epoch) / 'complete.json', receipt)
        return receipt
    try:
        b._execute('gpt_ioud:sft:initialize', {'kind': 'initialize_fresh_base', 'model': m['model'], 'rank': rank, 'seed': 0}, Decimal(0), initialize, allow_cached=False)
        pair, result = run_format_curriculum(train, b.checkpoint, validate)
        assert done <= 140
        if result['pass']:
            pair = b.checkpoint('selected-format-sft', final=True)
        save(out / 'completed.json', {'updates': done, 'selected': pair, 'format_gate_passed': result['pass'],
                                      'ready_for_RL': False, 'requires_format_stage_review': True,
                                      'semantic_stages_launched': False, 'no_RL_launched': True})
        meter.close()
    except BaseException as exc:
        save(out / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'never_replay_unknown': True})
        raise
    finally:
        b.close(status='success' if (out / 'completed.json').exists() else 'errored')

if __name__ == '__main__':
    execute(sys.argv[1])
