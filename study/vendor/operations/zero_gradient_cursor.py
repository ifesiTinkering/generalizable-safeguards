"""Reuse unchanged full optimizer/sampler pairs; prove every durable cursor advance.

Only an exactly hash-bound native Engine is patched. Positive updates, validation,
retention, milestones and finalization retain their native call ordering.
"""
import copy
import hashlib
import inspect
import json
import os
import textwrap
from pathlib import Path

EXPECTED_ENGINE = 'ca987fc4e81db7dc1a5896db04a2dfca307158fc88aef3fbaf9766ad7c260926'
KEY = 'zero_gradient_cursor_v1'
SCHEMA = 'claim2-own-full-zero-gradient-cursor-v1'
RECEIPT = 'zero_gradient_cursor_commit.json'
MARKER = 'zero_gradient_generation.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cursor(state):
    if 'queue_cursor' in state:
        return copy.deepcopy(state['queue_cursor'])
    return {k: state[k] for k in ('pool_pass', 'next_batch')}


def bind_current(state, *, marker_sha256=None):
    """Call only immediately after a confirmed full+sampler save, before state save."""
    marker_sha256 = marker_sha256 or state.get(KEY, {}).get('generation_marker_sha256')
    if not isinstance(marker_sha256, str) or len(marker_sha256) != 64:
        raise RuntimeError('Missing immutable generation marker binding')
    state[KEY] = {
        'generation_marker_sha256': marker_sha256,
        'schema': SCHEMA, 'checkpoint': state['checkpoint'],
        'sampling_path': state['sampling_path'], 'updates': state['updates'],
        'saved_attempt': state['attempt'], 'saved_cursor': cursor(state),
        'tail_sha256': None, 'zero_attempts': 0,
    }


def check_binding(state):
    b = state.get(KEY)
    if not isinstance(b, dict) or b.get('schema') != SCHEMA:
        raise RuntimeError('Missing full optimizer/cursor binding')
    for key in ('checkpoint', 'sampling_path', 'updates'):
        if b.get(key) != state.get(key):
            raise RuntimeError('Full optimizer/cursor binding mismatch: ' + key)
    if not isinstance(b.get('saved_attempt'), int) or not isinstance(b.get('zero_attempts'), int):
        raise RuntimeError('Invalid full optimizer/cursor counters')
    if b['zero_attempts'] < 0 or b['saved_attempt'] + b['zero_attempts'] != state['attempt']:
        raise RuntimeError('Full optimizer/cursor attempt discontinuity')
    if (b['zero_attempts'] == 0) != (b.get('tail_sha256') is None):
        raise RuntimeError('Missing/unexpected zero-gradient proof tail')
    if b['zero_attempts'] == 0 and b['saved_cursor'] != cursor(state):
        raise RuntimeError('Saved full optimizer cursor changed without proof')
    return b



def resolve_source_generation(bound_files, source_path, *, job_id, target):
    """Resolve the producer through immutable same-seed admission DAG links.

    V40's initialization fallback directly binds the failed admission, whose
    own bound_files links the earlier producer. Never infer legacy from that
    missing direct link. Every traversed admission and its input-state binding
    is checked; paths remain opaque, with bounded traversal and cycle checks.
    """
    if bound_files is None:
        return {'producers': [], 'verified_admission_hashes': {}}
    visited, active, data_cache, producers = {}, set(), {}, []
    source_path = Path(source_path).resolve()
    def data(path):
        path = str(Path(path))
        if path not in data_cache:
            raw = Path(path).read_bytes()
            data_cache[path] = (hashlib.sha256(raw).hexdigest(), raw)
        return data_cache[path]
    def walk(path, expected, depth):
        path = str(Path(path))
        if depth > 64 or len(visited) + len(active) >= 512:
            raise RuntimeError('Source admission provenance traversal bound exceeded')
        digest, raw = data(path)
        if digest != expected:
            raise RuntimeError('Changed transitive source admission hash')
        value = json.loads(raw)
        if not isinstance(value, dict) or (value.get('job_id'), value.get('target')) != (job_id, target):
            return  # Unrelated roster admissions are never provenance for this seed.
        if path in active:
            raise RuntimeError('Cyclic source admission provenance')
        if path in visited:
            if visited[path] != expected:
                raise RuntimeError('Conflicting source admission binding')
            return
        active.add(path)
        bindings = value.get('bound_files', {})
        prior = value.get('source_state')
        if prior and bindings.get(prior) != data(prior)[0]:
            raise RuntimeError('Changed transitive source-state binding')
        output = value.get('output_root')
        if output and (Path(output) / 'training' / job_id / 'state.json').resolve() == source_path:
            producers.append(value)
        for child, digest in bindings.items():
            if Path(child).suffix == '.json' and 'admission' in Path(child).name:
                walk(child, digest, depth + 1)
        active.remove(path)
        visited[path] = expected
    for path, digest in bound_files.items():
        if Path(path).suffix == '.json' and 'admission' in Path(path).name:
            walk(path, digest, 0)
    if len(producers) != 1:
        raise RuntimeError('Missing or ambiguous bound source generation producer')
    return {'producers': producers, 'verified_admission_hashes': visited}

def verify_source(state, source_path, *, job_id, target, bound_files=None):
    """Provider-free verification before own-full restore. URI names are opaque.

    A legacy V47 state has no skipped checkpoint saves and needs no new proof.
    Every V52 state carries this binding from its first restored full save.
    """
    root = Path(source_path).parent
    marker_path = root / MARKER
    provenance = resolve_source_generation(bound_files, source_path, job_id=job_id, target=target)
    source_admissions = provenance['producers']
    requires = any(any(Path(p).name == 'zero_gradient_cursor.py' for p in a.get('bound_files', {})) for a in source_admissions)
    if KEY not in state and not marker_path.exists() and not requires:
        return {'mode': 'legacy_checkpoint_each_attempt', 'zero_attempts_verified': 0}
    if KEY not in state or not marker_path.exists():
        raise RuntimeError('V52 source lost its required generation marker or cursor binding')
    b = check_binding(state)
    if sha(marker_path) != b.get('generation_marker_sha256'):
        raise RuntimeError('Changed immutable source generation marker')
    marker = json.loads(marker_path.read_text())
    if marker.get('schema') != SCHEMA or (marker.get('job_id'), marker.get('target')) != (job_id, target) or marker.get('state_directory') != str(root.resolve()):
        raise RuntimeError('Wrong immutable source generation marker')
    ap = marker['admission_path']
    if sha(ap) != marker['admission_sha256'] or (bound_files is not None and provenance['verified_admission_hashes'].get(ap) != marker['admission_sha256']):
        raise RuntimeError('Source generation admission is not immutably bound')
    admission = json.loads(Path(ap).read_text())
    if (admission.get('job_id'), admission.get('target')) != (job_id, target):
        raise RuntimeError('Source generation marker belongs to another job')
    helper = marker['helper_path']
    if admission.get('bound_files', {}).get(helper) != marker['helper_sha256'] or sha(helper) != marker['helper_sha256']:
        raise RuntimeError('Source generation helper is not immutably bound')
    tail = b['tail_sha256']
    after = cursor(state)
    for attempt in range(state['attempt'], b['saved_attempt'], -1):
        dest = root / f'attempt_{attempt:06d}'
        path = dest / RECEIPT
        if sha(path) != tail:
            raise RuntimeError('Changed/missing zero-gradient cursor proof')
        proof = json.loads(path.read_text())
        wanted = {
            'schema': SCHEMA, 'job_id': job_id, 'target': target,
            'checkpoint': b['checkpoint'], 'sampling_path': b['sampling_path'],
            'updates': b['updates'], 'saved_attempt': b['saved_attempt'],
            'attempt': attempt, 'previous_attempt': attempt - 1,
            'cursor_after': after, 'optimizer_dispatched': False,
        }
        if any(proof.get(k) != value for k, value in wanted.items()):
            raise RuntimeError('Zero-gradient proof identity/cursor mismatch')
        inputs = dest / 'update_inputs.json'
        if sha(inputs) != proof['update_inputs_sha256']:
            raise RuntimeError('Changed zero-gradient update inputs')
        if json.loads(inputs.read_text()).get('records') != []:
            raise RuntimeError('Zero-gradient proof contains optimizer records')
        if (dest / 'optimizer_metrics.json').exists():
            raise RuntimeError('Zero-gradient attempt has optimizer receipt')
        after = proof['cursor_before']
        tail = proof['previous_tail_sha256']
    if after != b['saved_cursor'] or tail is not None:
        raise RuntimeError('Zero-gradient chain does not reach saved full cursor')
    return {'mode': SCHEMA, 'zero_attempts_verified': b['zero_attempts'],
            'saved_attempt': b['saved_attempt'], 'restored_cursor_attempt': state['attempt'],
            'checkpoint': b['checkpoint'], 'sampling_path': b['sampling_path'],
            'updates': b['updates'], 'tail_sha256': b['tail_sha256']}


def immutable_receipt(path, value):
    """Receipt is durable before state can reference it; never overwrite evidence."""
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha(path)



def write_generation_marker(directory, admission_path, *, job_id, target):
    helper = str(Path(__file__).resolve())
    admission_path = str(Path(admission_path).resolve())
    admission = json.loads(Path(admission_path).read_text())
    if admission.get('bound_files', {}).get(helper) != sha(helper):
        raise RuntimeError('Generation helper absent from admission source bindings')
    if (admission.get('job_id'), admission.get('target')) != (job_id, target):
        raise RuntimeError('Generation marker admission identity mismatch')
    return immutable_receipt(Path(directory) / MARKER, {
        'schema': SCHEMA, 'state_directory': str(Path(directory).resolve()),
        'job_id': job_id, 'target': target, 'admission_path': admission_path,
        'admission_sha256': sha(admission_path), 'helper_path': helper,
        'helper_sha256': sha(helper),
    })


def enter(engine):
    if getattr(engine, '_zero_gradient_commit_inflight', False):
        raise RuntimeError('Prior commit did not finish; own-full recovery required')
    check_binding(engine.state)
    attempt = engine.prepared_step[2]
    if attempt != engine.state['attempt'] + 1 or engine.state.get('pending') != f'training_attempt_{attempt}':
        raise RuntimeError('Wrong prepared zero-gradient/optimizer attempt boundary')
    # Remains set on *any* failure, including a successful optimizer followed by
    # failed checkpoint/state/retention. No later sampling or update on dirty weights.
    engine._zero_gradient_commit_inflight = True


def choose_pair(engine, prepared, dest, attempt, proposed_cursor, name, permanent, updated):
    if updated:
        return engine.backend.checkpoint(name, permanent=True) if permanent else engine.backend.checkpoint(name)
    b = check_binding(engine.state)
    if prepared.get('records') != [] or (dest / 'optimizer_metrics.json').exists():
        raise RuntimeError('Unproven zero-gradient attempt')
    pair = copy.deepcopy(engine.state['pair_registry'][b['checkpoint']])
    if pair.get('checkpoint') != b['checkpoint'] or pair.get('sampling_path') != b['sampling_path']:
        raise RuntimeError('Current owned full/sampler pair missing from registry')
    proof = {
        'schema': SCHEMA, 'job_id': engine.job['job_id'], 'target': engine.config['target'],
        'checkpoint': b['checkpoint'], 'sampling_path': b['sampling_path'],
        'updates': b['updates'], 'saved_attempt': b['saved_attempt'],
        'attempt': attempt, 'previous_attempt': engine.state['attempt'],
        'cursor_before': cursor(engine.state),
        'cursor_after': copy.deepcopy(proposed_cursor) if engine.queue else {k: proposed_cursor[k] for k in ('pool_pass', 'next_batch')},
        'previous_tail_sha256': b['tail_sha256'],
        'update_inputs_sha256': sha(dest / 'update_inputs.json'),
        'optimizer_dispatched': False,
    }
    engine._zero_gradient_next_tail = immutable_receipt(dest / RECEIPT, proof)
    return pair


def commit_binding(engine, updated):
    if updated:
        bind_current(engine.state)
    else:
        b = engine.state[KEY]
        b['tail_sha256'] = engine._zero_gradient_next_tail
        b['zero_attempts'] += 1
        del engine._zero_gradient_next_tail
        check_binding(engine.state)


def success(engine):
    engine._zero_gradient_commit_inflight = False


def patched_commit_source(native_source):
    edits = (
        ('        batch,cursor,attempt,output,groups=self.prepared_step\n',
         '        _zg_enter(self)\n        batch,cursor,attempt,output,groups=self.prepared_step\n'),
        ("        pair=self.backend.checkpoint(name,permanent=True) if permanent else self.backend.checkpoint(name)\n",
         '        pair=_zg_choose_pair(self,prepared,dest,attempt,cursor,name,permanent,updated)\n'),
        ("        self.state['pair_registry'][pair['checkpoint']]=pair\n",
         "        _zg_commit_binding(self,updated)\n        self.state['pair_registry'][pair['checkpoint']]=pair\n"),
        ('        return self.state\n', '        _zg_success(self)\n        return self.state\n'),
    )
    for old, new in edits:
        if native_source.count(old) != 1:
            raise RuntimeError('Changed native commit patch location')
        native_source = native_source.replace(old, new)
    return native_source


def patched_finish_source(native_source):
    old = "        self.state.update(pair,final_pair=pair,status='finalizing',stop_reason=reason,pending=None)\n"
    if native_source.count(old) != 1:
        raise RuntimeError('Changed native final patch location')
    return native_source.replace(old, old + '        _zg_bind_current(self.state)\n')


def install(module):
    if sha(module.__file__) != EXPECTED_ENGINE:
        raise RuntimeError('Changed native Engine source')
    cls = module.Engine
    if getattr(cls, '_zero_gradient_cursor_installed', False):
        raise RuntimeError('Zero-gradient cursor patch already installed')
    source = patched_commit_source(inspect.getsource(cls.commit_step))
    namespace = dict(module.__dict__, _zg_enter=enter, _zg_choose_pair=choose_pair,
                     _zg_commit_binding=commit_binding, _zg_success=success, _zg_bind_current=bind_current)
    exec(compile(textwrap.dedent(source), str(Path(__file__).resolve()) + ':native_commit', 'exec'), namespace)
    final_source = patched_finish_source(inspect.getsource(cls.finish))
    exec(compile(textwrap.dedent(final_source), str(Path(__file__).resolve()) + ':native_finish', 'exec'), namespace)
    native_prepare = cls.prepare_step
    def prepare(self):
        if getattr(self, '_zero_gradient_commit_inflight', False):
            raise RuntimeError('Prior commit did not finish; own-full recovery required')
        return native_prepare(self)
    cls.commit_step = namespace['commit_step']
    cls.prepare_step = prepare
    cls.finish = namespace['finish']
    cls._zero_gradient_cursor_installed = True
