"""New-process-only retention for samplers referenced by a live provider session."""
import hashlib
import uuid
import threading
import time
from decimal import Decimal
from pathlib import Path

EXPECTED = '12cec41ec422de8c76309e14e2a156891fb1e5aff100ad1463145c1328d36c4b'

class LifecycleBoundaryPause(RuntimeError):
    """Operational own-checkpoint pause; never scientific completion."""

def registry(backend):
    return backend.service._claim2_sampler_registry

def check_boundary(engines):
    for engine in engines.values():
        backend = engine.backend
        r = registry(backend)
        with r['lock']:
            if time.monotonic() - r['started'] >= 24 * 3600:
                raise LifecycleBoundaryPause('Live sampler session reached 24-hour rotation boundary')
        # Reserve the next pair AND a potential final pair before admission.
        if len(backend._owned_paths()) + 4 > 2 * (backend.config['storage_max_pairs_per_job'] + 1):
            raise LifecycleBoundaryPause('Live sampler storage reached funded checkpoint rotation boundary')


def install(module):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Unreviewed backend source')
    cls = module.LiveBackend
    if getattr(cls, '_live_sampler_retention', False):
        raise ValueError('Lifecycle patch already installed')
    original_set = cls.set_sampling_path
    original_checkpoint = cls.checkpoint

    def checkpoint(self, *args, **kwargs):
        self._sampler_creation_thread = threading.get_ident()
        try:
            return original_checkpoint(self, *args, **kwargs)
        finally:
            self._sampler_creation_thread = None

    def set_sampling_path(self, path):
        if not hasattr(self.service, '_claim2_sampler_registry'):
            self.service._claim2_sampler_registry = {'paths': set(), 'lock': threading.RLock(), 'started': time.monotonic()}
        r = registry(self)
        with r['lock']:
            if path not in r['paths'] and getattr(self, '_sampler_creation_thread', None) != threading.get_ident():
                raise LifecycleBoundaryPause('Cannot attach inherited or unverified-expiry sampler')
            # The same ServiceClient is shallow-copied for async validation.
            # Record before creating the sampling client; unknown creation stays pinned.
            r['paths'].add(path)
        if getattr(self, 'sampling_path', None) == path and getattr(self, 'sampler', None) is not None:
            return
        return original_set(self, path)

    def prune(self, keep_pairs, *, request_id=None):
        if not isinstance(keep_pairs, list):
            raise ValueError('Provide explicit owned checkpoint pairs')
        keep = {p[k] for p in keep_pairs for k in ('checkpoint', 'sampling_path')}
        owned = self._owned_paths()
        if not keep or not keep <= owned:
            raise PermissionError('Missing or foreign keep paths')
        # Old SDK sampling sessions remain associated with the ServiceClient.
        # Deleting their weights can poison that client even after all samples
        # returned. Retain all owned samplers until an explicit terminal-session
        # cleanup; optimizer checkpoint retention remains unchanged.
        r = registry(self)
        with r['lock']:
            retained_samplers = set(r['paths'])
        if not retained_samplers <= owned:
            raise RuntimeError('Live sampler missing from durable owned journal')
        keep |= retained_samplers
        # Successor task preserves checkpoints pending separate cleanup review.
        # The existing boundary rotates the client before its funded envelope
        # fills; ownership_scope carries all durable pins into the next client.
        return {'kept_paths': sorted(owned), 'deleted': [],
                'cleanup_deferred_paths': sorted(owned - keep),
                'live_session_sampler_retention': sorted(retained_samplers)}

    cls.set_sampling_path = set_sampling_path
    cls.checkpoint = checkpoint
    cls.prune = prune
    cls._live_sampler_retention = True
