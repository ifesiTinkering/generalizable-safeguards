"""Authorized-only, metered Tinker transport for the isolated Claim 2 study.

Importing this module creates no clients and imports no model SDK. Every
provider operation has an exclusive durable journal and a ledger reservation
before dispatch. Unknown outcomes retain their reservations and block resume.
There is no provider retry. Pruning is restricted to this job's registered
checkpoint paths and requires an explicit nonempty set of checkpoints to keep.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import protocol
from cost_plan import MODEL, sampling_cost, training_upper_bound


class UncertainOperation(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Provider result contains a nonfinite float")
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_numpy"):
        return {"data": _jsonable(value.to_numpy().tolist()),
                "dtype": str(value.dtype), "shape": list(value.shape)}
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"Unsupported provider result type: {type(value).__name__}")


def disable_retries(client: Any) -> None:
    """Installed SDK layout guard, copied into this study without shared imports.

    ServiceClient(max_retries=0) alone is insufficient for this SDK. Mutate its
    HTTP-pool settings and holder retry wrapper on the SDK event loop BEFORE
    billable requests. Unknown SDK internals fail closed.
    """
    holder = client.holder
    if getattr(holder, "_claim2_no_retries", False):
        return
    required = ("_constructor_kwargs", "_client_pools", "run_coroutine_threadsafe", "execute_with_retries")
    if any(not hasattr(holder, key) for key in required):
        raise RuntimeError("Unknown SDK retry layout; refusing provider dispatch")

    async def install():
        holder._constructor_kwargs["max_retries"] = 0
        for pool in holder._client_pools.values():
            if not hasattr(pool, "_constructor_kwargs") or not hasattr(pool, "_clients"):
                raise RuntimeError("Unknown SDK connection pool")
            pool._constructor_kwargs["max_retries"] = 0
            for http in pool._clients:
                http.max_retries = 0

        async def once(func, *args, **kwargs):
            return await func(*args, **kwargs)

        holder.execute_with_retries = once
        holder._claim2_no_retries = True

    holder.run_coroutine_threadsafe(install()).result()


class LiveBackend:
    def __init__(self, config: dict[str, Any], outdir: str | Path, meter: Any,
                 state: dict[str, Any] | None = None, *, sdk: Any = None):
        # Defense in depth: the root runner must gate as well. This runs before
        # even importing Tinker or constructing a service, including fake tests.
        protocol.require_authorized()
        self.config = dict(config)
        if config.get("model") != MODEL:
            raise ValueError("No verified prices for this model")
        if config.get("storage_reserve_confirmed") is not True:
            raise ValueError("Study storage/overhead reserve must be funded first")
        self.outdir = Path(outdir)
        self.journal = self.outdir / "transport_journal"
        self.journal.mkdir(parents=True, exist_ok=True)
        self.meter = meter
        for p in self.journal.glob("*.json"):
            existing = json.loads(p.read_text())
            if existing.get("status") != "complete":
                raise UncertainOperation(f"Unresolved provider operation: {p.name}; no automatic replay")
            if state is None and existing.get("spec", {}).get("kind") == "update":
                raise UncertainOperation("Existing updates require a durable resume checkpoint")
        if sdk is None:
            import tinker
            from tinker.lib.retry_handler import RetryConfig
            self.sdk, self.retry_class = tinker, RetryConfig
        else:
            self.sdk, self.retry_class = sdk, sdk.RetryConfig
        self.service = self.sdk.ServiceClient()
        disable_retries(self.service)
        self.client = None
        self.sampler = None
        self.sampling_path = None
        checkpoint = state["checkpoint"] if state else config["initial_checkpoint"]
        initialize_id = "initialize:" + uuid.uuid4().hex

        def initialize():
            method = (self.service.create_training_client_from_state_with_optimizer
                      if state else self.service.create_training_client_from_state)
            self.client = method(checkpoint)
            disable_retries(self.client)
            return {"checkpoint": checkpoint, "restore_optimizer": bool(state)}

        self._execute(initialize_id, {"kind": "initialize", "checkpoint": checkpoint,
                      "restore_optimizer": bool(state)}, Decimal(0), initialize)
        if state and state.get("sampling_path"):
            self.set_sampling_path(state["sampling_path"])

    def _path(self, request_id: str) -> Path:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("A nonempty stable request_id is required")
        return self.journal / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")

    def _execute(self, request_id: str, spec: dict[str, Any], upper: Decimal,
                 invoke: Any, actual_cost: Any = None, *, allow_cached: bool = True) -> dict[str, Any]:
        path = self._path(request_id)
        spec = _jsonable(spec)
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("request_id") != request_id or prior.get("spec") != spec:
                raise ValueError("Request ID reused with different operation")
            if prior.get("status") == "complete" and allow_cached:
                return prior["result"]
            raise UncertainOperation("Operation already exists; no automatic update replay or unknown-call retry")
        # Ledger requires a strictly positive reservation. Non-token control
        # operations reserve one microdollar and settle zero; their storage is
        # independently covered by the study's funded overhead reserve.
        reservation = max(upper, Decimal("0.000001"))
        record = {"request_id": request_id, "spec": spec, "upper_bound_usd": str(reservation),
                  "status": "prepared", "storage_covered_by_study_reserve": True}
        # Exclusive creation closes races; protocol.save supplies fsync thereafter.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, allow_nan=False); f.flush(); os.fsync(f.fileno())
        try:
            if not self.meter.reserve(request_id, reservation, spec):
                raise UncertainOperation("Ledger ID already reserved or used; refusing a duplicate dispatch")
            if not self.meter.dispatch(request_id):
                raise UncertainOperation("Ledger dispatch already claimed")
            record["status"] = "dispatched"
            protocol.save(path, record)
            result = _jsonable(invoke())
            charge = actual_cost(result) if actual_cost else upper
            record.update(status="response_received", result=result, actual_usd=str(charge))
            protocol.save(path, record)
            self.meter.settle(request_id, charge)
            record["status"] = "complete"
            protocol.save(path, record)
            return result
        except BaseException as exc:
            # Never release a reservation or replay after a dispatch exception.
            record["error_type"] = type(exc).__name__
            protocol.save(path, record)
            raise

    def set_sampling_path(self, path: str) -> None:
        if not isinstance(path, str) or not path.startswith("tinker://"):
            raise ValueError("Expected a Tinker sampling path")
        disable_retries(self.service)
        self.sampler = self.service.create_sampling_client(
            model_path=path, retry_config=self.retry_class(enable_retry_logic=False))
        disable_retries(self.sampler)
        retry = getattr(getattr(self.sampler, "retry_handler", None), "config", None)
        if getattr(retry, "enable_retry_logic", True):
            raise RuntimeError("Sampling retries must be disabled")
        self.sampling_path = path

    def sample(self, prompt_tokens: list[int], seed: int, max_tokens: int,
               request_id: str) -> dict[str, Any]:
        if self.sampler is None:
            raise RuntimeError("Save/set a sampling checkpoint before sampling")
        if not prompt_tokens or any(type(t) is not int or t < 0 for t in prompt_tokens):
            raise ValueError("Invalid prompt tokens")
        if type(seed) is not int or not 0 <= seed < 2**31 or type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Invalid sampling seed or completion ceiling")
        stops = self.config.get("stop_sequences")
        if not isinstance(stops, (list, tuple)) or not stops:
            raise ValueError("Renderer stop_sequences must be supplied explicitly")
        spec = {"kind": "sample", "sampling_path": self.sampling_path, "seed": seed,
                "max_tokens": max_tokens, "temperature": 1.0, "prompt_tokens": len(prompt_tokens),
                "prompt_sha256": hashlib.sha256(json.dumps(prompt_tokens).encode()).hexdigest(),
                "stop_sequences": list(stops)}
        upper = sampling_cost(len(prompt_tokens), max_tokens)

        def invoke():
            result = self.sampler.sample(self.sdk.ModelInput.from_ints(list(prompt_tokens)), num_samples=1,
                sampling_params=self.sdk.SamplingParams(max_tokens=max_tokens, temperature=1.0,
                                                       seed=seed, stop=list(stops))).result()
            if len(result.sequences) != 1:
                raise RuntimeError("Expected exactly one independently seeded sample")
            sequence = result.sequences[0]
            tokens = list(sequence.tokens)
            if sequence.logprobs is None or len(sequence.logprobs) != len(tokens):
                raise RuntimeError("Missing aligned sampling logprobs")
            return {"tokens": tokens, "logprobs": list(sequence.logprobs)}

        return self._execute(request_id, spec, upper, invoke,
                             lambda r: sampling_cost(len(prompt_tokens), len(r["tokens"])))

    def update(self, prepared: dict[str, Any], request_id: str) -> dict[str, Any]:
        if self.config.get('algorithm_candidate_audited') is not True:raise RuntimeError('Official Tinker candidate requires exact algorithm audit before dispatch')
        if prepared.get('loss_fn')!='importance_sampling' or prepared.get('loss_fn_config') is not None or prepared.get('reference_kl_coefficient')!=0.:raise RuntimeError('Unexpected algorithm objective')
        records = prepared["records"]
        if not records:
            return {"skipped": True, "reason": "no_nonconstant_groups", "backward_metrics": {},
                    "optimizer_metrics": {}, "loss_fn_outputs": []}
        datums = []
        for record in records:
            n = len(record["input_tokens"])
            fields = {"target_tokens": (record["target_tokens"], "int64"),
                      "logprobs": (record["sampling_logprobs"], "float32"),
                      "advantages": (record["advantages"], "float32")}
            if any(len(values) != n for values, _ in fields.values()):
                raise ValueError("Datum length mismatch")
            datums.append(self.sdk.Datum(model_input=self.sdk.ModelInput.from_ints(record["input_tokens"]),
                loss_fn_inputs={key: self.sdk.types.TensorData(data=values, dtype=dtype, shape=[n])
                                for key, (values, dtype) in fields.items()}))
        upper = training_upper_bound([len(r["input_tokens"]) + 1 for r in records])
        adam = self.config["adam"]
        spec = {"kind": "update", "datum_count": len(records), "loss_fn": "importance_sampling", "loss_fn_config": prepared["loss_fn_config"], "reference_kl_coefficient": 0.,
                "normalization": prepared["normalization"], "denominator": prepared["denominator"],
                "training_tokens_upper_bound": sum(len(r["input_tokens"]) + 1 for r in records),
                "learning_rate": self.config["learning_rate"], "adam": adam,
                "records_sha256": hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()}

        def invoke():
            disable_retries(self.client)
            backward = self.client.forward_backward(datums, loss_fn="importance_sampling")
            step = self.client.optim_step(self.sdk.AdamParams(learning_rate=self.config["learning_rate"], **adam))
            b, s = backward.result(), step.result()
            return {"skipped": False, "backward_metrics": getattr(b, "metrics", {}),
                    "optimizer_metrics": getattr(s, "metrics", {}),
                    "loss_fn_output_type": getattr(b, "loss_fn_output_type", None),
                    "loss_fn_outputs": getattr(b, "loss_fn_outputs", []),
                    "accounting": "conservative_full_sequence_token_bound_retained"}

        return self._execute(request_id, spec, upper, invoke, allow_cached=False)

    def checkpoint(self, name: str, *, final: bool = False, permanent: bool = False, request_id: str | None = None) -> dict[str, Any]:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("Invalid checkpoint name")
        ttl = None if final or permanent else 172800
        if permanent and self.config.get("permanent_retention_authorized") is not True:raise RuntimeError("Permanent retention not admitted")
        # All jobs share the cohort ledger namespace, even though their journals
        # and provider training clients are separate. Names alone collide.
        rid = request_id or self.outdir.name + ":checkpoint:" + name
        state = self._execute(rid + ":state", {"kind": "save_state", "name": name, "ttl_seconds": ttl}, Decimal(0),
            lambda: {"path": self.client.save_state(name, ttl_seconds=ttl, overwrite=False).result().path})
        sample = self._execute(rid + ":sampler", {"kind": "save_sampler", "name": name, "ttl_seconds": ttl}, Decimal(0),
            lambda: {"path": self.client.save_weights_for_sampler(name, ttl_seconds=ttl).result().path})
        self.set_sampling_path(sample["path"])
        return {"checkpoint": state["path"], "sampling_path": sample["path"], "ttl_seconds": ttl}

    def _owned_paths(self) -> set[str]:
        paths = set()
        deleted = set()
        for p in self.journal.glob("*.json"):
            r = json.loads(p.read_text())
            if r.get("status") == "complete" and r.get("spec", {}).get("kind") in {"save_state", "save_sampler"}:
                path = r.get("result", {}).get("path")
                if isinstance(path, str) and path.startswith("tinker://") and path != self.config["initial_checkpoint"]:
                    paths.add(path)
            if r.get("status") == "complete" and r.get("spec", {}).get("kind") == "delete_own_checkpoint":
                deleted.add(r["spec"]["path"])
        return paths - deleted

    def retain(self, pair: dict[str, str], ttl_seconds: int = 604800,
               *, request_id: str | None = None) -> dict[str, Any]:
        if ttl_seconds is not None and (type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 604800):
            raise ValueError("Finite retention must be at most seven days")
        if ttl_seconds is None and self.config.get("permanent_retention_authorized") is not True:raise RuntimeError("Permanent retention not admitted")
        self._check_retention_coverage(ttl_seconds)
        paths = [pair["checkpoint"], pair["sampling_path"]]
        if any(p not in self._owned_paths() for p in paths):
            raise PermissionError("Only checkpoint paths created by this job's completed save journals may be retained")
        rest = self.service.create_rest_client()
        rid = request_id or "retain:" + uuid.uuid4().hex
        results = []
        for index, path in enumerate(paths):
            def invoke(path=path):
                rest.set_checkpoint_ttl_from_tinker_path(path, ttl_seconds).result()
                import time
                renewed=time.time()
                return {"path": path, "ttl_seconds": ttl_seconds, "renewed_at_unix":renewed, "expires_at_unix":renewed+ttl_seconds if ttl_seconds is not None else None}
            results.append(self._execute(rid + ":" + str(index), {"kind": "retain", "path": path,
                "ttl_seconds": ttl_seconds}, Decimal(0), invoke))
        return {"retained": results}

    def _check_retention_coverage(self, ttl_seconds, pair_count=1):
        if self.config.get('storage_reserve_confirmed') is not True:
            raise RuntimeError('Storage allocation not funded')
        if (ttl_seconds is not None and ttl_seconds > 604800) or len(self._owned_paths()) > 2*(self.config['storage_max_pairs_per_job']+1):
            raise RuntimeError('Unexpected retention exposure')
        return True

    def renew_archives(self, pairs):
        """Renew owned archives at most daily, only with an explicit funded receipt.

        Receipt is admission evidence, not a claim that provider storage is free.
        Parent must populate it after approving total retained storage exposure.
        """
        import time
        pairs=list({(p['checkpoint'],p['sampling_path']):p for p in pairs}.values())
        if not pairs:return
        now=time.time();expiry={}
        for p in self.journal.glob('*.json'):
            record=json.loads(p.read_text())
            if record.get('status')=='complete' and record.get('spec',{}).get('kind')=='retain':
                value=record.get('result',{})
                expiry[value.get('path')]=max(expiry.get(value.get('path'),0),float('inf') if value.get('ttl_seconds') is None else value.get('expires_at_unix',0))
        for pair in pairs:
            if any(expiry.get(pair[k],0)<now+6*86400 for k in ('checkpoint','sampling_path')):
                self._check_retention_coverage(604800, len(pairs))
                self.retain(pair,604800)

    def prune(self, keep_pairs: list[dict[str, str]], *, request_id: str | None = None) -> dict[str, Any]:
        """Keep explicit owned archives plus rolling validations and best/current.

        This operation never lists or deletes global checkpoints. Every candidate
        must originate in this job's completed save journal. A failed/unknown
        deletion retains its reservation and blocks automatic resume.
        """
        if not isinstance(keep_pairs, list):
            raise ValueError("Provide explicit owned checkpoint pairs")
        unique={(p['checkpoint'],p['sampling_path']) for p in keep_pairs}
        if not unique:
            raise ValueError("Retain an explicit nonempty owned set")
        owned = self._owned_paths()
        keep = set()
        for pair in unique:
            keep.update(pair)
        if not keep <= owned:
            raise PermissionError("Prune keep-set includes a path not owned by this job")
        candidates = sorted(owned - keep)
        rest = self.service.create_rest_client()
        rid = request_id or "prune:" + uuid.uuid4().hex
        results = []
        for path in candidates:
            def invoke(path=path):
                rest.delete_checkpoint_from_tinker_path(path).result()
                return {"path": path, "deleted": True}
            suffix = hashlib.sha256(path.encode()).hexdigest()
            results.append(self._execute(rid + ":" + suffix,
                {"kind": "delete_own_checkpoint", "path": path, "keep_paths": sorted(keep)}, Decimal(0), invoke))
        return {"kept_paths": sorted(keep), "deleted": results}

    def close(self, status: str = "success") -> None:
        if status not in {"success", "errored", "interrupted"}:
            raise ValueError("Invalid service close status")
        self.service.close(status=status).result()
