"""Single fresh-job operational adapter; frozen scientific functions stay unchanged.

This entrypoint never adopts a live or interrupted output directory. Unknown
operations remain in their original journals for explicit recovery review.
"""

import fcntl
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from types import MethodType

from . import bootstrap
from .config import live_inputs, load_external_rows, sha, verify_sources


def install_scientific_adapters():
    import async_validation_engine
    import branch_namespace
    import evidence_budget
    import evidence_engine
    import live_backend
    import safe_admission
    import sampler_lifecycle
    import selection_majority_stop
    import validation_protocol
    import zero_gradient_cursor

    prior = validation_protocol.observe
    if evidence_engine.observe is not prior or async_validation_engine.observe is not prior:
        raise RuntimeError("Unexpected validation aliases")
    validation_protocol.observe = evidence_engine.observe = async_validation_engine.observe = (
        selection_majority_stop.observe
    )
    zero_gradient_cursor.install(evidence_engine)
    branch_namespace.install(live_backend)
    sampler_lifecycle.install(live_backend)
    safe_admission.install(
        evidence_budget, "dc4005bdc18cee26fb7980f63ae6f2ff9fd3606d1b9b6f4eaa0a57f0e98a7d9e"
    )


def guard_optimizer(engine, backend):
    original = backend.update

    def update(*args, **kwargs):
        s = engine.state
        if (
            s["updates"] >= 155
            or s.get("stop_decision_update") is not None
            or s.get("patience_status", {}).get("stop")
        ):
            raise RuntimeError("No optimizer dispatch after cap or valid validation stop")
        result = original(*args, **kwargs)
        if result.get("skipped") is not False:
            raise RuntimeError("A dispatched optimizer must report actual success")
        return result

    backend.update = update


def prepare_fresh_engine(engine, marker_sha256):
    import zero_gradient_cursor

    if engine.state is not None:
        raise RuntimeError("Fresh runner cannot overwrite or reset a saved state")
    engine.queue.initial_cursor = MethodType(
        lambda queue: queue.migrate(queue.original.initial_cursor()), engine.queue
    )
    engine.initialize()
    zero_gradient_cursor.bind_current(engine.state, marker_sha256=marker_sha256)
    from protocol import save

    save(engine.path, engine.state)


def cohort_bound(engine):
    from cost_plan import sampling_upper_bound, training_upper_bound
    from data.deterministic_queue import effective_domain
    from judge_factory import grading_upper

    cfg = engine.config
    bound = Decimal(".1")
    batch, _ = engine.planned_batch()
    for row in batch:
        candidates = [
            r for r in engine.rows["train"] if effective_domain(r) == effective_domain(row)
        ]
        max_prompt = max(len(engine.prompts[r["row_id"]]) for r in candidates)
        bound += (
            sampling_upper_bound(max_prompt, cfg["max_tokens"], n=cfg["samples_per_prompt"])
            * 3
            * cfg["max_domain_replacements"]
        )
        if engine.job["arm"] in ("C", "D"):
            bound += (
                max(grading_upper(r) for r in candidates)
                * cfg["samples_per_prompt"]
                * cfg["max_domain_replacements"]
            )
    longest = max(len(engine.prompts[r["row_id"]]) for r in engine.rows["train"])
    return bound + training_upper_bound(
        [longest + cfg["max_tokens"]] * (cfg["batch_size"] * cfg["samples_per_prompt"])
    )


def run_loop(engine, manager, ledger, budget_root, root):
    from cohort_budget import CohortMeter
    from judge_factory import make_judge
    from protocol import save
    from sampler_lifecycle import check_boundary
    from science_stop import should_stop

    sequence = 0
    while True:
        manager.poll(engine)
        stop = should_stop(engine.state, engine.config)
        if stop:
            if manager.has_pending(engine):
                time.sleep(0.2)
                continue
            meter = CohortMeter(
                ledger,
                engine.config["study_id"] + ":final",
                Decimal(".1"),
                root / "finalization.json",
            )
            engine.backend.meter = meter
            engine.state["actual_stop_update"] = engine.state["updates"]
            save(engine.path, engine.state)
            engine.finish(stop)
            meter.close()
            return
        check_boundary({engine.job["job_id"]: engine})
        sequence += 1
        save(root / "sequence.json", {"last_consumed_id": sequence})
        meter = CohortMeter(
            ledger,
            engine.config["study_id"] + ":cohort:" + str(sequence),
            cohort_bound(engine),
            root / "cohorts" / f"{sequence:06d}.json",
        )
        engine.backend.meter = meter
        engine.judge = make_judge(budget_root, meter, None)
        unavailable = engine.prepare_step()
        manager.poll(engine)
        if engine.state.get("stop_decision_update") is not None:
            engine.state["pending"] = None
            save(
                engine.out
                / f"attempt_{engine.state['attempt'] + 1:06d}"
                / "update_not_admitted.json",
                {"reason": "ordered_validation_stop", "optimizer_dispatched": False},
            )
            save(engine.path, engine.state)
        else:
            engine.commit_step(unavailable)
        if all(v["status"] in ("settled", "cancelled") for v in meter.calls.values()):
            meter.close()
        else:
            save(
                root / "unknown_holds.json",
                {"cohort_sequence": sequence, "reservation_retained": True},
            )
            raise RuntimeError("Unknown physical outcome; stop before new admission")


def run(cfg):
    live_inputs(cfg)
    verify_sources()
    bootstrap.install(cfg["model_family"])
    import judge_factory

    judge_factory.configure_prepaid(
        cfg.get("semantic_api_prepaid", False), cfg.get("max_judge_calls", 1000000)
    )
    rows, bindings = load_external_rows(cfg)
    # Construct and validate the exact queue and renderer before any SDK client.
    from data.appended_queue import AppendedQueue

    AppendedQueue(rows["train"], rows["supplement"])
    from .model_adapter import prepare

    renderer, prompts, decode = prepare(cfg, rows)
    for key in ("TINKER_API_KEY", "OPENAI_API_KEY"):
        if not os.environ.get(key):
            raise ValueError("Missing environment credential: " + key)
    root = Path(cfg["output_directory"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)  # No accidental resume/overwrite.
    with (root / "owner.lock").open("x") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = dict(
            cfg,
            study_id="evidence:public:" + uuid.uuid4().hex,
            algorithm_candidate_audited=True,
            storage_reserve_confirmed=True,
            permanent_retention_authorized=True,
            storage_max_pairs_per_job=70,
            stop_sequences=renderer.get_stop_sequences(),
            source_verification_passed=True,
            explicit_paid_opt_in=True,
        )
        cfg["budget_namespace"] = cfg["study_id"]
        cfg["output_root"] = str(root)
        from supplement_protocol import authorize

        authorize(cfg)
        install_scientific_adapters()
        import zero_gradient_cursor
        from cohort_budget import CohortMeter
        from cost_plan import sampling_upper_bound
        from evidence_budget import Ledger
        from evidence_engine import Engine
        from judge_factory import grading_upper, make_judge
        from live_backend import LiveBackend
        from protocol import save
        from validation_manager import ValidationManager

        job = {
            "job_id": f"{cfg['model_family']}_{cfg['target']}_{cfg['arm']}_seed{cfg['seed']}",
            "seed": cfg["seed"],
            "arm": cfg["arm"],
        }
        dest = root / "training" / job["job_id"]
        dest.mkdir(parents=True)
        budget = root / "accounting"
        ledger = Ledger(budget / "budget.sqlite", cfg["cash_limit_usd"], cash_stop_enabled=True)
        sid = cfg["study_id"] + ":storage"
        if not ledger.reserve(
            sid,
            Decimal(str(cfg["storage_reserve_usd"])),
            {"kind": "storage_allocation_not_provider_invoice"},
        ) or not ledger.dispatch(sid):
            raise RuntimeError("Storage envelope was not reserved")
        save(root / "configuration.json", cfg)
        save(root / "external_data_bindings.json", bindings)
        helper = str(Path(zero_gradient_cursor.__file__).resolve())
        admission = {
            "job_id": job["job_id"],
            "target": cfg["target"],
            "output_root": str(root),
            "bound_files": {helper: sha(helper)},
            "config_sha256": cfg["config_sha256"],
            "fresh_study": True,
        }
        ap = root / "portable_admission.json"
        save(ap, admission)
        marker = zero_gradient_cursor.write_generation_marker(
            dest, ap, job_id=job["job_id"], target=cfg["target"]
        )
        samples = judges = manager = backend = None
        status = "errored"
        try:
            meter = CohortMeter(
                ledger, cfg["study_id"] + ":initial", Decimal(".1"), root / "initialize_cohort.json"
            )
            backend = LiveBackend(cfg, dest, meter)
            source_model = cfg["initial_checkpoint"].removeprefix("tinker://").split("/")[0]
            if backend.client.model_id == source_model:
                raise RuntimeError("SFT source was not cloned into an isolated fresh RL model")
            samples = ThreadPoolExecutor(max_workers=cfg["sample_workers"])
            judges = ThreadPoolExecutor(max_workers=cfg["judge_workers"])
            training_rows = {
                "train": rows["train"] + rows["supplement"],
                "validation": rows["validation"],
            }
            engine = Engine(
                cfg,
                job,
                dest,
                backend,
                prompts,
                decode,
                make_judge(budget, meter, None),
                training_rows,
                samples,
                judges,
            )
            guard_optimizer(engine, backend)
            prepare_fresh_engine(engine, marker)
            meter.close()
            validation_bound = sum(
                (
                    3 * sampling_upper_bound(len(prompts[r["row_id"]]), 23000) + grading_upper(r)
                    for r in rows["validation"]
                ),
                Decimal(".1"),
            )
            manager = ValidationManager(
                root, ledger, None, validation_bound, cfg["study_id"], workers=2
            )
            manager.samples.shutdown(wait=True)
            manager.judges.shutdown(wait=True)
            manager.samples = ThreadPoolExecutor(max_workers=cfg["validation_sample_workers"])
            manager.judges = ThreadPoolExecutor(max_workers=cfg["validation_judge_workers"])
            engine.validation_manager = manager
            manager.enqueue(engine)
            save(
                root / "initialization_committed.json",
                {
                    "fresh_RL_optimizer": True,
                    "model_id": backend.client.model_id,
                    "source_full": cfg["initial_checkpoint"],
                    "updates": 0,
                    "checkpoint": engine.state["checkpoint"],
                },
            )
            run_loop(engine, manager, ledger, budget, root)
            status = "success"
        except BaseException as exc:
            from sampler_lifecycle import LifecycleBoundaryPause

            if isinstance(exc, LifecycleBoundaryPause):
                status = "interrupted"
            save(
                root / "halt.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "scientific_completion": False,
                    "unknown_outcomes_preserved": True,
                    "automatic_resume": False,
                },
            )
            raise
        finally:
            if manager is not None:
                manager.close()
            if samples is not None:
                samples.shutdown(wait=True)
            if judges is not None:
                judges.shutdown(wait=True)
            if backend is not None:
                backend.close(status=status)
            save(
                root / "owner_quiesced.json",
                {
                    "pid": os.getpid(),
                    "status": status,
                    "unix": time.time(),
                    "no_checkpoint_deletions": True,
                },
            )
