"""One matched sampling interface, with durable once-only task records.

Planning is offline. Live execution requires explicit checkpoint bindings,
matching source rows, and an output directory. Unknown calls are never replayed.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from .io import ROOT, read_json, sha256, write_json


def seed_for(row_id, draw):
    return int(
        hashlib.sha256(f"claim2-final-eval-v1:{row_id}:{draw}".encode()).hexdigest()[:8],
        16,
    ) % (2**31)


@lru_cache(maxsize=1)
def checkpoint_roster():
    return {
        r["checkpoint_id"]: r for r in read_json(ROOT / "data/runs/checkpoints.json")["checkpoints"]
    }


def selected(cfg):
    roster = (
        {
            r["checkpoint_id"]: r
            for r in read_json(resolve_path(cfg["checkpoint_roster"]))["checkpoints"]
        }
        if cfg.get("checkpoint_roster")
        else checkpoint_roster()
    )
    names = cfg["checkpoints"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Use an explicit, unique checkpoint list")
    chosen = [roster[n] for n in names]
    if len({r["model_family"] for r in chosen}) != 1:
        raise ValueError("Run each model family in its own process")
    identities = [(r["internal_target"], r["arm"], r["seed"]) for r in chosen]
    if len(identities) != len(set(identities)):
        raise ValueError("Choose one checkpoint per correctness target, arm, and seed")
    return chosen


def panel_rows(cfg):
    if cfg["panel"] not in ("ood", "id", "policy"):
        raise ValueError("Unknown panel")
    j = read_json(ROOT / f"data/panels/{cfg['panel']}.json")
    return j.get("row_ids") or [r["row_id"] for r in j["rows"]]


def plan(cfg, command="evaluate"):
    """Count the exact experiment grid. This needs no private files or SDK."""
    chosen = selected(cfg)
    rows = panel_rows(cfg)
    expected_modes = ["OFF", "TWO_STAGE_ON"] if cfg["panel"] == "ood" else ["TWO_STAGE_ON"]
    if cfg.get("modes", expected_modes) != expected_modes:
        raise ValueError("Modes differ from the frozen panel")
    draws = 3 if cfg["panel"] == "ood" else 1
    if cfg.get("draws", draws) != draws:
        raise ValueError("Draw count differs from the frozen panel")
    workers = cfg.get("sample_workers", 64)
    if type(cfg.get("grade_workers", 64)) is not int or cfg.get("grade_workers", 64) < 1:
        raise ValueError("Positive grade_workers required")
    if type(workers) is not int or workers < 1:
        raise ValueError("Positive sample_workers required")
    answers = len(chosen) * len(rows) * draws
    requests = len(chosen) * len(rows) * draws * sum(1 if m == "OFF" else 2 for m in expected_modes)
    return dict(
        command=command,
        panel=cfg["panel"],
        family=chosen[0]["model_family"],
        checkpoints=len(chosen),
        questions=len(rows),
        modes=expected_modes,
        draws_per_question=draws,
        answer_slots=answers * len(expected_modes),
        maximum_model_requests=requests,
        sample_workers=workers,
        grade_workers=cfg.get("grade_workers", 64),
        provider_calls=0,
        output_directory=cfg.get("output_directory"),
        semantic_grading=cfg.get("semantic_grading", False),
        budget_note="Tinker sampling and prepaid semantic grading are tracked separately",
    )


def resolve_path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def source_rows(cfg):
    """Load canonical inputs only after checking their recorded payload hash."""
    path = resolve_path(cfg["rows_json"])
    if cfg.get("rows_sha256") != sha256(path):
        raise ValueError("Row payload checksum mismatch")
    data = read_json(path)
    data = data.get("rows", data) if isinstance(data, dict) else data
    rows = {r["row_id"]: r for r in data}
    if len(rows) != len(data):
        raise ValueError("Duplicate source row ID")
    wanted = set(panel_rows(cfg))
    if not wanted <= rows.keys():
        raise ValueError("Missing frozen row payloads")
    return {k: rows[k] for k in sorted(wanted)}


def validate_bindings(cfg, chosen):
    """Check every checkpoint URI against its roster hash before any SDK call."""
    path = resolve_path(cfg["checkpoint_bindings"])
    if not path.is_file():
        raise ValueError("Missing checkpoint bindings; see docs/evaluation.md#prepare-local-inputs")
    bindings = read_json(path)
    for checkpoint in chosen:
        uri = bindings.get(checkpoint["checkpoint_id"])
        if not isinstance(uri, str) or not uri.startswith("tinker://"):
            raise ValueError("Missing sampler URI for " + checkpoint["checkpoint_id"])
        if hashlib.sha256(uri.encode()).hexdigest() != checkpoint["checkpoint_identity_sha256"]:
            raise ValueError(
                "Checkpoint binding does not match the roster: " + checkpoint["checkpoint_id"]
            )
    return bindings


def preflight(cfg, command="evaluate"):
    """Validate live inputs offline; report credential presence without exposing keys.

    The command neither constructs a provider client nor creates an output folder.
    Matching checkpoint weights and source payloads are required in addition to keys.
    """
    import os

    report = plan(cfg, command)
    rows = source_rows(cfg)
    chosen = selected(cfg)
    if command == "evaluate":
        validate_bindings(cfg, chosen)
        if not resolve_path(cfg["tokenizer_path"]).is_dir():
            raise ValueError(
                "Missing matching local tokenizer; see docs/evaluation.md#prepare-local-inputs"
            )
        SamplingBudget(cfg)  # Validate the budget without reserving or spending it.
    else:
        out = resolve_path(cfg["output_directory"])
        receipt = out / "configuration.json"
        if not receipt.is_file() or read_json(receipt) != cfg:
            raise ValueError(
                "Grading requires the matching configuration receipt from a sampling run"
            )
        if next((out / "records").glob("*/raw.json.gz"), None) is None:
            raise ValueError("No saved raw answers to grade in the configured run")
    required = ["TINKER_API_KEY"] if command == "evaluate" else []
    if cfg.get("semantic_grading"):
        required.append("OPENAI_API_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError("Set these environment variables before execution: " + ", ".join(missing))
    report.update(
        input_checks="passed",
        canonical_questions=len(rows),
        credentials_present=required,
        provider_calls=0,
    )
    return report


class NativeTinker:
    """The historical renderer suffixes and two-stage caps, with no retries."""

    async def initialize(self, cfg, chosen):
        # Fail on missing/mismatched private inputs before creating provider clients.
        bindings = validate_bindings(cfg, chosen)
        self.cash = SamplingBudget(cfg)
        from study import bootstrap

        family = chosen[0]["model_family"].lower()
        bootstrap.install(family)
        import tinker
        from tinker.lib.retry_handler import RetryConfig
        from tinker_cookbook import renderers
        from transformers import AutoTokenizer

        from study.vendor.shared.live_backend import disable_retries

        self.tinker = tinker
        self.family = family
        tok = AutoTokenizer.from_pretrained(
            str(resolve_path(cfg["tokenizer_path"])), local_files_only=True
        )
        self.off = renderers.get_renderer(
            "qwen3_5_disable_thinking" if family == "qwen" else "gpt_oss_no_sysprompt",
            tok,
        )
        self.on = renderers.get_renderer(
            "qwen3_5" if family == "qwen" else "gpt_oss_no_sysprompt", tok
        )
        self.service = tinker.ServiceClient()
        disable_retries(self.service)
        self.samplers = {}
        self.prompt_cache = OrderedDict()
        for r in chosen:
            uri = bindings[r["checkpoint_id"]]
            self.samplers[r["checkpoint_id"]] = await self.service.create_sampling_client_async(
                model_path=uri, retry_config=RetryConfig(enable_retry_logic=False)
            )

    async def answer(self, checkpoint, row, mode, draw, folder):
        from objective_prompts import messages

        from .native import gpt_modes, native_stages, thinking_adapter, two_stage_tokens

        seed = seed_for(row["row_id"], draw)
        on = mode == "TWO_STAGE_ON"
        tok = self.on.tokenizer
        cache_key = (row["row_id"], mode)
        if cache_key not in self.prompt_cache:
            msg = messages(row, "iou", "A")  # Identical prompt across reward choices.
            if self.family == "gpt":
                prompt = gpt_modes.prompt(self.on, msg, "ON" if on else "OFF")
            else:
                prompt = thinking_adapter.build_generation_prompt(
                    self.on if on else self.off, msg, thinking=on
                ).to_ints()
            self.prompt_cache[cache_key] = prompt
            if len(self.prompt_cache) > 128:
                self.prompt_cache.popitem(last=False)
        self.prompt_cache.move_to_end(cache_key)
        base = self.prompt_cache[cache_key]

        async def sample(tokens, cap, stop, stage):
            if len(tokens) + cap > 65536:
                raise ValueError("Context overflow; rows are never truncated")
            journal = folder / (stage + ".json")
            if journal.exists():
                raise RuntimeError(
                    "Prior physical request exists; explicit reconciliation required"
                )
            hold = self.cash.reserve(len(tokens), cap)
            write_json(
                journal,
                dict(
                    reserved_usd=str(hold),
                    status="dispatched",
                    seed=seed,
                    prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    max_tokens=cap,
                ),
            )
            response = await self.samplers[checkpoint["checkpoint_id"]].sample_async(
                self.tinker.ModelInput.from_ints(tokens),
                num_samples=1,
                sampling_params=self.tinker.SamplingParams(
                    max_tokens=cap, temperature=1.0, seed=seed, stop=stop
                ),
            )
            if len(response.sequences) != 1:
                raise ValueError("Expected one response")
            seq = response.sequences[0]
            r = dict(tokens=list(seq.tokens), logprobs=list(seq.logprobs or []))
            for field in ("stop_reason", "finish_reason"):
                v = getattr(seq, field, None)
                if v is not None:
                    r[field] = getattr(v, "value", str(v))
            if len(r["tokens"]) != len(r["logprobs"]):
                raise ValueError("Missing token/logprob alignment")
            self.cash.settle(hold, len(tokens), len(r["tokens"]))
            write_json(
                journal,
                dict(
                    status="complete",
                    input_tokens=len(tokens),
                    output_tokens=len(r["tokens"]),
                    result=r,
                ),
            )
            return r

        assembly = None
        if on:
            first = await sample(
                base,
                10000,
                [native_stages.CLOSE]
                if self.family == "gpt"
                else two_stage_tokens.stage1_stop_sequences(),
                "reasoning",
            )
            try:
                if self.family == "gpt":
                    assembly = native_stages.assemble(base, first, tok)
                else:
                    terminal = set(
                        tok.encode("<|im_end|>", add_special_tokens=False)
                        + tok.encode("<|endoftext|>", add_special_tokens=False)
                    )
                    assembly = two_stage_tokens.assemble_stage2(
                        base, first, tok, context_limit=65536, terminal_token_ids=terminal
                    )
            except ValueError as exc:
                return dict(
                    text="",
                    clean_termination=False,
                    known_pre_final_invalid=True,
                    row_failure=str(exc),
                    stage2_not_requested=True,
                    reasoning_text=tok.decode(first["tokens"], skip_special_tokens=False),
                )
            base = assembly["stage2_prompt_tokens"]
            write_json(folder / "assembly.json", assembly)
        final = await sample(base, 23000, self.off.get_stop_sequences(), "final")
        if self.family == "gpt":
            decoded = gpt_modes.decode_text(
                tok.decode(final["tokens"], skip_special_tokens=False), "OFF"
            )
            text = decoded["text"]
            clean = decoded["clean"]
        elif on:
            decoded = two_stage_tokens.decode_final(self.off, final, assembly)
            text = decoded["text"]
            clean = decoded["clean_termination"]
        else:
            decoded = thinking_adapter.decode_with_audit(self.off, final["tokens"], thinking=False)
            text = decoded["text"]
            clean = decoded["clean"]
        return dict(
            text=text,
            clean_termination=clean,
            decoder_audit=decoded,
            reasoning_text=assembly["reasoning_text"] if assembly else None,
        )


async def _execute_sample(cfg, backend, grade_pool):
    spec = plan(cfg)
    chosen = selected(cfg)
    rows = source_rows(cfg)
    out = resolve_path(cfg["output_directory"])
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "configuration.json"
    if receipt.exists() and read_json(receipt) != cfg:
        raise ValueError("Output configuration changed")
    write_json(receipt, cfg)
    backend = backend or NativeTinker()
    await backend.initialize(cfg, chosen)
    sample_sem = asyncio.Semaphore(spec["sample_workers"])
    grade_sem = asyncio.Semaphore(spec["grade_workers"])
    grader = make_grader(cfg, chosen[0]["model_family"]) if cfg.get("grade_as_ready") else None
    grade_pending = set()
    grade_lock = asyncio.Lock()
    loop = asyncio.get_running_loop()

    async def grade_one(folder, row):
        async with grade_sem:
            await loop.run_in_executor(grade_pool, grade_file, folder, row, grader)

    async def enqueue_grade(folder, row):
        async with grade_lock:
            grade_pending.add(asyncio.create_task(grade_one(folder, row)))
            if len(grade_pending) >= 2 * spec["grade_workers"]:
                done, pending = await asyncio.wait(
                    grade_pending, return_when=asyncio.FIRST_COMPLETED
                )
                grade_pending.clear()
                grade_pending.update(pending)
                for task in done:
                    task.result()

    async def one(cp, row, mode, draw):
        key = hashlib.sha256(
            f"{cp['checkpoint_id']}:{row['row_id']}:{mode}:{draw}".encode()
        ).hexdigest()
        folder = out / "records" / key
        folder.mkdir(parents=True, exist_ok=True)
        raw = folder / "raw.json.gz"
        started = folder / "started.json"
        async with sample_sem:
            if not raw.exists():
                if started.exists():
                    raise RuntimeError(
                        "Unknown prior sampling outcome; no automatic replay: " + key
                    )
                identity = dict(
                    checkpoint_id=cp["checkpoint_id"],
                    row_id=row["row_id"],
                    mode=mode,
                    draw=draw,
                    reward_target=cp["internal_target"],
                    checkpoint_identity_sha256=cp["checkpoint_identity_sha256"],
                )
                write_json(started, identity)
                try:
                    value = await backend.answer(cp, row, mode, draw, folder)
                except Exception as exc:
                    write_json(
                        folder / "unknown.json", dict(error_type=type(exc).__name__, no_retry=True)
                    )
                    raise
                write_json(raw, dict(identity, **value))
            elif (
                read_json(raw).get("checkpoint_identity_sha256") != cp["checkpoint_identity_sha256"]
            ):
                # A mutable custom roster must never relabel old answers as new weights.
                raise ValueError(
                    "Saved answer checkpoint identity differs from the selected roster"
                )
        if cfg.get("grade_as_ready"):
            await enqueue_grade(folder, row)

    pending = set()
    # Round-robin checkpoints inside the grid so every seed can be in flight.
    # Sampling and grading have independent bounded work sets and worker pools.
    for row, draw, mode, cp in itertools.product(
        rows.values(), range(spec["draws_per_question"]), spec["modes"], chosen
    ):
        pending.add(asyncio.create_task(one(cp, row, mode, draw)))
        if len(pending) >= 2 * spec["sample_workers"]:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
    if pending:
        await asyncio.gather(*pending)
    if grade_pending:
        await asyncio.gather(*grade_pending)
    write_json(
        out / "complete.json",
        dict(spec, provider_calls="Recorded in per-stage journals", complete=True),
    )


def make_grader(cfg, family):
    from study import bootstrap

    bootstrap.install(family.lower())
    if cfg.get("semantic_grading"):
        from astra_judge import AstraJudge

        return AstraJudge(
            resolve_path(cfg["output_directory"]) / "judge",
            max_calls=cfg["max_judge_calls"],
        )
    return None


def grade_file(folder, row, judge):
    from reward_pipeline import grounded, parse, score_all, turns

    folder = Path(folder)
    dest = folder / "scored.json.gz"
    raw = read_json(folder / "raw.json.gz")
    target = raw.get("reward_target")
    if target is None:
        target = checkpoint_roster()[raw["checkpoint_id"]]["internal_target"]
    if target == "SFT":
        target = "iou"
    if dest.exists():
        if read_json(dest)["raw_sha256"] != sha256(folder / "raw.json.gz"):
            raise ValueError("Changed raw response")
        return
    status = None
    try:
        scores = score_all(row, raw["text"], target, judge, clean=raw["clean_termination"])
        if scores["valid"] and judge is None:
            parsed = parse(raw["text"], [str(r["rule_id"]) for r in row["all_policy_rules"]])
            source = turns(row["conversation"])
            if any(grounded(source, a) for a in parsed["rule_assessments"]):
                status = "semantic_not_requested"
                scores["evidence_all"] = scores["evidence_fraction"] = None
                scores["rewards"]["C"] = scores["rewards"]["D"] = None
    except Exception as exc:
        # Provider/judge failures remain unavailable, never zero evidence.
        scores = score_all(row, raw["text"], target, None, clean=raw["clean_termination"])
        scores["evidence_all"] = scores["evidence_fraction"] = None
        scores["rewards"]["C"] = scores["rewards"]["D"] = None
        status = "semantic_unavailable:" + type(exc).__name__
    write_json(
        dest,
        dict(
            row_id=row["row_id"],
            raw_sha256=sha256(folder / "raw.json.gz"),
            scores=scores,
            semantic_unavailability=status,
        ),
    )


async def _execute_grade(cfg, grade_pool):
    selected(cfg)
    rows = source_rows(cfg)
    out = resolve_path(cfg["output_directory"])
    if read_json(out / "configuration.json") != cfg:
        raise ValueError("Output configuration changed")
    judge = make_grader(cfg, selected(cfg)[0]["model_family"])
    sem = asyncio.Semaphore(cfg.get("grade_workers", 64))

    async def one(p):
        r = read_json(p)
        async with sem:
            await asyncio.get_running_loop().run_in_executor(
                grade_pool, grade_file, p.parent, rows[r["row_id"]], judge
            )

    pending = set()
    for p in (out / "records").glob("*/raw.json.gz"):
        pending.add(asyncio.create_task(one(p)))
        if len(pending) >= 2 * cfg.get("grade_workers", 64):
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
    if pending:
        await asyncio.gather(*pending)


async def execute_sample(cfg, backend=None):
    import fcntl

    out = resolve_path(cfg["output_directory"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "owner.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with ThreadPoolExecutor(max_workers=plan(cfg)["grade_workers"]) as pool:
                return await _execute_sample(cfg, backend, pool)
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


class SamplingBudget:
    def __init__(self, cfg):
        from decimal import Decimal

        self.D = Decimal
        if cfg.get("provider_prices_verified") is not True:
            raise ValueError("Verify provider sampling prices before execution")
        self.limit = Decimal(str(cfg["sampling_cash_limit_usd"]))
        self.input = Decimal(str(cfg["sampling_prices_per_million"]["input"]))
        self.output = Decimal(str(cfg["sampling_prices_per_million"]["output"]))
        if (
            not all(x.is_finite() and x >= 0 for x in (self.limit, self.input, self.output))
            or self.limit <= 0
        ):
            raise ValueError("Invalid explicit sampling budget")
        self.path = resolve_path(cfg["output_directory"]) / "sampling-budget.json"
        prior = read_json(self.path) if self.path.exists() else {}
        self.spent = Decimal(prior.get("spent_uncached_upper_usd", "0"))
        self.held = Decimal(prior.get("held_upper_usd", "0"))
        if prior and (
            prior["limit_usd"] != str(self.limit)
            or prior["input_price"] != str(self.input)
            or prior["output_price"] != str(self.output)
        ):
            raise ValueError("Budget settings changed for an existing run")

    def cost(self, n, k):
        return (self.input * n + self.output * k) / 1000000

    def save(self):
        write_json(
            self.path,
            dict(
                limit_usd=str(self.limit),
                spent_uncached_upper_usd=str(self.spent),
                held_upper_usd=str(self.held),
                input_price=str(self.input),
                output_price=str(self.output),
                semantic_grading_cash_usd="0",
                note="Uncached sampling upper accounting; reconcile provider invoices separately",
            ),
        )

    def reserve(self, n, k):
        bound = self.cost(n, k)
        if self.spent + self.held + bound > self.limit:
            raise RuntimeError("Explicit Tinker sampling budget exhausted before dispatch")
        self.held += bound
        self.save()
        return bound

    def settle(self, hold, n, k):
        self.held -= hold
        self.spent += self.cost(n, k)
        self.save()


async def execute_grade(cfg):
    import fcntl

    out = resolve_path(cfg["output_directory"])
    with (out / "owner.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with ThreadPoolExecutor(max_workers=plan(cfg)["grade_workers"]) as pool:
                return await _execute_grade(cfg, pool)
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)
