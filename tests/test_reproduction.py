import asyncio
import gzip
import json

import numpy as np
import pytest

from guardstudy import evaluation
from guardstudy.archives import shard
from guardstudy.inference import ENDPOINTS, calibration_scope
from guardstudy.io import ROOT, read_json, sha256, write_json
from guardstudy.metrics import evidence_bounds, summarize
from guardstudy.reproduce import comparisons
from study.config import read_config


def test_model_specific_adapter_ranks(tmp_path):
    for family, rank in [("qwen", 64), ("gpt", 32)]:
        cfg = read_json(ROOT / f"configs/training/{family}.example.json")
        p = tmp_path / (family + ".json")
        write_json(p, cfg)
        assert read_config(p)["lora_rank"] == rank
        cfg["lora_rank"] = rank + 1
        write_json(p, cfg)
        with pytest.raises(ValueError):
            read_config(p)


def test_missing_evidence_does_not_become_zero_or_drop_answer():
    data = [[1, 0.8, 0.5, 1 / 3, 0, None, None, 0.5, 0.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    m = summarize(data)
    assert m["rule_F1"] == 0.25 and m["parseability"] == 0.5
    assert m["supported_predicted_fraction"] is None
    b = evidence_bounds(data, [2, 0])
    assert b["supported_predicted_fraction"] == {"lower": 0.0, "upper": 0.5}


def test_published_calibration_inference_replays_from_answers():
    with np.errstate(all="ignore"):
        actual = calibration_scope("GPT", "iou", "safepyramid", "TWO_STAGE_ON")
    expected = next(
        r
        for r in read_json(ROOT / "data/statistics/ood_calibration.json.gz")["results"]
        if all(r[k] == actual[k] for k in ("family", "target", "source", "mode", "regime"))
    )
    for m in ENDPOINTS:
        a, b = actual["metrics"][m], expected["metrics"][m]
        assert a["raw_p"] == b["raw_p"]
        assert a["difference"] == pytest.approx(b["difference"], abs=1e-12)
        assert a["percentile95"] == pytest.approx(b["percentile95"], abs=1e-12)


def test_correctness_export_preserves_p_values_and_reverses_interval():
    result = next(
        r
        for r in comparisons()
        if r["panel"] == "correctness"
        and r["family"] == "GPT"
        and r["source"] == "dynabench"
        and r["regime"] == "OOD"
        and r["mode"] == "OFF"
        and r["target"] == "iou"
        and r["arm"] == "A"
        and r["metric"] == "rule_F1"
    )
    source = next(
        r
        for r in read_json(ROOT / "data/statistics/correctness_points.json")["results"]
        if r["family"] == "GPT"
        and r["source"] == "dynabench"
        and r["regime"] == "OOD"
        and r["mode"] == "OFF"
        and r["high_target"] == "iou"
        and r["arm"] == "A"
    )["tests"]["rule_F1"]
    assert result["difference"] == -source["difference"]
    assert result["interval95"] == [
        -source["percentile95"][1],
        -source["percentile95"][0],
    ]
    assert result["adjusted_p"] == source["BH_q"]


def test_answer_archive_roundtrip(tmp_path):
    p = tmp_path / "input.jsonl.gz"
    records = [{"text": "A" * 1000, "n": n} for n in range(50)]
    with gzip.open(p, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "archive"
    m = shard(p, out, max_mib=1)
    back = []
    for part in m["parts"]:
        assert sha256(out / part["path"]) == part["sha256"]
        with gzip.open(out / part["path"], "rt") as f:
            back.extend(json.loads(line) for line in f)
    assert back == records and m["records"] == 50


def test_offline_plan_does_not_import_provider(monkeypatch):
    import sys

    before = set(sys.modules)
    report = evaluation.plan({"panel": "policy", "checkpoints": ["qwen-jaccard-b-s2"]})
    assert (
        report["questions"] == 84
        and report["answer_slots"] == 84
        and report["maximum_model_requests"] == 168
    )
    assert report["provider_calls"] == 0 and not {"tinker", "transformers"} & (
        set(sys.modules) - before
    )


class FakeBackend:
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.checkpoint_order = []

    async def initialize(self, cfg, chosen):
        pass

    async def answer(self, checkpoint, row, mode, draw, folder):
        self.calls += 1
        self.checkpoint_order.append(checkpoint["checkpoint_id"])
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return dict(
            text='<answer>["1"]</answer><evidence></evidence><analysis>Rule one applies.</analysis><confidence>0.8</confidence>',
            clean_termination=True,
        )


def test_parallel_pipeline_is_idempotent_and_grades_as_answers_arrive(tmp_path, monkeypatch):
    rs = [
        dict(
            row_id="test-" + str(n),
            conversation="Turn 1 - User: Hello\nTurn 2 - Assistant: Example",
            policy="A rule.",
            policy_framework="Example",
            all_policy_rules=[dict(rule_id="1")],
            ground_truth_rule_ids=["1"],
        )
        for n in range(4)
    ]
    p = tmp_path / "rows.json"
    write_json(p, rs)
    monkeypatch.setattr(evaluation, "panel_rows", lambda cfg: [r["row_id"] for r in rs])
    cfg = dict(
        panel="id",
        checkpoints=["qwen-jaccard-a-s2", "qwen-jaccard-b-s2"],
        rows_json=str(p),
        rows_sha256=sha256(p),
        output_directory=str(tmp_path / "run"),
        sample_workers=4,
        grade_workers=4,
        grade_as_ready=True,
        semantic_grading=False,
    )
    backend = FakeBackend()
    asyncio.run(evaluation.execute_sample(cfg, backend))
    assert backend.calls == 8 and backend.peak > 1
    assert len(set(backend.checkpoint_order[:4])) == 2
    scored = list((tmp_path / "run/records").glob("*/scored.json.gz"))
    assert len(scored) == 8
    assert all(read_json(p)["scores"]["f1"] == 1 for p in scored)
    asyncio.run(evaluation.execute_sample(cfg, backend))
    assert backend.calls == 8


def test_native_two_stage_preserves_sampled_reasoning_and_rejects_short_unknown_stop():
    from guardstudy.native.two_stage_tokens import assemble_stage2

    class Tokenizer:
        def encode(self, s, **kwargs):
            return list(map(ord, s))

        def decode(self, tokens, **kwargs):
            return "".join(map(chr, tokens))

    tok = Tokenizer()
    base = tok.encode("<|im_start|>assistant\n<think>\n")
    tokens = tok.encode("Reasoning.</think>")
    r = {"tokens": tokens, "logprobs": [0.0] * len(tokens)}
    a = assemble_stage2(base, r, tok, context_limit=65536)
    assert (
        a["stage1_sampled_tokens"] == tokens
        and a["stage2_prompt_tokens"][: len(base) + len(tokens)] == base + tokens
    )
    assert a["supplied_tokens_are_sampled"] is False and a["stage2_max_tokens"] == 23000
    with pytest.raises(ValueError):
        assemble_stage2(base, {"tokens": [65], "logprobs": [0.0]}, tok, context_limit=65536)


def test_prepaid_training_grading_keeps_cash_out_of_bounds():
    from study import bootstrap

    bootstrap.install("qwen")
    from decimal import Decimal

    import judge_factory

    judge_factory.configure_prepaid(True, 10)
    assert judge_factory.grading_upper({}) == Decimal(".000006")

    class Grant:
        def reserve(self, key, spec):
            return True

    class Meter:
        def reserve(self, key, value, spec):
            assert (
                value == Decimal(".000001")
                and spec["funding"] == "grant"
                and spec["cash_expense_usd"] == "0"
            )
            return True

    a = judge_factory.PrepaidAdapter("gpt-6-astra", Meter(), Grant(), post=lambda *a, **k: None)
    assert a.reserve("test", {"model": "gpt-6-astra"}) is True
    judge_factory.configure_prepaid(False)


def test_sampling_budget_holds_unknown_calls_before_admitting_more(tmp_path):
    from decimal import Decimal

    from guardstudy.evaluation import SamplingBudget

    cfg = dict(
        output_directory=str(tmp_path),
        provider_prices_verified=True,
        sampling_cash_limit_usd="0.00005",
        sampling_prices_per_million={"input": 1, "output": 1},
    )
    budget = SamplingBudget(cfg)
    hold = budget.reserve(10, 20)
    assert hold == Decimal("0.00003")
    with pytest.raises(RuntimeError):
        budget.reserve(10, 20)
    budget.settle(hold, 10, 5)
    assert budget.spent == Decimal("0.000015") and budget.held == 0
    assert budget.reserve(10, 20) == hold


def test_checkpoint_selection_rejects_incomplete_and_prefers_later_tie():
    from guardstudy.selection import choose, composite

    rows = [
        dict(
            row_id=str(n),
            valid=True,
            exact=1,
            f1=1,
            iou=1,
            confidence=1,
            evidence_all=1,
            evidence_fraction=1,
        )
        for n in range(90)
    ]
    assert composite(rows)["score"] == 1
    rows[0]["judge_unavailable"] = True
    with pytest.raises(ValueError, match="Unresolved"):
        composite(rows)
    winner = choose(
        [
            dict(status="complete", update=u, score=s)
            for u, s in [(90, 1), (100, 0.8), (110, 0.8), (156, 1)]
        ]
    )
    assert winner["update"] == 110


def test_duplicate_seed_checkpoint_is_rejected(tmp_path):
    original = dict(evaluation.checkpoint_roster()["qwen-jaccard-b-s2"])
    duplicate = dict(original, checkpoint_id="newer-qwen-jaccard-b-s2")
    roster = tmp_path / "roster.json"
    write_json(roster, {"checkpoints": [original, duplicate]})
    with pytest.raises(ValueError, match="one checkpoint"):
        evaluation.selected(
            {
                "checkpoint_roster": str(roster),
                "checkpoints": [original["checkpoint_id"], duplicate["checkpoint_id"]],
            }
        )
