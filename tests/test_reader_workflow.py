"""Check user-facing exports and preflight boundaries without provider calls."""

import csv
import hashlib
import json
import sys

import pytest

from guardstudy import evaluation
from guardstudy.io import ROOT, read_json, write_json
from guardstudy.recompute import write_test_csv
from guardstudy.verify import verify_analysis_inputs


def test_evidence_support_export_keeps_separate_adjustment_and_bounds(tmp_path):
    source = read_json(ROOT / "data/statistics/ood_latest_C.json")
    row = source["ood_all_support_sensitivity"][0]
    result = dict(schema="portable-evidence-inference-v1", results=[], support_sensitivity=[row])
    write_test_csv(result, tmp_path / "evidence.json.gz")
    with (tmp_path / "evidence.support.csv").open() as f:
        exported = list(csv.DictReader(f))
    assert exported
    for r in exported:
        original = row["metrics"][r["metric"]]
        assert json.loads(r["evidence_bounds"]) == original["identification_bounds"]
        assert json.loads(r["interval95"]) == original["bootstrap95_envelope"]
        assert float(r["adjusted_p_upper"]) == original["secondary_BH512_upper"]
        assert r["effect_direction"] == row["contrast"]
        assert "separate support-only family" in r["p_adjustment"]


def test_recomputed_correctness_csv_matches_paper_direction(tmp_path):
    row = next(
        r
        for r in read_json(ROOT / "data/statistics/correctness_points.json")["results"]
        if r["tests"]["rule_F1"]["difference"] is not None
    )
    result = dict(
        schema="portable-correctness-conservative-v2",
        results=[dict(row, tests={"rule_F1": row["tests"]["rule_F1"]})],
    )
    write_test_csv(result, tmp_path / "correctness.json.gz")
    with (tmp_path / "correctness.csv").open() as f:
        r = next(csv.DictReader(f))
    original = row["tests"]["rule_F1"]
    assert float(r["difference"]) == -original["difference"]
    assert float(r["adjusted_p"]) == original["BH_q"]
    assert json.loads(r["interval95"]) == [-v for v in reversed(original["percentile95"])]
    assert r["effect_direction"] == "binary-minus-" + row["high_target"]


def test_preflight_checks_inputs_without_sdk_or_output_directory(tmp_path, monkeypatch):
    cp = dict(evaluation.checkpoint_roster()["qwen-jaccard-b-s2"])
    uri = "tinker://test-only/sampler_weights/checkpoint"
    cp["checkpoint_identity_sha256"] = hashlib.sha256(uri.encode()).hexdigest()
    monkeypatch.setattr(evaluation, "selected", lambda cfg: [cp])
    monkeypatch.setattr(evaluation, "panel_rows", lambda cfg: ["question"])
    rows = tmp_path / "rows.json"
    write_json(rows, [dict(row_id="question")])
    bindings = tmp_path / "bindings.json"
    write_json(bindings, {cp["checkpoint_id"]: uri})
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    from guardstudy.io import sha256

    cfg = dict(
        panel="id",
        checkpoints=[cp["checkpoint_id"]],
        rows_json=str(rows),
        rows_sha256=sha256(rows),
        checkpoint_bindings=str(bindings),
        tokenizer_path=str(tokenizer),
        output_directory=str(tmp_path / "new-run"),
        sampling_cash_limit_usd=100,
        provider_prices_verified=True,
        sampling_prices_per_million=dict(input=1, output=1),
        semantic_grading=True,
    )
    monkeypatch.setenv("TINKER_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    before = set(sys.modules)
    result = evaluation.preflight(cfg)
    assert result["input_checks"] == "passed" and result["provider_calls"] == 0
    assert not (tmp_path / "new-run").exists()
    assert not {"tinker", "transformers"} & (set(sys.modules) - before)
    write_json(bindings, {cp["checkpoint_id"]: uri + "-changed"})
    with pytest.raises(ValueError, match="does not match"):
        evaluation.preflight(cfg)
    assert not (tmp_path / "new-run").exists()


def test_archived_p_values_cannot_change_without_integrity_failure(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "tests.json"
    write_json(path, dict(p=0.01))
    from guardstudy.io import sha256

    write_json(
        tmp_path / "provenance/analysis_inputs.json",
        dict(files=[dict(path="data/tests.json", sha256=sha256(path))]),
    )
    assert verify_analysis_inputs(tmp_path) == 1
    write_json(path, dict(p=0.04))
    with pytest.raises(ValueError, match="Changed archived analysis input"):
        verify_analysis_inputs(tmp_path)


def test_new_correctness_analysis_rejects_mixed_archived_inputs(tmp_path, capsys):
    from guardstudy.cli import main

    output = tmp_path / "result.json"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "recompute",
                "--analysis",
                "correctness",
                "--qwen-cells",
                "new-qwen.json.gz",
                "--output",
                str(output),
            ]
        )
    assert exc.value.code == 2
    assert "--gpt-id-cells" in capsys.readouterr().err
    assert not output.exists()


def test_evidence_claim_export_joins_every_saved_support_test():
    from guardstudy.reproduce import comparisons, key

    source = read_json(ROOT / "data/statistics/ood_latest_C.json")
    expected = {
        key(dict(r, regime="OOD")): r
        for r in source["ood_all_support_sensitivity"]
        if r["target"] in ("binary", "iou")
    }
    exported = [
        r
        for r in comparisons()
        if r["panel"] == "ood"
        and r["metric"] == "supported_predicted_fraction"
        and r["contrast"] in ("C-B", "D-B")
    ]
    assert len(exported) == len(expected) and expected
    for row in exported:
        support = expected[key(row)]["metrics"]["supported_predicted_fraction"]
        assert row["evidence_bounds"] == support["identification_bounds"]
        assert row["evidence_interval95"] == support["bootstrap95_envelope"]
        assert row["adjusted_p"] == support["secondary_BH512_upper"]
        assert row["evidence_adjusted_p_upper"] == support["secondary_BH512_upper"]
        assert row["available"]


def test_resume_cannot_relabel_saved_answers_as_different_weights(tmp_path, monkeypatch):
    import asyncio

    from guardstudy.io import sha256

    cp = dict(evaluation.checkpoint_roster()["qwen-jaccard-b-s2"])
    cp["checkpoint_identity_sha256"] = "changed-weights"
    monkeypatch.setattr(evaluation, "selected", lambda cfg: [cp])
    monkeypatch.setattr(evaluation, "panel_rows", lambda cfg: ["question"])
    source = tmp_path / "rows.json"
    write_json(source, [dict(row_id="question")])
    cfg = dict(
        panel="id",
        checkpoints=[cp["checkpoint_id"]],
        rows_json=str(source),
        rows_sha256=sha256(source),
        output_directory=str(tmp_path / "run"),
        sample_workers=1,
        grade_workers=1,
        grade_as_ready=False,
    )
    key = hashlib.sha256(f"{cp['checkpoint_id']}:question:TWO_STAGE_ON:0".encode()).hexdigest()
    write_json(
        tmp_path / "run/records" / key / "raw.json.gz",
        dict(checkpoint_identity_sha256="old-weights"),
    )

    class NeverSample:
        async def initialize(self, cfg, chosen):
            pass

        async def answer(self, *args):
            pytest.fail("A completed answer must not trigger another provider call")

    with pytest.raises(ValueError, match="checkpoint identity differs"):
        asyncio.run(evaluation.execute_sample(cfg, NeverSample()))
