# ruff: noqa: E402
# Historical modules are imported after installing their model-specific namespace.
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from study.bootstrap import install

install("qwen")
import evidence_engine
import zero_gradient_cursor as z
from protocol import save

from study.config import FIXED
from study.runner import guard_optimizer, prepare_fresh_engine


class Backend:
    def __init__(self):
        self.saves = []
        self.updates = []
        self.current = None

    def checkpoint(self, name, **kwargs):
        self.saves.append(name)
        return {
            "checkpoint": "tinker://synthetic:train:0/weights/" + name,
            "sampling_path": "tinker://synthetic:train:0/sampler_weights/" + name,
        }

    def set_sampling_path(self, path):
        self.current = path

    def update(self, prepared, **kwargs):
        self.updates.append(prepared)
        return {"skipped": False}

    def renew_archives(self, pairs):
        pass

    def prune(self, pairs):
        pass


class Boundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not getattr(evidence_engine.Engine, "_zero_gradient_cursor_installed", False):
            z.install(evidence_engine)

    def fixture(self, path):
        original = []
        supplement = []
        for domain in ("privacy", "fraud", "defamation"):
            base = {
                "row_id": domain,
                "domain": domain,
                "native_level": "L0",
                "ground_truth_rule_ids": ["1"],
            }
            original.append(base)
            supplement.append(dict(base, row_id=domain + "-extra", supplement_source="policytraj"))
        cfg = dict(FIXED, target="binary", study_id="evidence:synthetic")
        job = {"job_id": "synthetic", "seed": 2, "arm": "A"}
        backend = Backend()
        engine = evidence_engine.Engine(
            cfg,
            job,
            path,
            backend,
            {},
            None,
            None,
            {"train": original + supplement, "validation": []},
            None,
            None,
        )
        engine.validation_manager = SimpleNamespace(
            enqueue=lambda e: None, has_pending=lambda e: False
        )
        ap = path / "portable_admission.json"
        save(
            ap,
            {
                "job_id": "synthetic",
                "target": "binary",
                "bound_files": {str(Path(z.__file__).resolve()): z.sha(z.__file__)},
            },
        )
        marker = z.write_generation_marker(path, ap, job_id="synthetic", target="binary")
        guard_optimizer(engine, backend)
        prepare_fresh_engine(engine, marker)
        return engine, backend

    def prepare(self, e, rewards):
        batch, cursor = e.planned_batch()
        attempt = e.state["attempt"] + 1
        groups = [
            {
                "prompt_tokens": [1, 2],
                "sequences": [{"tokens": [3, 4], "logprobs": [-0.5, -0.5]} for _ in rewards],
                "rewards": rewards,
            }
            for _ in batch
        ]
        output = {
            r["row_id"]: [
                (seq, {"row_id": r["row_id"], "reward": reward})
                for seq, reward in zip(g["sequences"], rewards)
            ]
            for r, g in zip(batch, groups)
        }
        e.state["pending"] = f"training_attempt_{attempt}"
        save(e.path, e.state)
        e.prepared_step = (batch, cursor, attempt, output, groups)

    def test_zero_attempt_advances_durable_cursor_without_saving_weights(self):
        with tempfile.TemporaryDirectory() as d:
            e, b = self.fixture(Path(d))
            pair = e.state["checkpoint"]
            before = copy.deepcopy(e.state["queue_cursor"])
            self.prepare(e, [0.0] * 10)
            e.commit_step(set())
            saved = json.loads(e.path.read_text())
            self.assertEqual(saved["attempt"], 1)
            self.assertEqual(saved["updates"], 0)
            self.assertNotEqual(saved["queue_cursor"], before)
            self.assertEqual(saved["checkpoint"], pair)
            self.assertEqual(b.saves, ["initial"])
            self.assertEqual(b.updates, [])
            self.assertEqual(
                z.verify_source(saved, e.path, job_id="synthetic", target="binary")[
                    "zero_attempts_verified"
                ],
                1,
            )

    def test_positive_154_to155_once_then_optimizer_guard(self):
        with tempfile.TemporaryDirectory() as d:
            e, b = self.fixture(Path(d))
            e.state["updates"] = 154
            z.bind_current(e.state)
            self.prepare(e, [0.0, 1.0] * 5)
            e.commit_step(set())
            self.assertEqual(e.state["updates"], 155)
            self.assertEqual(len(b.updates), 1)
            with self.assertRaises(RuntimeError):
                b.update({"records": [{}]})

    def test_failed_positive_save_forbids_later_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            e, b = self.fixture(Path(d))
            self.prepare(e, [0.0, 1.0] * 5)
            b.checkpoint = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("synthetic save failure")
            )
            with self.assertRaises(RuntimeError):
                e.commit_step(set())
            with self.assertRaisesRegex(RuntimeError, "Prior commit did not finish"):
                e.prepare_step()

    def test_missing_generation_marker_cannot_downgrade_proof(self):
        with tempfile.TemporaryDirectory() as d:
            e, b = self.fixture(Path(d))
            self.prepare(e, [0.0] * 10)
            e.commit_step(set())
            (Path(d) / z.MARKER).unlink()
            with self.assertRaises(RuntimeError):
                z.verify_source(e.state, e.path, job_id="synthetic", target="binary")


if __name__ == "__main__":
    unittest.main()
