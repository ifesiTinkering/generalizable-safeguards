# ruff: noqa: E402
# Historical modules are imported after installing their model-specific namespace.
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from study.bootstrap import install

install("qwen")
import selection_majority_stop as stopping
from reward_pipeline import reward_values, score_all, set_f1
from update import build_update


def answer(ids, q=0.5):
    return (
        "<answer>"
        + json.dumps(ids)
        + "</answer><evidence></evidence><analysis></analysis><confidence>"
        + str(q)
        + "</confidence>"
    )


def row(gold):
    return {
        "row_id": "synthetic",
        "conversation": "Turn 0 - User: Synthetic example.",
        "policy": "Synthetic rule.",
        "policy_framework": "Synthetic framework",
        "all_policy_rules": [{"rule_id": "1", "rule_text": "Synthetic rule."}],
        "ground_truth_rule_ids": gold,
    }


def group(rewards):
    return {
        "prompt_tokens": [1, 2],
        "sequences": [{"tokens": [3, 4], "logprobs": [-0.5, -0.6]} for _ in rewards],
        "rewards": rewards,
    }


class Science(unittest.TestCase):
    def test_reward_mapping(self):
        self.assertEqual(reward_values(1, 0.8, 1, 0.5), {"A": 1, "B": 0.98, "C": 0.98, "D": 0.73})

    def test_empty_correct_has_no_vacuous_evidence_bonus(self):
        result = score_all(row([]), answer([], 1), "binary", None)
        self.assertTrue(result["valid"])
        self.assertEqual(result["exact"], 1)
        self.assertEqual(result["evidence_fraction"], 0)
        self.assertEqual(result["rewards"]["D"], 0.5)

    def test_invalid_is_below_every_valid_reward(self):
        result = score_all(row(["1"]), "bad", "iou", None)
        self.assertEqual(set(result["rewards"].values()), {-0.6})
        self.assertEqual(reward_values(0, 1, 0, 0)["D"], -0.5)

    def test_set_metrics_and_empty_convention(self):
        self.assertEqual(set_f1(set(), set()), 1)
        self.assertAlmostEqual(set_f1({"1", "2"}, {"1", "3", "4"}), 0.4)

    def test_exact_constant_group_never_manufactures_update(self):
        result = build_update([group([0.1] * 10)])
        self.assertEqual(result["records"], [])
        self.assertEqual(result["logs"]["groups"][0]["raw_advantages"], [0.0] * 10)

    def test_raw_centered_token_sum_alignment(self):
        result = build_update([group([0.0, 1.0])])
        a, b = result["records"]
        self.assertEqual(a["advantages"], [0.0, -0.5, -0.5])
        self.assertEqual(b["advantages"], [0.0, 0.5, 0.5])
        self.assertEqual(a["input_tokens"], [1, 2, 3])
        self.assertEqual(a["target_tokens"], [0, 3, 4])
        self.assertEqual(result["normalization"], "sum")
        self.assertEqual(result["denominator"], 1)
        self.assertEqual(result["loss_fn"], "importance_sampling")
        self.assertEqual(result["reference_kl_coefficient"], 0)

    def test_two_failed_components_stop(self):
        points = [
            {"update": 10 * i, "components": dict(zip(stopping.NAMES, x))}
            for i, x in enumerate(
                [
                    (0.5, 0.5, 0.1),
                    (0.5, 0.5, 0.2),
                    (0.5, 0.5, 0.1),
                    (0.5, 0.5, 0.1),
                    (0.5, 0.5, 0.1),
                ]
            )
        ]
        result = stopping.evaluate(points)
        self.assertTrue(result["stop"])
        self.assertEqual(result["improved"], ["evidence"])

    def test_unknown_is_not_a_failed_vote(self):
        points = [
            {"update": 10 * i, "components": dict(zip(stopping.NAMES, x))}
            for i, x in enumerate(
                [
                    (0.5, 0.5, 0.1),
                    (0.5, None, 0.2),
                    (0.5, 0.5, 0.1),
                    (0.5, 0.5, 0.1),
                    (0.5, 0.5, 0.1),
                ]
            )
        ]
        result = stopping.evaluate(points)
        self.assertFalse(result["stop"])
        self.assertEqual(result["unknown"], ["calibration"])

    def test_three_selection_components_invalid_zero(self):
        m = {
            "exact_set_accuracy": 0.5,
            "macro_f1": 0.6,
            "jaccard_iou": 0.4,
            "valid_rate": 0.8,
            "exact_correctness_brier": 0.25,
            "semantic_aggregate_evidence": 0.2,
            "semantic_per_rule_evidence": 0.4,
        }
        got = stopping.components(m)
        self.assertAlmostEqual(got["correctness"], 0.5)
        self.assertAlmostEqual(got["calibration"], 0.6)
        self.assertAlmostEqual(got["evidence"], 0.3)

    def test_config_and_source_verification_are_offline(self):
        for family in ("qwen", "gpt"):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "study",
                    "validate",
                    "--config",
                    str(ROOT / "configs" / "training" / f"{family}.example.json"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["provider_calls"], 0)

    def test_paid_entrypoint_requires_both_flags(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "study",
                "train",
                "--config",
                str(ROOT / "configs/training/qwen.example.json"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("no provider was constructed", completed.stderr)


if __name__ == "__main__":
    unittest.main()
