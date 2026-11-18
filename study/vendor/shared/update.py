"""Single shared update adapted only from pinned official Tinker tutorial104.

Reference:8486b8df2fc32e9713cf96dd5f75d43c7ef70a17/tutorials/104_first_rl.py:274-305.
Reward-minus-groupmean repeats on completion tokens; prompt coefficients zero;
SDK importance_sampling sums losses. No STD/PPO/response averaging/KL variants.
"""
from __future__ import annotations

import math
from typing import Any


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _tokens(values: Any, name: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty token sequence")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return list(values)


def build_update(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Official tutorial104 centered/raw-token-sum update for every target/arm.

    Additional infrastructure checks reject bad tensors and stabilize exact
    constant groups; neither arm nor correctness target is an input here.
    """
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a nonempty list")

    prepared = []
    all_response_tokens = 0
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError("each group must be a mapping")
        prompt = _tokens(group.get("prompt_tokens"), "prompt_tokens")
        sequences, rewards = group.get("sequences"), group.get("rewards")
        if not isinstance(sequences, (list, tuple)) or len(sequences) < 2:
            raise ValueError("each group needs at least two sequences")
        if not isinstance(rewards, (list, tuple)) or len(rewards) != len(sequences):
            raise ValueError("sequence/reward count mismatch")
        reward_values = [_finite(r, "reward") for r in rewards]
        # Exact constant detection prevents a rounded mean from manufacturing
        # tiny, same-sign "advantages" for numerically identical rewards.
        constant = all(r == reward_values[0] for r in reward_values)
        mean_reward = reward_values[0] if constant else sum(reward_values) / len(reward_values)
        centered = [0.0] * len(reward_values) if constant else [r - mean_reward for r in reward_values]
        seqs = []
        for sequence in sequences:
            if not isinstance(sequence, dict):
                raise ValueError("each sequence must be a mapping")
            tokens = _tokens(sequence.get("tokens"), "response tokens")
            logprobs = sequence.get("logprobs")
            if not isinstance(logprobs, (list, tuple)) or len(logprobs) != len(tokens):
                raise ValueError("sampled token/logprob length mismatch")
            logprobs = [_finite(v, "sampled logprob") for v in logprobs]
            seqs.append((tokens, logprobs))
            all_response_tokens += len(tokens)
        prepared.append((group_index, prompt, seqs, reward_values, centered, mean_reward))

    denominator = 1
    records = []
    group_logs = []
    active_tokens = 0
    raw_ratio_one_loss = 0.0
    for group_index, prompt, seqs, rewards, centered, mean_reward in prepared:
        active = any(centered)
        group_logs.append({
            "group_index": group_index,
            "sample_count": len(seqs),
            "reward_mean": mean_reward,
            "reward_variance": sum(a * a for a in centered) / len(centered),
            "raw_advantages": centered,
            "active": active,
            "response_tokens": sum(len(t) for t, _ in seqs),
        })
        if not active:
            continue
        padding = len(prompt) - 1
        for sequence_index, ((tokens, logprobs), reward, raw_advantage) in enumerate(zip(seqs, rewards, centered)):
            advantage = raw_advantage / denominator
            record = {
                "input_tokens": prompt + tokens[:-1],
                "target_tokens": [0] * padding + tokens,
                "sampling_logprobs": [0.0] * padding + logprobs,
                "advantages": [0.0] * padding + [advantage] * len(tokens),
                "group_index": group_index,
                "sequence_index": sequence_index,
                "prompt_token_count": len(prompt),
                "response_token_count": len(tokens),
                "reward": reward,
                "raw_advantage": raw_advantage,
                "scaled_advantage": advantage,
            }
            if len({len(record[k]) for k in ("input_tokens", "target_tokens", "sampling_logprobs", "advantages")}) != 1:
                raise AssertionError("internal next-token alignment failure")
            records.append(record)
            active_tokens += len(tokens)
            raw_ratio_one_loss -= raw_advantage * len(tokens)

    active_groups = sum(g["active"] for g in group_logs)
    return {
        "records": records,
        "denominator": denominator,
        "all_response_tokens": all_response_tokens,
        "normalization": "sum",
        "loss_fn": "importance_sampling", "loss_fn_config": None, "reference_kl_coefficient": 0.,
        "logs": {
            "algorithm": "official_Tinker_centered_raw_token_sum_importance_sampling",
            "group_count": len(groups),
            "active_groups": active_groups,
            "constant_groups": len(groups) - active_groups,
            "sample_count": sum(g["sample_count"] for g in group_logs),
            "datum_count": len(records),
            "all_response_tokens": all_response_tokens,
            "active_response_tokens": active_tokens,
            "normalization_denominator": denominator,
            "optimizer_update_needed": bool(records),
            "ratio_one_raw_loss": raw_ratio_one_loss,
            "ratio_one_scaled_loss": raw_ratio_one_loss / denominator,
            "groups": group_logs,
        },
    }
