"""Conversation-paired inference on the fixed OOD checkpoint panel.

The RNG sequence and six endpoints match the frozen calibration analysis.
Integer-frequency kernels batch the calculations without changing the statistic.
"""

import itertools

import numpy as np

from .data import cells
from .metrics import METRICS, calculate_many

ENDPOINTS = (
    "rule_F1",
    "IoU_ECE10",
    "exact_error_AUROC",
    "F1_AURC",
    "F1_loss_capture_review20",
    "confidence_F1_Spearman",
)
QSEEDS = [0, 1, 2, 3, 4, 5, 7, 36, 42]


def adjust(rows, endpoints=ENDPOINTS):
    slots = [r["metrics"][m] for r in rows for m in endpoints]
    n = len(slots)
    order = sorted(range(n), key=lambda i: slots[i]["raw_p"])
    running = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        running = min(running, slots[i]["raw_p"] * n / (rank + 1))
        slots[i]["BH_q"] = running
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, slots[i]["raw_p"] * (n - rank)))
        slots[i]["Holm_p"] = running
    return rows


def calibration_scope(
    family,
    target,
    source,
    mode,
    regime="OOD",
    reps=1999,
    batch=64,
    paired_seeds=None,
    arms=("A", "B"),
):
    """Recompute one within-model arm contrast using paired conversations.

    Return an effect, percentile interval, and raw permutation p-value for each
    endpoint. ``calibration_all`` adds the full family's adjusted p-values.
    The legacy ID slots are intentionally unavailable here: the later ID400
    experiment has a separate packet and correction family.
    """
    ss = ([0, 1, 2, 3] if target == "binary" else QSEEDS) if family == "Qwen" else [2, 4, 5]
    if paired_seeds is not None:
        ss = paired_seeds
    domains = (
        ["content_moderation", "specialized_advice", "intellectual_property"]
        if regime == "OOD"
        else ["privacy", "fraud", "defamation"]
    )
    # ID slots in this original OOD family remain unavailable. ID400 has its own family.
    idx = cells("ood", family)
    required = [
        (target, a, s, mode, source + ":" + d) for a, s, d in itertools.product(arms, ss, domains)
    ]
    missing = [k for k in required if k not in idx]
    result = dict(
        family=family,
        target=target,
        source=source,
        mode=mode,
        regime=regime,
        contrast=arms[1] + "-" + arms[0],
        paired_seeds=ss,
        complete=not missing,
    )
    if missing:
        result["metrics"] = {
            m: dict(difference=None, percentile95=None, raw_p=1.0, available=False)
            for m in ENDPOINTS
        }
        return result
    convs = sorted({v for k in required for v in idx[k]["conversation_ids"]})
    ci = {c: i for i, c in enumerate(convs)}
    pairs = []
    for s, d in itertools.product(ss, domains):
        a, b = [idx[target, arm, s, mode, source + ":" + d] for arm in arms]
        for name in ("row_ids", "draws", "conversation_ids"):
            if a[name] != b[name]:
                raise ValueError("Paired task mismatch: " + name)
        pairs.append(
            (
                np.asarray(a["values"], float),
                np.asarray(b["values"], float),
                np.array([ci[c] for c in a["conversation_ids"]]),
            )
        )
    take = [METRICS.index(m) for m in ENDPOINTS]

    def evaluate(weights, swaps=False):
        out = []
        for a, b, ix in pairs:
            w = weights[:, ix]
            if swaps:
                joined = np.concatenate([a, b])
                wa = np.concatenate([1 - w, w], axis=1)
                wb = np.concatenate([w, 1 - w], axis=1)
                delta = calculate_many(joined, wb)[:, take] - calculate_many(joined, wa)[:, take]
            else:
                delta = calculate_many(b, w)[:, take] - calculate_many(a, w)[:, take]
            out.append(delta)
        return np.mean(out, axis=0)

    point = evaluate(np.ones((1, len(convs)), dtype=int))[0]
    rng = np.random.default_rng(816731)
    bootw = np.stack(
        [rng.multinomial(len(convs), np.full(len(convs), 1 / len(convs))) for _ in range(reps)]
    )
    swaps = np.stack([rng.integers(0, 2, len(convs), dtype=np.int8) for _ in range(reps)])
    boot = np.concatenate([evaluate(bootw[i : i + batch]) for i in range(0, reps, batch)])
    null = np.concatenate([evaluate(swaps[i : i + batch], True) for i in range(0, reps, batch)])
    result["conversations"] = len(convs)
    result["metrics"] = {}
    for k, m in enumerate(ENDPOINTS):
        bv = boot[:, k]
        nv = null[:, k]
        nb = int(np.isfinite(bv).sum())
        nn = int(np.isfinite(nv).sum())
        available = bool(np.isfinite(point[k]) and nn >= 0.95 * reps)
        raw = (
            (1 + int((np.abs(nv[np.isfinite(nv)]) >= abs(point[k]) - 1e-14).sum()) + reps - nn)
            / (reps + 1)
            if available
            else 1.0
        )
        result["metrics"][m] = dict(
            difference=float(point[k]) if np.isfinite(point[k]) else None,
            percentile95=np.quantile(bv[np.isfinite(bv)], [0.025, 0.975]).tolist()
            if nb >= 0.95 * reps
            else None,
            raw_p=raw,
            available=available,
            defined_replicates=nb,
            defined_permutations=nn,
            total_replicates=reps,
        )
    return result


def calibration_all():
    """Recreate the complete original family, retaining its unavailable slots."""
    rows = [
        calibration_scope(f, t, s, m, r)
        for f, t, s, m, r in itertools.product(
            ("Qwen", "GPT"),
            ("binary", "iou"),
            ("safepyramid", "dynabench"),
            ("OFF", "TWO_STAGE_ON"),
            ("ID", "OOD"),
        )
    ]
    return dict(
        schema="portable-calibration-inference-v1",
        provider_calls=0,
        family_size=192,
        results=adjust(rows),
        scope="Conditional on fixed checkpoints and retained conversations",
    )
