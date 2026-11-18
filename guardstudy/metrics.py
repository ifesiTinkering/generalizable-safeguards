"""The paper's sixteen recorded metrics, including the unprinted support count.

The fifteen-metric main tables omit only supported prediction count. The
canonical kernels preserve confidence ties, fractional review at the boundary,
invalid-answer denominators, and undefined rather than fabricated correlations.
"""

from __future__ import annotations

import math

import numpy as np

from .vendor.secondary_metrics import NAMES
from .vendor.secondary_metrics import calculate as secondary

METRICS = ("rule_F1", "Jaccard_IoU") + tuple(NAMES)
PRINTED_METRICS = tuple(m for m in METRICS if m != "supported_prediction_count")
COLUMNS = (
    "valid",
    "confidence",
    "f1",
    "iou",
    "exact",
    "evidence_fraction",
    "supported_count",
    "precision",
    "recall",
)
LABELS = {
    "rule_F1": "Rule F1",
    "Jaccard_IoU": "Jaccard similarity",
    "rule_precision": "Precision",
    "rule_recall": "Recall",
    "exact_set_accuracy": "Exact rule-set accuracy",
    "parseability": "Parseable answers",
    "confidence_mean": "Mean confidence",
    "confidence_SD": "Confidence standard deviation",
    "IoU_ECE10": "Expected calibration error (10 bins)",
    "exact_error_AUROC": "Error detection: AUROC",
    "exact_error_AP": "Error detection: average precision",
    "F1_AURC": "F1 loss under confidence-based selection: AURC",
    "F1_loss_capture_review20": "F1 loss found by 20% review",
    "confidence_F1_Spearman": "Confidence–F1 rank correlation",
    "supported_predicted_fraction": "Mean supported rule fraction per answer",
    "supported_prediction_count": "Supported predicted rule count",
}
LOWER_IS_BETTER = {"IoU_ECE10", "F1_AURC"}


def calculate_many(values, weights):
    data = np.asarray(values, dtype=float)
    weights = np.asarray(weights)
    if data.ndim != 2 or data.shape[1] != 9:
        raise ValueError("Expected nine columns; see guardstudy.metrics.COLUMNS")
    if weights.ndim != 2 or weights.shape[1] != len(data):
        raise ValueError("Observation and frequency-weight dimensions differ")
    other = secondary(data, weights)
    total = weights.sum(axis=1)
    safe = np.where(total > 0, total, 1)
    f1 = (weights @ (data[:, 0] * data[:, 2])) / safe
    iou = (weights @ (data[:, 0] * data[:, 3])) / safe
    f1[total == 0] = np.nan
    iou[total == 0] = np.nan
    return np.column_stack([f1, iou, other])


def summarize(values) -> dict:
    if not len(values):
        return {name: None for name in METRICS}
    result = calculate_many(values, np.ones((1, len(values)), dtype=int))[0]
    return {name: float(v) if math.isfinite(v) else None for name, v in zip(METRICS, result)}


def strict_mean(values):
    values = list(values)
    if not values or any(v is None for v in values):
        return None
    return float(np.mean(values))


def evidence_bounds(values, predicted_counts):
    """Unknown semantic grades remain bounded; an empty prediction earns zero."""
    data = np.asarray(values, dtype=float)
    counts = np.asarray(predicted_counts)
    lower = np.nan_to_num(data[:, 5], nan=0.0)
    upper = np.where(np.isnan(data[:, 5]), (counts > 0).astype(float), data[:, 5])
    count_lower = np.nan_to_num(data[:, 6], nan=0.0)
    count_upper = np.where(np.isnan(data[:, 6]), counts, data[:, 6])
    return {
        "supported_predicted_fraction": {
            "lower": float(lower.mean()),
            "upper": float(upper.mean()),
        },
        "supported_prediction_count": {
            "lower": float(count_lower.mean()),
            "upper": float(count_upper.mean()),
        },
        "unknown_semantic_answers": int(np.isnan(data[:, 5]).sum()),
    }
