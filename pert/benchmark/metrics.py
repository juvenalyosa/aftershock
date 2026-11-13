"""Dependency-free evaluation metrics (correlations, ROC AUC, average precision).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


# --- column extraction from dict rows --------------------------------------------------


def as_float(value: object) -> Optional[float]:
    """Finite float or None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def numeric_column(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    """Column values as floats (NaN when missing)."""
    values: List[float] = []
    for row in rows:
        val = as_float(row.get(key))
        values.append(float("nan") if val is None else val)
    return values


def binary_label_column(rows: Sequence[Dict[str, object]], key: str) -> Optional[List[int]]:
    """Column as 0/1 labels (value > 0 -> 1); None as soon as one row has no value."""
    labels: List[int] = []
    for row in rows:
        value = as_float(row.get(key))
        if value is None:
            return None
        labels.append(1 if value > 0 else 0)
    return labels


# --- correlations ---------------------------------------------------------------------------


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """Ranks with ties averaged (1-based)."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Pearson correlation (None if undefined)."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(mask)) < 2:
        return None
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    x_sd = float(np.std(x_arr))
    y_sd = float(np.std(y_arr))
    if x_sd <= 1e-12 or y_sd <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation (None if undefined)."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(mask)) < 2:
        return None
    return pearson(_rankdata(x_arr[mask]), _rankdata(y_arr[mask]))


# --- binary classification ------------------------------------------------------------------


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Area under the ROC curve via the rank-sum statistic (None if only one class)."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata(s)
    sum_pos = float(np.sum(ranks[y == 1]))
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Average precision of ``scores`` for binary ``labels``."""
    pairs = [
        (float(score), int(label))
        for label, score in zip(labels, scores)
        if math.isfinite(float(score))
    ]
    positives = sum(label == 1 for _, label in pairs)
    if positives == 0:
        return None
    pairs.sort(key=lambda item: item[0], reverse=True)
    hit = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            hit += 1
            precision_sum += hit / float(rank)
    return float(precision_sum / positives)


def best_binary_threshold(labels: Sequence[int], scores: Sequence[float]) -> Optional[Dict[str, float]]:
    """Threshold with the best balanced accuracy (ties broken by Youden's J), with sensitivity/specificity."""
    pairs = [
        (int(label), float(score))
        for label, score in zip(labels, scores)
        if math.isfinite(float(score))
    ]
    if not pairs or not any(label == 1 for label, _ in pairs) or not any(label == 0 for label, _ in pairs):
        return None
    thresholds = sorted({score for _, score in pairs}, reverse=True)
    best: Optional[Dict[str, float]] = None
    for threshold in thresholds:
        tp = sum(label == 1 and score >= threshold for label, score in pairs)
        fp = sum(label == 0 and score >= threshold for label, score in pairs)
        fn = sum(label == 1 and score < threshold for label, score in pairs)
        tn = sum(label == 0 and score < threshold for label, score in pairs)
        sensitivity = tp / float(max(1, tp + fn))
        specificity = tn / float(max(1, tn + fp))
        balanced_accuracy = 0.5 * (sensitivity + specificity)
        youden_j = sensitivity + specificity - 1.0
        candidate = {
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "youden_j": float(youden_j),
        }
        if best is None or (candidate["balanced_accuracy"], candidate["youden_j"]) > (
            best["balanced_accuracy"],
            best["youden_j"],
        ):
            best = candidate
    return best
