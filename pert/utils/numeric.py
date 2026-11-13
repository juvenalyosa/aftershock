"""Generic numerical helpers shared by the scoring layers.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    """Robustly scale non-negative scores to [0, 1].

    Values are divided by the 95th percentile of the positive entries (falling
    back to the maximum) and clipped at 1, so a single outlier residue cannot
    flatten everybody else.  Non-finite and negative inputs count as 0.
    """
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    positive = arr[arr > 1e-12]
    if positive.size == 0:
        return np.zeros_like(arr)
    scale = float(np.percentile(positive, 95.0))
    if scale <= 1e-12:
        scale = float(np.max(positive))
    if scale <= 1e-12:
        return np.zeros_like(arr)
    return np.clip(arr / scale, 0.0, 1.0)


def zscore_columns(x: np.ndarray) -> np.ndarray:
    """Z-score each column; constant columns are left centred (std treated as 1)."""
    x = np.asarray(x, dtype=float)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def bootstrap_mean_ci(values: Sequence[float], rng: np.random.Generator, n_boot: int) -> Tuple[float, float]:
    """95% bootstrap confidence interval of the mean of ``values`` (NaNs ignored)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_boot <= 0:
        value = float(arr[0])
        return value, value
    means = np.zeros(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        means[i] = float(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
