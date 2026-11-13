"""Time-series statistics for correlated MD samples.

Frames of a trajectory are not independent; uncertainty is therefore estimated
with a *block* bootstrap whose block length is tied to the integrated
autocorrelation time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def iat_from_series(x: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Integrated autocorrelation time (in frames) of a scalar series; >= 1."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    var = np.var(x)
    if var <= 1e-12:
        return 1.0
    n = x.shape[0]
    if max_lag is None:
        max_lag = min(n // 2, 2000)
    tau = 1.0
    for lag in range(1, max_lag + 1):
        c = np.dot(x[:-lag], x[lag:]) / (n - lag)
        rho = c / var
        if rho <= 0:
            break
        tau += 2.0 * rho
    return max(1.0, tau)


def bootstrap_labels(labels: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap resample of a series (plain bootstrap when ``block_len <= 1``)."""
    n = labels.shape[0]
    if block_len <= 1:
        return labels[rng.integers(0, n, size=n)]
    out = np.empty(n, dtype=labels.dtype)
    write = 0
    while write < n:
        start = int(rng.integers(0, max(1, n - block_len + 1)))
        end = min(start + block_len, n)
        span = end - start
        span = min(span, n - write)
        out[write : write + span] = labels[start : start + span]
        write += span
    return out
