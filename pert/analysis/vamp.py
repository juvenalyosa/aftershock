"""Stage 3b: slow collective coordinates (linear VAMP / TICA-like projection).

The field features are high-dimensional and mostly noise.  ``LinearVAMP``
keeps the directions that are most predictable after a lag time, i.e. the slow
motions of each region.  One model per region (B, P, L) is fitted on the WT and
mutant trajectories *together*, so both ensembles are projected onto the same
axes and can be compared directly.

Cost.  The textbook computation diagonalises ``n_features x n_features``
matrices, which is slow for the toy grid (2313 features) and impossible for
the default one (~18 000).  The default solver does the same computation in a
much smaller space, without approximation:

1. features that never change (empty element channels, empty grid cells) are
   dropped: they cannot contribute to any time-lagged correlation;
2. when there are fewer lagged pairs than features, the problem is solved in
   the subspace spanned by the data (at most ``2 * n_pairs`` dimensions).

``solver="full"`` keeps the original code path, bit for bit.

Over-fitting.  With few frames almost any direction looks "slow" in-sample.
``diagnostics()`` reports the singular values, how many of them exceed 0.9 and
the number of lagged pairs per effective dimension; ``overfit_risk`` is set
when there are fewer than ``MIN_PAIRS_PER_DIMENSION`` of the latter.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Fewer lagged pairs than this per effective dimension: the slow directions
# found are largely an artefact of the small sample.
MIN_PAIRS_PER_DIMENSION = 5.0

SOLVERS = ("auto", "full")


def _invsqrt_and_eigenvalues(matrix: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse square root of a symmetric PSD matrix and its (unclipped) eigenvalues."""
    raw_vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(raw_vals, eps, None)
    invsqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return invsqrt, raw_vals


def _symmetric_invsqrt(matrix: np.ndarray, eps: float) -> np.ndarray:
    """Inverse square root of a symmetric PSD matrix (eigenvalues floored at ``eps``)."""
    return _invsqrt_and_eigenvalues(matrix, eps)[0]


class LinearVAMP:
    """Linear VAMP: project features onto their slowest decorrelating directions.

    Covariances ``C00``, ``C11`` and the time-lagged ``C01`` are estimated from
    pairs ``(x_t, x_{t+lag})``; the leading singular vectors of the whitened
    Koopman matrix ``C00^-1/2 C01 C11^-1/2`` give the projection.  ``transform``
    returns the latent coordinates ``u``.

    ``solver="auto"`` (default) solves the problem in the smallest exact
    subspace and fixes the sign of every component (largest loading positive),
    so results do not depend on the linear-algebra library.  ``solver="full"``
    is the original full-dimensional computation.
    """
    def __init__(self, lag: int, n_components: int, reg: float = 1e-6, solver: str = "auto"):
        if solver not in SOLVERS:
            raise ValueError(f"Unknown VAMP solver '{solver}'. Use one of {SOLVERS}.")
        self.lag = int(lag)
        self.n_components = int(n_components)
        self.reg = float(reg)
        self.solver = solver
        self.mean_: Optional[np.ndarray] = None
        self.proj_: Optional[np.ndarray] = None
        self.svals_: Optional[np.ndarray] = None
        self.diagnostics_: Dict[str, object] = {}

    def fit_from_trajectories(self, trajectories: Sequence[np.ndarray]) -> "LinearVAMP":
        x0_list: List[np.ndarray] = []
        x1_list: List[np.ndarray] = []
        for x in trajectories:
            if x.shape[0] <= self.lag:
                continue
            x0_list.append(x[:-self.lag])
            x1_list.append(x[self.lag :])
        if not x0_list:
            raise ValueError("Not enough trajectory frames for requested VAMP lag.")
        x0 = np.vstack(x0_list)
        x1 = np.vstack(x1_list)
        return self.fit_pairs(x0, x1)

    def _solve(self, xc0: np.ndarray, xc1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Whitened-Koopman SVD on centred pairs.

        Returns ``(projection, all singular values, variances)`` where the
        variances are the eigenvalues of the instantaneous covariance.
        """
        n_samples, n_feat = xc0.shape
        c00 = (xc0.T @ xc0) / max(1, n_samples - 1) + self.reg * np.eye(n_feat)
        c11 = (xc1.T @ xc1) / max(1, n_samples - 1) + self.reg * np.eye(n_feat)
        c01 = (xc0.T @ xc1) / max(1, n_samples - 1)

        w0, vals00 = _invsqrt_and_eigenvalues(c00, self.reg)
        w1 = _symmetric_invsqrt(c11, self.reg)
        k = w0 @ c01 @ w1
        u, s, _ = np.linalg.svd(k, full_matrices=False)

        n_keep = int(min(self.n_components, u.shape[1]))
        return w0 @ u[:, :n_keep], s, vals00 - self.reg

    def fit_pairs(self, x0: np.ndarray, x1: np.ndarray) -> "LinearVAMP":
        if x0.shape != x1.shape:
            raise ValueError("x0/x1 must have equal shape.")
        n_samples, n_feat = x0.shape
        if n_samples < 3:
            raise ValueError("At least 3 lagged pairs are required for VAMP.")

        self.mean_ = x0.mean(axis=0)
        xc0 = x0 - self.mean_[None, :]
        xc1 = x1 - self.mean_[None, :]

        if self.solver == "full":
            proj, svals, variances = self._solve(xc0, xc1)
            n_active = working_dim = n_feat
        else:
            # 1. Features that never change carry no time-lagged correlation.
            active = np.flatnonzero(np.any(xc0 != 0.0, axis=0) | np.any(xc1 != 0.0, axis=0))
            a0, a1 = xc0[:, active], xc1[:, active]
            n_active = int(active.size)
            if n_active == 0:
                proj_active, svals, variances = np.zeros((0, 0)), np.zeros(0), np.zeros(0)
                working_dim = 0
            elif 2 * n_samples < n_active:
                # 2. Fewer pairs than features: work in the span of the data.
                stacked = np.vstack([a0, a1])
                _, sing, vt = np.linalg.svd(stacked, full_matrices=False)
                tol = sing[0] * max(stacked.shape) * np.finfo(np.float64).eps
                basis = vt[: max(1, int(np.sum(sing > tol)))].T
                reduced, svals, variances = self._solve(a0 @ basis, a1 @ basis)
                proj_active = basis @ reduced
                working_dim = int(basis.shape[1])
            else:
                proj_active, svals, variances = self._solve(a0, a1)
                working_dim = n_active

            # Always return n_components columns (extra ones are empty directions).
            n_keep = int(min(self.n_components, n_feat))
            proj = np.zeros((n_feat, n_keep), dtype=np.float64)
            if n_active:
                proj[active, : proj_active.shape[1]] = proj_active[:, :n_keep]
            # Deterministic orientation: the largest loading of each component is positive.
            for j in range(n_keep):
                pivot = int(np.argmax(np.abs(proj[:, j])))
                if proj[pivot, j] < 0.0:
                    proj[:, j] *= -1.0

        n_keep = proj.shape[1]
        kept = np.zeros(n_keep, dtype=np.float64)
        kept[: min(n_keep, svals.size)] = svals[:n_keep]
        self.proj_ = proj
        self.svals_ = kept

        # Sampling diagnostics: directions whose variance exceeds the regularisation
        # are the ones the model can actually use.
        effective_dim = int(np.sum(variances > self.reg))
        pairs_per_dim = float(n_samples / effective_dim) if effective_dim else float("inf")
        self.diagnostics_ = {
            "solver": self.solver,
            "n_pairs": int(n_samples),
            "n_features": int(n_feat),
            "n_active_features": int(n_active),
            "working_dimension": int(working_dim),
            "effective_dimension": effective_dim,
            "pairs_per_effective_dimension": None if not np.isfinite(pairs_per_dim) else pairs_per_dim,
            "singular_values": [float(v) for v in kept],
            "n_singular_values_above_0.9": int(np.sum(svals > 0.9)),
            "overfit_risk": bool(pairs_per_dim < MIN_PAIRS_PER_DIMENSION),
        }
        return self

    def diagnostics(self) -> Dict[str, object]:
        """Solver, problem sizes, kept singular values and the over-fitting indicators."""
        return dict(self.diagnostics_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.proj_ is None:
            raise RuntimeError("LinearVAMP must be fit before transform.")
        return (x - self.mean_[None, :]) @ self.proj_
