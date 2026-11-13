"""Stage 3e: static linear response of the pocket to its surroundings.

``Gamma = Cov(u_B, u_P) Cov(u_P)^-1`` is the regression matrix that maps a
displacement of the context latents onto the expected displacement of the
pocket latents.  The change ``||Gamma_mut - Gamma_wt||_F`` measures how much
the mutation rewires that coupling.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from pert.analysis.statistics import bootstrap_labels, iat_from_series


def susceptibility_gamma(
    u_b: np.ndarray,
    u_p: np.ndarray,
    n_bootstrap: int = 300,
    random_seed: int = 13,
) -> Dict[str, object]:
    """Linear response matrix ``Gamma = Cov(u_B, u_P) Cov(u_P)^-1`` with block-bootstrap intervals."""
    n = min(len(u_b), len(u_p))
    ub = u_b[-n:]
    up = u_p[-n:]
    cov_b = np.cov(ub, rowvar=False)
    cov_p = np.cov(up, rowvar=False) + 1e-8 * np.eye(up.shape[1])
    cov_bp = np.cov(np.hstack([ub, up]).T)[: ub.shape[1], ub.shape[1] :]
    gamma = cov_bp @ np.linalg.inv(cov_p)
    gamma_norm = float(np.linalg.norm(gamma, ord="fro"))
    gamma_sv = np.linalg.svd(gamma, compute_uv=False)
    gamma_top_sv = float(gamma_sv[0] if len(gamma_sv) else 0.0)

    tau = iat_from_series(ub[:, 0])
    block_len = max(1, int(math.ceil(5.0 * tau)))
    rng = np.random.default_rng(random_seed)
    norms = []
    svs = []
    for _ in range(n_bootstrap):
        idx = bootstrap_labels(np.arange(n), block_len, rng).astype(int)
        ub_b = ub[idx]
        up_b = up[idx]
        cov_p_b = np.cov(up_b, rowvar=False) + 1e-8 * np.eye(up_b.shape[1])
        cov_bp_b = np.cov(np.hstack([ub_b, up_b]).T)[: ub.shape[1], ub.shape[1] :]
        gamma_b = cov_bp_b @ np.linalg.inv(cov_p_b)
        norms.append(float(np.linalg.norm(gamma_b, ord="fro")))
        sv = np.linalg.svd(gamma_b, compute_uv=False)
        svs.append(float(sv[0] if len(sv) else 0.0))

    return {
        "gamma_matrix": gamma.tolist(),
        "fro_norm": gamma_norm,
        "top_singular_value": gamma_top_sv,
        "fro_norm_ci95_low": float(np.percentile(norms, 2.5)),
        "fro_norm_ci95_high": float(np.percentile(norms, 97.5)),
        "top_singular_ci95_low": float(np.percentile(svs, 2.5)),
        "top_singular_ci95_high": float(np.percentile(svs, 97.5)),
    }
