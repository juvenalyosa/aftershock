"""Stage 3d: directed coupling between protein context and binding pocket.

Transfer entropy (TE) from the context ``P`` to the pocket ``B`` conditioned on
the ligand ``L`` asks: *does the past of the surroundings improve the
prediction of the pocket beyond what the pocket's and ligand's own past
already give?*  Under a linear-Gaussian (VAR) model this is

    TE(P -> B | L) = 1/2 * ln( det(Sigma_restricted) / det(Sigma_full) )

where the restricted model predicts ``B_t`` from the lags of ``B`` and ``L`` and
the full model also uses the lags of ``P`` (Granger causality in nats).

Model order and ridge penalty are selected by blocked cross-validation + BIC
with residual-whiteness and stability checks; significance comes from
phase-randomised surrogates of the driver; the reverse direction is reported
as a control.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from pert.config import TEConfig


# --- VAR building blocks ---------------------------------------------------------


def _lag_matrix(series: np.ndarray, p: int) -> np.ndarray:
    """Design matrix of ``p`` lags: row ``t`` holds ``[x_{t-1}, ..., x_{t-p}]``."""
    t, d = series.shape
    rows = t - p
    blocks = []
    for lag in range(1, p + 1):
        blocks.append(series[p - lag : t - lag, :])
    return np.hstack(blocks).reshape(rows, p * d)


def _fit_ridge_multivariate(x: np.ndarray, y: np.ndarray, ridge: float) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge regression ``y ~ x``; returns ``(coefficients, residuals)``."""
    xtx = x.T @ x
    n_feat = xtx.shape[0]
    beta = np.linalg.solve(xtx + ridge * np.eye(n_feat), x.T @ y)
    resid = y - x @ beta
    return beta, resid


def _residual_cov(resid: np.ndarray) -> np.ndarray:
    """Regularised covariance matrix of regression residuals."""
    cov = np.cov(resid, rowvar=False, bias=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    cov = cov + 1e-10 * np.eye(cov.shape[0])
    return cov


# --- model-selection diagnostics ---------------------------------------------------


def _bic_from_residual_cov(resid: np.ndarray, n_params: int) -> float:
    """BIC of a Gaussian regression model from its residuals."""
    n = resid.shape[0]
    m = resid.shape[1]
    cov = _residual_cov(resid)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        logdet = 1e6
    ll = -0.5 * n * (m * np.log(2.0 * np.pi) + logdet + m)
    bic = -2.0 * ll + n_params * np.log(max(2, n))
    return float(bic)


def _max_abs_resid_autocorr(resid: np.ndarray, lag_max: int) -> float:
    """Largest |autocorrelation| of the residuals up to ``lag_max`` (whiteness check)."""
    resid = resid - np.mean(resid, axis=0, keepdims=True)
    worst = 0.0
    for d in range(resid.shape[1]):
        x = resid[:, d]
        var = np.var(x)
        if var <= 1e-12:
            continue
        for lag in range(1, min(lag_max, len(x) - 1) + 1):
            c = np.dot(x[:-lag], x[lag:]) / (len(x) - lag)
            worst = max(worst, abs(float(c / var)))
    return worst


def _y_stability(beta_full: np.ndarray, y_dim: int, p: int) -> float:
    """Spectral radius of the companion matrix of the target's own lags (< 1 = stable VAR)."""
    if p < 1:
        return 0.0
    companion = np.zeros((y_dim * p, y_dim * p), dtype=np.float64)
    for lag in range(p):
        sl = slice(lag * y_dim, (lag + 1) * y_dim)
        companion[0:y_dim, sl] = beta_full[sl, :].T
    if p > 1:
        companion[y_dim:, :-y_dim] = np.eye(y_dim * (p - 1))
    eigvals = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigvals)))


def _blocked_cv_mse(x: np.ndarray, y: np.ndarray, ridge: float, folds: int = 3) -> float:
    """Cross-validated MSE with contiguous folds (respects time ordering)."""
    n = x.shape[0]
    if n < folds + 5:
        _, resid = _fit_ridge_multivariate(x, y, ridge)
        return float(np.mean(resid * resid))
    splits = np.linspace(0, n, folds + 1, dtype=int)
    mse_vals = []
    for i in range(folds):
        a = splits[i]
        b = splits[i + 1]
        if b - a < 2:
            continue
        mask = np.ones(n, dtype=bool)
        mask[a:b] = False
        x_train = x[mask]
        y_train = y[mask]
        x_valid = x[a:b]
        y_valid = y[a:b]
        beta, _ = _fit_ridge_multivariate(x_train, y_train, ridge)
        resid = y_valid - x_valid @ beta
        mse_vals.append(float(np.mean(resid * resid)))
    if not mse_vals:
        _, resid = _fit_ridge_multivariate(x, y, ridge)
        return float(np.mean(resid * resid))
    return float(np.mean(mse_vals))


# --- surrogates and TE estimators ---------------------------------------------------


def _phase_randomize(ts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate series with the same power spectrum but random Fourier phases.

    Destroys any time-lagged dependence on the target while keeping each
    component's autocorrelation: the null model of the TE significance test.
    """
    n, d = ts.shape
    out = np.zeros_like(ts)
    for dim in range(d):
        x = ts[:, dim]
        x0 = x - np.mean(x)
        fx = np.fft.rfft(x0)
        if fx.shape[0] > 2:
            random_phase = rng.uniform(0.0, 2.0 * np.pi, size=fx.shape[0] - 2)
            fx[1:-1] *= np.exp(1j * random_phase)
        xr = np.fft.irfft(fx, n=n)
        out[:, dim] = xr + np.mean(x)
    return out


def _te_with_fixed_model(
    y: np.ndarray,
    x: np.ndarray,
    w: Optional[np.ndarray],
    p: int,
    ridge: float,
) -> float:
    """Linear-Gaussian TE ``x -> y | w`` for a given VAR order and ridge penalty."""
    y_target = y[p:, :]
    x_parts = [_lag_matrix(y, p)]
    if w is not None and w.shape[1] > 0:
        x_parts.append(_lag_matrix(w, p))
    xr = np.hstack(x_parts)
    xf = np.hstack([xr, _lag_matrix(x, p)])

    _, resid_r = _fit_ridge_multivariate(xr, y_target, ridge)
    _, resid_f = _fit_ridge_multivariate(xf, y_target, ridge)
    cov_r = _residual_cov(resid_r)
    cov_f = _residual_cov(resid_f)
    _, logdet_r = np.linalg.slogdet(cov_r)
    _, logdet_f = np.linalg.slogdet(cov_f)
    return float(0.5 * (logdet_r - logdet_f))


def conditional_te_linear_gaussian(
    y: np.ndarray,
    x: np.ndarray,
    w: Optional[np.ndarray],
    cfg: TEConfig,
) -> Dict[str, object]:
    """Conditional transfer entropy ``x -> y | w`` with model selection and significance.

    ``y`` is the target (pocket), ``x`` the driver (context) and ``w`` the
    conditioning set (ligand).  For each VAR order the ridge penalty is chosen by
    blocked CV; among orders with white residuals and a stable model the lowest
    BIC wins.  Returns the TE, the reverse-direction TE, their difference and a
    surrogate p-value (phase-randomised driver).
    """
    n = min(len(y), len(x), len(w) if w is not None else len(y))
    y = y[-n:]
    x = x[-n:]
    if w is not None:
        w = w[-n:]

    candidates = []
    for p in range(1, cfg.max_order + 1):
        if n <= p + 5:
            continue
        y_target = y[p:, :]
        parts = [_lag_matrix(y, p)]
        if w is not None and w.shape[1] > 0:
            parts.append(_lag_matrix(w, p))
        xr = np.hstack(parts)
        xf = np.hstack([xr, _lag_matrix(x, p)])

        best_ridge = None
        best_cv = float("inf")
        for ridge in cfg.ridge_values:
            cv = _blocked_cv_mse(xf, y_target, ridge=ridge, folds=3)
            if cv < best_cv:
                best_cv = cv
                best_ridge = ridge
        ridge = float(best_ridge if best_ridge is not None else cfg.ridge_values[0])

        beta_r, resid_r = _fit_ridge_multivariate(xr, y_target, ridge)
        beta_f, resid_f = _fit_ridge_multivariate(xf, y_target, ridge)
        cov_r = _residual_cov(resid_r)
        cov_f = _residual_cov(resid_f)
        _, logdet_r = np.linalg.slogdet(cov_r)
        _, logdet_f = np.linalg.slogdet(cov_f)
        te = 0.5 * (logdet_r - logdet_f)

        n_params = beta_f.size
        bic = _bic_from_residual_cov(resid_f, n_params=n_params)
        whiten = _max_abs_resid_autocorr(resid_f, lag_max=cfg.whiteness_lag)
        spectral_radius = _y_stability(beta_f, y_dim=y.shape[1], p=p)
        accepted = (whiten <= cfg.whiteness_threshold) and (spectral_radius < 1.0)
        candidates.append(
            {
                "p": p,
                "ridge": ridge,
                "te": float(te),
                "bic": float(bic),
                "whiteness_max_abs_acf": float(whiten),
                "spectral_radius": float(spectral_radius),
                "accepted": bool(accepted),
            }
        )

    if not candidates:
        raise ValueError("Insufficient data for VAR/TE estimation.")

    accepted = [c for c in candidates if c["accepted"]]
    chosen_pool = accepted if accepted else candidates
    chosen = min(chosen_pool, key=lambda c: c["bic"])

    p_sel = int(chosen["p"])
    ridge_sel = float(chosen["ridge"])
    te_obs = _te_with_fixed_model(y, x, w, p=p_sel, ridge=ridge_sel)
    reverse_obs = _te_with_fixed_model(x, y, w, p=p_sel, ridge=ridge_sel)

    rng = np.random.default_rng(cfg.random_seed)
    surrogate_vals = []
    for _ in range(cfg.surrogates):
        x_surr = _phase_randomize(x, rng)
        te_s = _te_with_fixed_model(y, x_surr, w, p=p_sel, ridge=ridge_sel)
        surrogate_vals.append(float(te_s))
    surrogate_vals_arr = np.asarray(surrogate_vals, dtype=np.float64)
    p_value = float((1 + np.sum(surrogate_vals_arr >= te_obs)) / (cfg.surrogates + 1))

    return {
        "selected_order": p_sel,
        "selected_ridge": ridge_sel,
        "te_driver_to_target": float(te_obs),
        "te_target_to_driver": float(reverse_obs),
        "directionality_delta": float(te_obs - reverse_obs),
        "surrogate_pvalue": p_value,
        "surrogate_te_mean": float(np.mean(surrogate_vals_arr)),
        "surrogate_te_std": float(np.std(surrogate_vals_arr)),
        "model_candidates": candidates,
    }
