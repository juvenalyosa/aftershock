"""Stage 3c: how the mutation changes the conformational ensemble of the site.

Works on the pocket latents ``u_B`` of WT and mutant:

* **basin populations** - a Gaussian mixture (size chosen by BIC) defines
  conformational basins.  The headline number is the total-variation distance
  between the WT and mutant basin populations (0 = same, 1 = disjoint).  For
  basins visited by *both* systems, ``ddG = -kT ln(p_mut / p_wt)`` is reported
  with block-bootstrap intervals; a basin seen in only one system has no
  defined free-energy difference and is flagged ``overlap = False`` instead of
  being given an arbitrary large value.
* **distribution shift** - a classifier-based Jensen-Shannon divergence and
  the classifier AUC, both evaluated on held-out *time blocks* so that they do
  not saturate by over-fitting; plus Gaussian KL (or GMM Monte-Carlo KL when
  the ensembles are not Gaussian).

Sampling honesty: a trajectory that never changes basin carries a single
effective sample of the basin populations, whatever its number of frames.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from pert.analysis.statistics import bootstrap_labels, iat_from_series
from pert.deps import ensure_sklearn, GaussianMixture, LogisticRegression, scipy_stats


def fit_bic_gmm(data: np.ndarray, max_components: int, random_seed: int = 13) -> GaussianMixture:
    """Gaussian mixture with the number of components (1..max) chosen by BIC."""
    ensure_sklearn()
    max_components = max(1, min(max_components, data.shape[0] - 1))
    best_model = None
    best_bic = float("inf")
    for k in range(1, max_components + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_seed,
        )
        model.fit(data)
        bic = model.bic(data)
        if bic < best_bic:
            best_bic = bic
            best_model = model
    if best_model is None:
        raise RuntimeError("Could not fit GaussianMixture model.")
    return best_model


# --- basin populations -> ddG ---------------------------------------------------

# Below this many effective samples in either system the basin populations (and
# anything derived from them) are flagged as not converged.
MIN_EFFECTIVE_SAMPLES = 10.0


def _label_autocorrelation_time(labels: np.ndarray, n_basins: int) -> Tuple[float, int]:
    """Autocorrelation time (frames) of a basin-label series and its number of transitions.

    A series that never leaves its basin has no observable decorrelation: its
    autocorrelation time is the whole trajectory (one effective sample).
    """
    transitions = int(np.count_nonzero(labels[1:] != labels[:-1]))
    if transitions == 0:
        return float(len(labels)), 0
    tau = max(iat_from_series((labels == basin).astype(float)) for basin in range(n_basins))
    return float(min(tau, len(labels))), transitions


def _basin_ddg(counts_wt: np.ndarray, counts_mut: np.ndarray, n_wt: int, n_mut: int, kbt: float) -> np.ndarray:
    """``-kT ln(p_mut / p_wt)`` per basin; NaN where a basin is empty in either system."""
    overlap = (counts_wt > 0) & (counts_mut > 0)
    ddg = np.full(counts_wt.shape, np.nan, dtype=np.float64)
    ddg[overlap] = -kbt * np.log((counts_mut[overlap] / n_mut) / (counts_wt[overlap] / n_wt))
    return ddg


def compute_basin_ddg(
    u_wt: np.ndarray,
    u_mut: np.ndarray,
    temperature_k: float,
    n_bootstrap: int = 400,
    random_seed: int = 13,
) -> Dict[str, object]:
    """Population shift of conformational basins of the pocket.

    A GMM fitted on WT+mutant latents defines the basins.  Returns

    * ``population_tv_distance`` (with a 95% block-bootstrap interval): half the
      summed absolute difference of the basin populations;
    * per basin: populations, counts, ``overlap`` and, only when the basin is
      populated in both systems, ``ddg_kcal_mol`` with its interval (``None``
      otherwise);
    * sampling diagnostics: transitions, autocorrelation time and effective
      samples of each system, and ``sampling_ok``.
    """
    gmm = fit_bic_gmm(np.vstack([u_wt, u_mut]), max_components=8, random_seed=random_seed)
    labels_wt = gmm.predict(u_wt)
    labels_mut = gmm.predict(u_mut)
    n_basins = int(gmm.n_components)
    n_wt = len(labels_wt)
    n_mut = len(labels_mut)
    kbt = 0.0019872041 * temperature_k

    counts_wt = np.bincount(labels_wt, minlength=n_basins).astype(float)
    counts_mut = np.bincount(labels_mut, minlength=n_basins).astype(float)
    p_wt = counts_wt / n_wt
    p_mut = counts_mut / n_mut
    overlap = (counts_wt > 0) & (counts_mut > 0)
    ddg = _basin_ddg(counts_wt, counts_mut, n_wt, n_mut, kbt)
    tv_distance = 0.5 * float(np.sum(np.abs(p_mut - p_wt)))

    if n_basins == 1:
        # One basin: there are no populations to estimate.  Report the sampling of
        # the slowest pocket coordinate instead so the diagnostics stay meaningful.
        tau_wt, transitions_wt = float(min(iat_from_series(u_wt[:, 0]), n_wt)), 0
        tau_mut, transitions_mut = float(min(iat_from_series(u_mut[:, 0]), n_mut)), 0
    else:
        tau_wt, transitions_wt = _label_autocorrelation_time(labels_wt, n_basins)
        tau_mut, transitions_mut = _label_autocorrelation_time(labels_mut, n_basins)
    tau = max(tau_wt, tau_mut)
    block_len = max(1, int(math.ceil(5.0 * tau)))
    n_eff_wt = float(n_wt / tau_wt)
    n_eff_mut = float(n_mut / tau_mut)
    sampling_ok = bool(min(n_eff_wt, n_eff_mut) >= MIN_EFFECTIVE_SAMPLES)

    rng = np.random.default_rng(random_seed)
    ddg_boot = np.full((n_bootstrap, n_basins), np.nan, dtype=np.float64)
    tv_boot = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        boot_wt = bootstrap_labels(labels_wt, block_len, rng)
        boot_mut = bootstrap_labels(labels_mut, block_len, rng)
        c_wt = np.bincount(boot_wt, minlength=n_basins).astype(float)
        c_mut = np.bincount(boot_mut, minlength=n_basins).astype(float)
        ddg_boot[b] = _basin_ddg(c_wt, c_mut, len(boot_wt), len(boot_mut), kbt)
        tv_boot[b] = 0.5 * np.sum(np.abs(c_mut / len(boot_mut) - c_wt / len(boot_wt)))

    basin_rows = []
    for basin in range(n_basins):
        # An interval is only meaningful if the basin stays populated in both
        # systems in most resamples.
        finite = np.isfinite(ddg_boot[:, basin])
        support = float(np.mean(finite)) if n_bootstrap > 0 else 0.0
        ci_low: Optional[float] = None
        ci_high: Optional[float] = None
        if overlap[basin] and support >= 0.5:
            ci_low = float(np.percentile(ddg_boot[finite, basin], 2.5))
            ci_high = float(np.percentile(ddg_boot[finite, basin], 97.5))
        basin_rows.append(
            {
                "basin": basin,
                "p_wt": float(p_wt[basin]),
                "p_mut": float(p_mut[basin]),
                "count_wt": int(counts_wt[basin]),
                "count_mut": int(counts_mut[basin]),
                "overlap": bool(overlap[basin]),
                "ddg_kcal_mol": float(ddg[basin]) + 0.0 if overlap[basin] else None,
                "ddg_ci95_low": ci_low,
                "ddg_ci95_high": ci_high,
                "ddg_bootstrap_support": support,
            }
        )

    if n_basins == 1:
        sampling_note = (
            "Single basin: WT and mutant occupy the same conformational basin, so basin "
            "populations carry no information (effective samples refer to the slowest pocket coordinate)."
        )
    elif sampling_ok:
        sampling_note = "Basin populations are sampled by both systems."
    else:
        sampling_note = (
            f"Not converged: effective samples WT={n_eff_wt:.1f}, MUT={n_eff_mut:.1f} "
            f"(< {MIN_EFFECTIVE_SAMPLES:.0f}); basin transitions WT={transitions_wt}, MUT={transitions_mut}. "
            "Population differences reflect where each trajectory happened to sit."
        )

    return {
        "n_basins": n_basins,
        "n_overlapping_basins": int(np.sum(overlap)),
        "population_tv_distance": tv_distance,
        "population_tv_ci95_low": float(np.percentile(tv_boot, 2.5)) if n_bootstrap > 0 else None,
        "population_tv_ci95_high": float(np.percentile(tv_boot, 97.5)) if n_bootstrap > 0 else None,
        "transitions_wt": transitions_wt,
        "transitions_mut": transitions_mut,
        "iat_wt": tau_wt,
        "iat_mut": tau_mut,
        "effective_samples_wt": n_eff_wt,
        "effective_samples_mut": n_eff_mut,
        "sampling_ok": sampling_ok,
        "sampling_note": sampling_note,
        "iat": tau,
        "block_len": block_len,
        "basins": basin_rows,
    }


# --- distribution shift ----------------------------------------------------------


def gaussianity_diagnostics(data: np.ndarray, random_seed: int = 13) -> Dict[str, object]:
    """Is a single Gaussian an acceptable model of ``data``? (BIC of 1 vs 2 components + normality test)."""
    ensure_sklearn()
    gmm1 = fit_bic_gmm(data, max_components=1, random_seed=random_seed)
    gmm2 = fit_bic_gmm(data, max_components=2, random_seed=random_seed)
    bic1 = float(gmm1.bic(data))
    bic2 = float(gmm2.bic(data))
    unimodal_pref = bic1 <= bic2 + 2.0

    min_p = None
    if scipy_stats is not None and data.shape[0] >= 20:
        pvals = []
        for d in range(data.shape[1]):
            try:
                stat, p = scipy_stats.normaltest(data[:, d])
                pvals.append(float(p))
            except Exception:
                continue
        if pvals:
            min_p = float(min(pvals))
    passed = unimodal_pref and (min_p is None or min_p > 1e-2)
    return {
        "bic_1_component": bic1,
        "bic_2_component": bic2,
        "unimodal_preferred": unimodal_pref,
        "normality_min_pvalue": min_p,
        "passed_for_gaussian_kl": passed,
    }


def gaussian_kl_divergence(
    mean0: np.ndarray,
    cov0: np.ndarray,
    mean1: np.ndarray,
    cov1: np.ndarray,
    reg: float = 1e-8,
) -> float:
    """Closed-form KL(N0 || N1) between two multivariate Gaussians."""
    dim = mean0.shape[0]
    cov0r = cov0 + reg * np.eye(dim)
    cov1r = cov1 + reg * np.eye(dim)
    inv1 = np.linalg.inv(cov1r)
    diff = (mean1 - mean0)[:, None]
    sign0, logdet0 = np.linalg.slogdet(cov0r)
    sign1, logdet1 = np.linalg.slogdet(cov1r)
    if sign0 <= 0 or sign1 <= 0:
        raise ValueError("Covariance determinant is non-positive.")
    term = (
        np.trace(inv1 @ cov0r)
        + float((diff.T @ inv1 @ diff).reshape(()))
        - dim
        + (logdet1 - logdet0)
    )
    return 0.5 * float(term)


def gmm_kl_monte_carlo(
    data0: np.ndarray,
    data1: np.ndarray,
    samples: int = 10000,
    random_seed: int = 13,
) -> float:
    """KL(data0 || data1) estimated by sampling from GMM fits of both ensembles."""
    ensure_sklearn()
    g0 = fit_bic_gmm(data0, max_components=6, random_seed=random_seed)
    g1 = fit_bic_gmm(data1, max_components=6, random_seed=random_seed)
    x, _ = g0.sample(samples)
    log_p0 = g0.score_samples(x)
    log_p1 = g1.score_samples(x)
    return float(np.mean(log_p0 - log_p1))


def js_divergence_density_ratio(
    data0: np.ndarray,
    data1: np.ndarray,
    random_seed: int = 13,
) -> float:
    """In-sample Jensen-Shannon estimate from a logistic-regression density ratio.

    The classifier is scored on its own training data, so the value is biased
    upwards and saturates near ln 2 for short trajectories.  Kept for
    comparison with old results; use ``classifier_separability`` instead.
    """
    ensure_sklearn()
    x = np.vstack([data0, data1])
    y = np.concatenate([np.zeros(data0.shape[0]), np.ones(data1.shape[0])], axis=0)
    clf = LogisticRegression(max_iter=2000, random_state=random_seed)
    clf.fit(x, y)
    probs = np.clip(clf.predict_proba(x)[:, 1], 1e-6, 1.0 - 1e-6)
    ratio = probs / (1.0 - probs)
    r0 = ratio[: data0.shape[0]]
    r1 = ratio[data0.shape[0] :]

    js = 0.5 * (
        np.mean(np.log(2.0 / (1.0 + r0))) + np.mean(np.log(2.0 * r1 / (1.0 + r1)))
    )
    return float(np.clip(js, 0.0, np.log(2.0)))


def _auc(scores_negative: np.ndarray, scores_positive: np.ndarray) -> float:
    """Probability that a positive sample outscores a negative one (ties count 1/2)."""
    ordered = np.sort(scores_negative)
    below = np.searchsorted(ordered, scores_positive, side="left")
    below_or_equal = np.searchsorted(ordered, scores_positive, side="right")
    return float(np.mean(0.5 * (below + below_or_equal)) / len(ordered))


def classifier_separability(
    data0: np.ndarray,
    data1: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 13,
) -> Dict[str, object]:
    """How well can a classifier tell the two ensembles apart on unseen frames?

    Each trajectory is cut into ``n_folds`` contiguous time blocks; a logistic
    regression is trained on all blocks but one and scored on the block left
    out.  From the held-out predictions:

    * ``classifier_auc``: 0.5 = indistinguishable, 1.0 = perfectly separable;
    * ``js_divergence``: Jensen-Shannon divergence from the held-out density
      ratio (0 = same distribution, ln 2 = disjoint).

    Contiguous blocks matter: random splits would put neighbouring (almost
    identical) frames on both sides and reward memorisation.  Two trajectories
    that have not mixed still look separable; only the WT-vs-WT null control
    tells how much separability is expected without any mutation.
    """
    ensure_sklearn()
    n0, n1 = int(data0.shape[0]), int(data1.shape[0])
    folds = int(min(n_folds, n0 // 2, n1 // 2))
    if folds < 2:
        return {
            "js_divergence": js_divergence_density_ratio(data0, data1, random_seed=random_seed),
            "classifier_auc": None,
            "cv_folds": 0,
            "estimator": "in_sample_logistic (too few frames for cross-validation)",
        }

    blocks0 = np.array_split(np.arange(n0), folds)
    blocks1 = np.array_split(np.arange(n1), folds)
    held_p0 = np.zeros(n0, dtype=np.float64)
    held_p1 = np.zeros(n1, dtype=np.float64)
    held_r0 = np.zeros(n0, dtype=np.float64)
    held_r1 = np.zeros(n1, dtype=np.float64)
    for fold in range(folds):
        test0, test1 = blocks0[fold], blocks1[fold]
        train0 = np.setdiff1d(np.arange(n0), test0)
        train1 = np.setdiff1d(np.arange(n1), test1)
        x = np.vstack([data0[train0], data1[train1]])
        y = np.concatenate([np.zeros(len(train0)), np.ones(len(train1))])
        clf = LogisticRegression(max_iter=2000, random_state=random_seed)
        clf.fit(x, y)
        prior_odds = len(train0) / float(len(train1))  # undo class imbalance of the training set
        for test, data, held_p, held_r in ((test0, data0, held_p0, held_r0), (test1, data1, held_p1, held_r1)):
            probs = np.clip(clf.predict_proba(data[test])[:, 1], 1e-6, 1.0 - 1e-6)
            held_p[test] = probs
            held_r[test] = probs / (1.0 - probs) * prior_odds

    js = 0.5 * (np.mean(np.log(2.0 / (1.0 + held_r0))) + np.mean(np.log(2.0 * held_r1 / (1.0 + held_r1))))
    return {
        "js_divergence": float(np.clip(js, 0.0, np.log(2.0))),
        "classifier_auc": _auc(held_p0, held_p1),
        "cv_folds": folds,
        "estimator": "time_blocked_cv_logistic",
    }


def distribution_shift(u_wt: np.ndarray, u_mut: np.ndarray) -> Dict[str, object]:
    """Distance between the WT and mutant pocket ensembles.

    Always reports the cross-validated ``js_divergence`` and ``classifier_auc``
    (see ``classifier_separability``) and, for reference, the legacy in-sample
    estimate ``js_divergence_in_sample``.  The KL divergence is Gaussian
    (``kl_wt_to_mut``) when both ensembles pass the Gaussianity checks and a
    GMM Monte-Carlo estimate (``gmm_kl_wt_to_mut``) otherwise; ``method`` says
    which one was used.
    """
    diag_wt = gaussianity_diagnostics(u_wt)
    diag_mut = gaussianity_diagnostics(u_mut)
    gaussian_ok = bool(diag_wt["passed_for_gaussian_kl"] and diag_mut["passed_for_gaussian_kl"])
    separability = classifier_separability(u_wt, u_mut)

    result: Dict[str, object] = {
        "method": "gaussian_kl" if gaussian_ok else "fallback_gmm_kl_js",
        "gaussian_diagnostics_wt": diag_wt,
        "gaussian_diagnostics_mut": diag_mut,
    }
    if gaussian_ok:
        mean0 = np.mean(u_wt, axis=0)
        mean1 = np.mean(u_mut, axis=0)
        cov0 = np.cov(u_wt, rowvar=False)
        cov1 = np.cov(u_mut, rowvar=False)
        result["kl_wt_to_mut"] = float(gaussian_kl_divergence(mean0, cov0, mean1, cov1))
    else:
        result["gmm_kl_wt_to_mut"] = float(gmm_kl_monte_carlo(u_wt, u_mut))

    result["js_divergence"] = separability["js_divergence"]
    result["classifier_auc"] = separability["classifier_auc"]
    result["js_cv_folds"] = separability["cv_folds"]
    result["js_estimator"] = separability["estimator"]
    result["js_divergence_in_sample"] = js_divergence_density_ratio(u_wt, u_mut)
    return result
