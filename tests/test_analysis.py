"""Numerical kernels of the analysis layer on synthetic data with a known answer."""

from __future__ import annotations

import numpy as np
import pytest

from pert.analysis.residue_flow import pairwise_te_matrix
from pert.analysis.transfer_entropy import conditional_te_linear_gaussian
from pert.analysis.vamp import LinearVAMP
from pert.config import TEConfig
from pert.structure.geometry import kabsch_align, kabsch_fit


def _driven_pair(n: int = 800, seed: int = 7):
    """x is autonomous; y is driven by the past of x."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.normal()
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 1] + 0.5 * rng.normal()
    return x, y, rng


def test_conditional_te_detects_direction():
    x, y, rng = _driven_pair()
    w = rng.normal(size=(len(x), 1))  # irrelevant conditioning variable
    result = conditional_te_linear_gaussian(
        y=y[:, None],
        x=x[:, None],
        w=w,
        cfg=TEConfig(max_order=2, surrogates=32, random_seed=3),
    )
    assert result["te_driver_to_target"] > 0.2
    assert result["te_driver_to_target"] > 10 * max(result["te_target_to_driver"], 1e-6)
    assert result["directionality_delta"] > 0
    assert result["surrogate_pvalue"] <= 0.05


def test_pairwise_te_matrix_is_directional():
    x, y, _ = _driven_pair()
    te = pairwise_te_matrix(np.column_stack([x, y]), lag=1)
    assert te.shape == (2, 2)
    assert te[0, 1] > 0.1  # x -> y
    assert te[0, 1] > 10 * te[1, 0]
    assert te[0, 0] == 0.0 and te[1, 1] == 0.0


def test_linear_vamp_recovers_the_slow_coordinate():
    rng = np.random.default_rng(11)
    n = 3000
    slow = np.zeros(n)
    for t in range(1, n):
        slow[t] = 0.995 * slow[t - 1] + 0.1 * rng.normal()
    fast = rng.normal(size=(n, 4))
    mixing = rng.normal(size=(5, 5))
    features = np.column_stack([slow, fast]) @ mixing
    vamp = LinearVAMP(lag=5, n_components=2, reg=1e-6).fit_from_trajectories([features])
    latents = vamp.transform(features)
    assert latents.shape == (n, 2)
    assert abs(np.corrcoef(latents[:, 0], slow)[0, 1]) > 0.95
    assert vamp.svals_[0] > vamp.svals_[1]


def test_kabsch_recovers_a_rigid_transform():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(12, 3))
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    shift = np.array([1.0, -2.0, 0.5])
    moved = (rotation @ points.T).T + shift
    rot, trans, rmsd = kabsch_fit(points, moved)
    assert np.allclose(rot, rotation, atol=1e-8)
    assert np.allclose(trans, shift, atol=1e-8)
    assert rmsd < 1e-8
    rot2, trans2 = kabsch_align(points, moved)
    assert np.allclose(rot2, rot) and np.allclose(trans2, trans)


def _low_rank_slow_data(n: int, n_features: int, seed: int, dead_columns: int = 0) -> np.ndarray:
    """Three slow signals with distinct time scales, mixed into many features (plus dead columns)."""
    rng = np.random.default_rng(seed)
    signals = np.zeros((n, 3))
    for t in range(1, n):
        signals[t] = np.array([0.98, 0.9, 0.6]) * signals[t - 1] + rng.normal(size=3)
    signals /= signals.std(axis=0)
    data = signals @ rng.normal(size=(3, n_features)) + 1e-6 * rng.normal(size=(n, n_features))
    if dead_columns:
        data[:, rng.choice(n_features, size=dead_columns, replace=False)] = 7.0  # constant features
    return data


@pytest.mark.parametrize(
    "n, n_features, dead_columns",
    [
        (120, 600, 250),  # fewer lagged pairs than features: solved in the span of the data
        (400, 90, 30),  # more pairs than features: only the constant columns are dropped
    ],
)
def test_reduced_vamp_solver_matches_the_full_computation(n, n_features, dead_columns):
    data = _low_rank_slow_data(n, n_features, seed=21, dead_columns=dead_columns)
    full = LinearVAMP(lag=3, n_components=3, reg=1e-6, solver="full").fit_from_trajectories([data])
    auto = LinearVAMP(lag=3, n_components=3, reg=1e-6, solver="auto").fit_from_trajectories([data])

    assert np.allclose(full.svals_, auto.svals_, atol=1e-9)
    u_full, u_auto = full.transform(data), auto.transform(data)
    signs = np.sign(np.sum(u_full * u_auto, axis=0))  # components are defined up to a sign
    assert np.allclose(u_full, u_auto * signs, atol=1e-6 * np.abs(u_full).max())

    info = auto.diagnostics()
    assert info["n_active_features"] == n_features - dead_columns
    assert info["working_dimension"] <= min(info["n_active_features"], 2 * info["n_pairs"])
    assert info["effective_dimension"] == 3  # only the three signals rise above the regularisation
    assert info["overfit_risk"] is False


def test_vamp_orientation_is_deterministic():
    data = _low_rank_slow_data(200, 300, seed=4)
    first = LinearVAMP(lag=2, n_components=3).fit_from_trajectories([data])
    flipped = LinearVAMP(lag=2, n_components=3).fit_from_trajectories([data[:, ::-1]])
    # same data with the feature order reversed: identical latents, not sign-flipped ones
    assert np.allclose(first.transform(data), flipped.transform(data[:, ::-1]), atol=1e-6)
    for column in first.proj_.T:
        assert column[np.argmax(np.abs(column))] > 0


def test_vamp_flags_too_few_frames_and_survives_constant_input():
    rng = np.random.default_rng(8)
    noise = rng.normal(size=(40, 300))  # 40 frames, hundreds of independent directions
    info = LinearVAMP(lag=2, n_components=4).fit_from_trajectories([noise]).diagnostics()
    assert info["overfit_risk"] is True
    assert info["n_singular_values_above_0.9"] > 4  # everything looks "slow" in-sample

    constant = np.ones((30, 12))
    model = LinearVAMP(lag=2, n_components=4).fit_from_trajectories([constant])
    assert model.transform(constant).shape == (30, 4)
    assert not model.transform(constant).any()


def test_unknown_vamp_solver_is_rejected():
    with pytest.raises(ValueError):
        LinearVAMP(lag=1, n_components=2, solver="fast")


def test_basin_ddg_plot_accepts_an_interval_that_excludes_the_estimate(tmp_path):
    """A bootstrap interval need not contain the estimate; drawing it must not fail.

    (Seen in a real run: a basin barely populated in one system. Matplotlib
    rejects the negative error-bar length this used to produce.)
    """
    from pert.visualization.run_plots import plot_basin_ddg

    def basin(index, ddg, low, high, p_wt=0.5):
        return {"basin": index, "p_wt": p_wt, "p_mut": 0.5, "ddg_kcal_mol": ddg, "ddg_ci95_low": low, "ddg_ci95_high": high}

    analysis = {
        "basin_ddg": {
            "basins": [
                basin(0, 0.4, 0.1, 0.9),      # ordinary
                basin(1, 1.5, -0.8, 1.2),     # estimate above the interval
                basin(2, -2.0, -1.0, 0.3),    # estimate below the interval
                basin(3, 0.7, None, None),    # no interval
                basin(4, None, None, None),   # populated in one system only
                basin(5, None, None, None, p_wt=0.0),
            ]
        }
    }
    out = tmp_path / "ddg.png"
    plot_basin_ddg(analysis, str(out))
    assert out.stat().st_size > 0
