"""The comparison must be blind to things that are not a mutation.

* rotating/translating a trajectory changes nothing (shared site frame);
* two simulations of the same protein are (nearly) indistinguishable, while a
  displaced site is not, and the null control tells them apart;
* disjoint or unsampled ensembles are reported as such instead of as numbers.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("openmm")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from pert.analysis.ensemble import classifier_separability, compute_basin_ddg  # noqa: E402
from pert.analysis.pocket_fields import compute_region_features  # noqa: E402
from pert.analysis.site_frame import build_shared_site_frames  # noqa: E402
from pert.analysis.topology_safe import comparison_metrics, run_topology_safe_analysis  # noqa: E402
from pert.config import FieldConfig, LatentConfig, TEConfig  # noqa: E402
from pert.deps import app  # noqa: E402
from pert.pipelines.null_control import calibrate_against_null, run_null_control  # noqa: E402

from synthetic import (  # noqa: E402
    make_trajectory,
    random_rotation,
    reference_positions,
    save_trajectory,
    write_run,
    write_shifted_pdb,
)

FIELD = dict(rb_ang=6.0, rp_ang=12.0, grid_spacing_ang=3.0, sigma_ang=3.0, max_grid_points=48)
LATENT = LatentConfig(lag=2, m_b=4, m_p=4, m_l=2, reg=1e-6)
TE = TEConfig(max_order=1, surrogates=8, random_seed=5)


def _features(pdb, positions, velocities, frame, **field_overrides):
    return compute_region_features(
        topology=app.PDBFile(pdb).topology,
        positions=positions,
        velocities=velocities,
        dt_ps=1.0,
        ligand_resnames=["ZN"],
        field_cfg=FieldConfig(**{**FIELD, **field_overrides}),
        site_frame=frame,
    )


def _analyse(tmp_path, tag, wt_pdb, wt_npz, mut_pdb, mut_npz, **field_overrides):
    return run_topology_safe_analysis(
        wt_pdb=wt_pdb,
        wt_traj_npz=wt_npz,
        mut_pdb=mut_pdb,
        mut_traj_npz=mut_npz,
        ligand_resnames=["ZN"],
        temperature_k=300.0,
        field_cfg=FieldConfig(**{**FIELD, **field_overrides}),
        latent_cfg=LATENT,
        te_cfg=TE,
        output_dir=str(tmp_path / tag),
    )


# --- shared site frame -----------------------------------------------------------------------


def test_site_frame_is_anchored_on_the_zinc_site(toy_reference_pdb):
    wt_frame, mut_frame, info = build_shared_site_frames(toy_reference_pdb, toy_reference_pdb, ["ZN"])
    assert info["anchor_count"] >= 6
    # the four zinc ligands of the finger are among the anchors
    assert {"CYS3", "CYS6", "HIS19", "HIS23"} <= set(info["anchor_residues_wt"])
    assert np.array_equal(wt_frame.reference_coords_nm, mut_frame.reference_coords_nm)


def test_fields_do_not_change_under_a_rigid_motion(toy_reference_pdb):
    reference = reference_positions(toy_reference_pdb)
    positions, velocities = make_trajectory(reference, seed=1, n_frames=12)
    rotation, shift = random_rotation(3), np.array([1.5, -0.7, 2.0])
    frame, _, _ = build_shared_site_frames(toy_reference_pdb, toy_reference_pdb, ["ZN"])

    original = _features(toy_reference_pdb, positions, velocities, frame)
    moved = _features(toy_reference_pdb, positions @ rotation.T + shift, velocities @ rotation.T, frame)
    for region in ("B", "P", "L"):
        assert np.allclose(original[region], moved[region], atol=1e-9), region
    assert np.abs(original["B"]).max() > 0 and np.abs(original["L"]).max() > 0


def test_fields_do_not_change_when_the_protein_tumbles(toy_reference_pdb):
    reference = reference_positions(toy_reference_pdb)
    positions, velocities = make_trajectory(reference, seed=2, n_frames=10)
    tumbling = np.stack([positions[t] @ random_rotation(100 + t).T for t in range(len(positions))])
    frame, _, _ = build_shared_site_frames(toy_reference_pdb, toy_reference_pdb, ["ZN"])

    original = _features(toy_reference_pdb, positions, velocities, frame, include_velocity=False)
    moved = _features(toy_reference_pdb, tumbling, velocities, frame, include_velocity=False)
    for region in ("B", "P", "L"):
        assert np.allclose(original[region], moved[region], atol=1e-9), region


def test_legacy_frame_depends_on_orientation(toy_reference_pdb):
    """Documents the defect that the shared frame fixes."""
    reference = reference_positions(toy_reference_pdb)
    positions, velocities = make_trajectory(reference, seed=1, n_frames=6)
    rotation = random_rotation(3)
    original = _features(toy_reference_pdb, positions, velocities, None)
    moved = _features(toy_reference_pdb, positions @ rotation.T, velocities @ rotation.T, None)
    assert not np.allclose(original["B"], moved["B"], atol=1e-3)


def test_a_rotated_copy_is_not_a_different_ensemble(tmp_path, toy_reference_pdb):
    reference = reference_positions(toy_reference_pdb)
    positions, velocities = make_trajectory(reference, seed=4, n_frames=120)
    rotation = random_rotation(9)
    wt_npz = save_trajectory(str(tmp_path / "wt.npz"), positions, velocities)
    rot_npz = save_trajectory(str(tmp_path / "rot.npz"), positions @ rotation.T + 3.0, velocities @ rotation.T)

    metrics = comparison_metrics(_analyse(tmp_path, "rot", toy_reference_pdb, wt_npz, toy_reference_pdb, rot_npz))
    assert metrics["js_divergence"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["classifier_auc"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["basin_population_tv"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["abs_te_delta"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["susceptibility_delta_fro_norm"] == pytest.approx(0.0, abs=1e-6)


# --- WT vs WT against a real effect -------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_panel(tmp_path_factory, toy_reference_pdb):
    """Three runs sharing one protein: two silent "mutants" (just more WT) and one displaced site."""
    root = tmp_path_factory.mktemp("panel")
    reference = reference_positions(toy_reference_pdb)
    shifted_pdb = write_shifted_pdb(toy_reference_pdb, str(root / "shifted.pdb"))
    shifted_reference = reference_positions(shifted_pdb)
    runs = {}
    for index, (scenario, mut_pdb, mut_reference) in enumerate(
        [
            ("silent_a", toy_reference_pdb, reference),
            ("silent_b", toy_reference_pdb, reference),
            ("shifted", shifted_pdb, shifted_reference),
        ]
    ):
        run_dir = str(root / scenario / "rep1")
        wt_npz = str(root / scenario / "rep1" / "md" / "wt_traj.npz")
        mut_npz = str(root / scenario / "rep1" / "md" / "mut_traj.npz")
        make_trajectory(reference, seed=10 + index, out_path=wt_npz)
        make_trajectory(mut_reference, seed=20 + index, out_path=mut_npz)
        runs[scenario] = write_run(run_dir, toy_reference_pdb, wt_npz, mut_pdb, mut_npz)
    return {"root": str(root), "runs": runs, "shifted_pdb": shifted_pdb}


def test_two_wt_simulations_are_indistinguishable_but_a_shifted_site_is_not(tmp_path, synthetic_panel, toy_reference_pdb):
    root = synthetic_panel["root"]
    wt_a = os.path.join(root, "silent_a", "rep1", "md", "wt_traj.npz")
    wt_b = os.path.join(root, "silent_b", "rep1", "md", "wt_traj.npz")
    shifted = os.path.join(root, "shifted", "rep1", "md", "mut_traj.npz")

    null = comparison_metrics(_analyse(tmp_path, "null", toy_reference_pdb, wt_a, toy_reference_pdb, wt_b))
    effect = comparison_metrics(_analyse(tmp_path, "effect", toy_reference_pdb, wt_a, synthetic_panel["shifted_pdb"], shifted))

    assert null["classifier_auc"] < 0.75
    assert null["js_divergence"] < 0.15
    assert effect["classifier_auc"] > 0.95
    assert effect["js_divergence"] > 0.4
    assert effect["js_divergence"] > 3 * null["js_divergence"]


def test_null_control_ranks_the_real_effect_above_the_null(synthetic_panel):
    summary = run_null_control(
        panel_dir=synthetic_panel["root"],
        ligand_resnames=["ZN"],
        field_cfg=FieldConfig(**FIELD),
        latent_cfg=LATENT,
        te_cfg=TE,
        temperature_k=300.0,
    )
    assert summary["wt_trajectories"] == 3 and summary["null_pairs"] == 3
    shifted = summary["scenarios"]["shifted"]
    assert "classifier_auc" in shifted["metrics_beyond_null"]
    # one replica and three null pairs can never be more than suggestive
    assert shifted["evidence"] == "suggestive"
    silent_z = max(summary["scenarios"][name]["metrics"]["classifier_auc"]["mean_z"] for name in ("silent_a", "silent_b"))
    assert shifted["metrics"]["classifier_auc"]["mean_z"] > silent_z + 3.0
    for path in summary["outputs"].values():
        assert os.path.exists(path)
    # the runs' own analysis directories are left untouched
    assert not os.path.exists(os.path.join(synthetic_panel["runs"]["shifted"], "analysis"))


def test_calibration_against_null():
    cal = calibrate_against_null(5.0, [1.0, 2.0, 3.0])
    assert cal["n_null"] == 3
    assert cal["null_mean"] == pytest.approx(2.0)
    assert cal["z"] == pytest.approx(3.0)
    assert cal["p_empirical"] == pytest.approx(0.25)  # smallest value three null pairs allow
    assert calibrate_against_null(1.5, [1.0, 2.0, 3.0])["p_empirical"] == pytest.approx(0.75)
    assert calibrate_against_null(1.0, [])["p_empirical"] is None


# --- honest reporting of what was not sampled ------------------------------------------------------


def test_disjoint_basins_have_no_free_energy():
    rng = np.random.default_rng(0)
    u_wt = rng.normal(0.0, 0.1, size=(200, 2))
    u_mut = rng.normal(5.0, 0.1, size=(200, 2))
    result = compute_basin_ddg(u_wt, u_mut, temperature_k=300.0, n_bootstrap=50)
    assert result["population_tv_distance"] == pytest.approx(1.0)
    assert result["n_overlapping_basins"] == 0
    assert all(row["ddg_kcal_mol"] is None and row["overlap"] is False for row in result["basins"])
    # neither trajectory ever changes basin: one effective sample each, not 200
    assert result["effective_samples_wt"] == pytest.approx(1.0)
    assert result["effective_samples_mut"] == pytest.approx(1.0)
    assert result["sampling_ok"] is False


def test_shared_basins_give_finite_ddg_with_intervals():
    rng = np.random.default_rng(1)
    centers = np.array([[0.0, 0.0], [5.0, 0.0]])

    def sample(p_second, n=600):
        labels = (rng.random(n) < p_second).astype(int)
        return centers[labels] + rng.normal(0.0, 0.1, size=(n, 2))

    result = compute_basin_ddg(sample(0.3), sample(0.6), temperature_k=300.0, n_bootstrap=100)
    assert result["n_basins"] == 2 and result["n_overlapping_basins"] == 2
    assert result["population_tv_distance"] == pytest.approx(0.3, abs=0.07)
    assert result["sampling_ok"] is True
    for row in result["basins"]:
        assert row["ddg_ci95_low"] <= row["ddg_kcal_mol"] <= row["ddg_ci95_high"]
    kbt = 0.0019872041 * 300.0
    values = sorted(abs(row["ddg_kcal_mol"]) for row in result["basins"])
    assert values == pytest.approx(sorted([kbt * np.log(0.6 / 0.3), kbt * np.log(0.7 / 0.4)]), abs=0.12)


def test_held_out_separability_does_not_saturate():
    rng = np.random.default_rng(2)
    same = classifier_separability(rng.normal(size=(300, 6)), rng.normal(size=(300, 6)))
    assert same["classifier_auc"] == pytest.approx(0.5, abs=0.08)
    assert same["js_divergence"] < 0.05
    different = classifier_separability(rng.normal(size=(300, 6)), rng.normal(size=(300, 6)) + 4.0)
    assert different["classifier_auc"] > 0.99
    assert different["js_divergence"] > 0.6


# --- metal-site integrity -----------------------------------------------------------------------


def test_metal_site_check_tells_an_intact_site_from_one_that_loses_a_ligand(tmp_path, toy_reference_pdb):
    from pert.analysis.metal_site import metal_site_series
    from pert.deps import app
    from pert.pipelines.site_check import run_site_check

    reference = reference_positions(toy_reference_pdb)
    positions, velocities = make_trajectory(reference, seed=5, n_frames=100, noise_nm=0.005)
    intact_npz = save_trajectory(str(tmp_path / "p" / "s" / "rep1" / "md" / "wt_traj.npz"), positions, velocities)

    (site,) = metal_site_series(toy_reference_pdb, intact_npz, ["ZN"])
    donors = sorted(d["donor"].split(":")[0][:3] + d["donor"].split(":")[1] for d in site["donors"])
    assert donors == ["CYSSG", "CYSSG", "HISNE2", "HISNE2"]  # the Cys2His2 site, nothing else
    assert site["status"] == "intact" and all(d["fraction_bound"] == 1.0 for d in site["donors"])

    # One histidine drifts 6 A away over the second half of the trajectory.
    atoms = list(app.PDBFile(toy_reference_pdb).topology.atoms())
    his = [a for a in atoms if a.residue.name == "HIS" and a.residue.id.strip() == "23"]
    broken = positions.copy()
    ramp = np.clip((np.arange(100) - 50) / 20.0, 0.0, 1.0)[:, None, None] * np.array([0.6, 0.0, 0.0])
    broken[:, [a.index for a in his], :] += ramp
    broken_npz = save_trajectory(str(tmp_path / "p" / "s" / "rep1" / "md" / "mut_traj.npz"), broken, velocities)
    (site,) = metal_site_series(toy_reference_pdb, broken_npz, ["ZN"])
    assert site["status"] == "lost"
    leaving = [d for d in site["donors"] if d["fraction_bound_last_fifth"] < 0.5]
    assert [d["donor"] for d in leaving] == ["HIS23:NE2"] and leaving[0]["final_ang"] > 3.5

    # Panel level: one row per simulation, and the summary says the panel is not clean.
    write_run(str(tmp_path / "p" / "s" / "rep1"), toy_reference_pdb, intact_npz, toy_reference_pdb, broken_npz)
    summary = run_site_check(str(tmp_path / "p"), ligand_resnames=["ZN"])
    assert summary["status_counts_by_system"] == {"wt": {"intact": 1, "lost": 0}, "mut": {"intact": 0, "lost": 1}}
    assert summary["all_intact"] is False and os.path.getsize(summary["outputs"]["plot"]) > 0


# --- verdict rules ------------------------------------------------------------------------------


def _calibrated_row(scenario, values, null, control=False, sampling_ok=True):
    """A run as the null control tabulates it, from its metric values."""
    from pert.analysis.topology_safe import COMPARISON_METRICS

    row = {"scenario": scenario, "control": control, "sampling_ok": sampling_ok,
           "effective_samples_min": 40.0 if sampling_ok else 3.0}
    for m in COMPARISON_METRICS:
        cal = calibrate_against_null(values[m], null[m])
        row.update({m: values[m], f"{m}_z": cal["z"], f"{m}_p": cal["p_empirical"],
                    f"{m}_beyond_null": values[m] > cal["null_max"]})
    return row


def test_verdict_rules_saturation_control_and_convergence():
    from pert.analysis.topology_safe import COMPARISON_METRICS
    from pert.pipelines.null_control import scenario_verdicts

    rng = np.random.default_rng(0)
    null = {m: list(rng.uniform(0.1, 0.3, size=28)) for m in COMPARISON_METRICS}
    null["basin_population_tv"] = list(rng.uniform(0.6, 0.999, size=27)) + [0.999]  # WT pairs reach the ceiling
    inside = {m: 0.2 for m in COMPARISON_METRICS}

    def verdict(rows, name="mutant"):
        return scenario_verdicts(rows, null)[name]

    # A clear, converged, replicated effect in an unsaturated metric is supported.
    effect = dict(inside, js_divergence=0.6)
    entry = verdict([_calibrated_row("mutant", effect, null) for _ in range(2)])
    assert entry["evidence"] == "supported" and entry["metrics_supporting"] == ["js_divergence"]

    # The same effect from unconverged simulations is only suggestive.
    entry = verdict([_calibrated_row("mutant", effect, null, sampling_ok=False) for _ in range(2)])
    assert entry["evidence"] == "suggestive" and "not converged" in entry["evidence_note"]

    # Beating a null that already sits at the ceiling of the metric does not count.
    entry = verdict([_calibrated_row("mutant", dict(inside, basin_population_tv=1.0), null) for _ in range(2)])
    assert entry["metrics_beyond_null"] == ["basin_population_tv"] and entry["metrics_supporting"] == []
    assert entry["evidence"] == "none" and "saturated" in entry["evidence_note"]

    # Nor does an effect that the modelled-WT control reaches as well.
    control = [_calibrated_row("wt_model", dict(inside, js_divergence=0.65 if rep else 0.2), null, control=True) for rep in range(2)]
    entry = verdict([_calibrated_row("mutant", effect, null) for _ in range(2)] + control)
    assert entry["evidence"] == "none" and "wt_model" in entry["evidence_note"]
    assert entry["metrics"]["js_divergence"]["all_replicas_beyond_control"] is False
    # ... while an effect above the control still does.
    entry = verdict([_calibrated_row("mutant", dict(inside, js_divergence=0.68), null) for _ in range(2)] + control)
    assert entry["evidence"] == "supported"
    # The control itself is judged against the null only, and labelled.
    entry = verdict(control, name="wt_model")
    assert entry["is_control"] and entry["evidence"] == "none" and entry["evidence_note"].startswith("Control")


def test_recalibration_reproduces_the_verdicts_without_trajectories(tmp_path, synthetic_panel):
    import shutil

    from pert.pipelines.null_control import recalibrate_null_control

    out_dir = str(tmp_path / "null")
    first = run_null_control(
        panel_dir=synthetic_panel["root"], ligand_resnames=["ZN"], field_cfg=FieldConfig(**FIELD),
        latent_cfg=LATENT, te_cfg=TE, temperature_k=300.0, output_dir=out_dir,
    )
    # Every "mutant" of this panel has the WT sequence, so all three are controls.
    assert sorted(first["control_scenarios"]) == ["shifted", "silent_a", "silent_b"]

    # A copy of the panel without trajectories (what a results archive holds).
    copy = str(tmp_path / "copy")
    shutil.copytree(synthetic_panel["root"], copy, ignore=shutil.ignore_patterns("*.npz", "null_control"))
    copy_out = str(tmp_path / "copy_null")
    shutil.copytree(out_dir, copy_out)
    second = recalibrate_null_control(copy, output_dir=copy_out)
    assert second["null_pairs"] == first["null_pairs"] and second["wt_trajectories"] == first["wt_trajectories"]
    for name, entry in first["scenarios"].items():
        again = second["scenarios"][name]
        assert again["evidence"] == entry["evidence"] and again["metrics_beyond_null"] == entry["metrics_beyond_null"]
        for metric, stats in entry["metrics"].items():
            assert again["metrics"][metric]["mean"] == pytest.approx(stats["mean"], rel=1e-9)
