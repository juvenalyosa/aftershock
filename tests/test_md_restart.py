"""MD stage: reproducible preparation, checkpoints, restart, extension, no silent overwrite.

Uses the Reference platform (deterministic) on the 25-residue toy protein, a few
dozen steps per run.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import pytest

pytest.importorskip("openmm")

from pert.config import MDConfig  # noqa: E402
from pert.deps import app  # noqa: E402
from pert.simulation.md import md_signature, run_openmm_md  # noqa: E402


@pytest.fixture(scope="module")
def toy_inputs(repo_root):
    pdb = os.path.join(repo_root, "data", "toy", "1ZNF.pdb")
    xml = os.path.join(repo_root, "data", "toy", "empty_ligand.xml")
    if not (os.path.exists(pdb) and os.path.exists(xml)):
        pytest.skip("toy system not available")
    return pdb, xml


def _config(**overrides) -> MDConfig:
    settings = dict(
        add_solvent=False,
        nonbonded_method="CutoffNonPeriodic",
        minimization_max_iterations=20,
        heating_steps=20,
        heating_stages=2,
        equilibration_steps=20,
        production_steps=80,
        report_interval=10,
        checkpoint_interval_steps=20,
        platform="Reference",
        random_seed=11,
    )
    settings.update(overrides)
    return MDConfig(**settings)


def _run(toy_inputs, directory, **kwargs):
    config = kwargs.pop("config", None) or _config()
    return run_openmm_md(toy_inputs[0], toy_inputs[1], config, str(directory / "wt"), **kwargs)


def _trajectory(artifacts):
    with np.load(artifacts.trajectory_npz) as data:
        return {key: data[key] for key in data.files}


def _dcd_frames(path: str) -> int:
    with open(path, "rb") as handle:
        return struct.unpack("<i", handle.read(12)[8:12])[0]


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory, toy_inputs):
    """One uninterrupted simulation that the other tests compare against."""
    directory = tmp_path_factory.mktemp("uninterrupted")
    return directory, _run(toy_inputs, directory)


def test_run_writes_a_restartable_and_documented_simulation(reference_run):
    directory, artifacts = reference_run
    trajectory = _trajectory(artifacts)
    assert artifacts.n_frames == 9 and trajectory["positions"].shape[0] == 9
    assert np.allclose(trajectory["time_ps"], np.arange(9) * 0.02)
    assert _dcd_frames(artifacts.trajectory_dcd) == 9  # one DCD frame per stored frame
    for name in ("wt_prepared.pdb", "wt_system.xml", "wt_start.xml", "wt_md.json", "wt_progress.json"):
        assert (directory / name).exists(), name
    log = json.loads((directory / "wt_md.json").read_text())
    assert log["completed"] is True and log["random_seed"] == 11 and log["resumes"] == 0
    progress = json.loads((directory / "wt_progress.json").read_text())
    assert progress["stage"] == "completed" and progress["steps_done"] == 80
    # trajectory parts are gone, the last checkpoint stays (to extend the run later)
    assert sorted(os.listdir(directory / "wt_restart")) == ["step_000000000080.chk", "step_000000000080.xml"]


def test_same_seed_gives_the_same_trajectory(tmp_path, toy_inputs, reference_run):
    again = _trajectory(_run(toy_inputs, tmp_path))
    first = _trajectory(reference_run[1])
    assert np.array_equal(first["positions"], again["positions"])
    assert np.array_equal(first["velocities"], again["velocities"])


def test_a_different_seed_gives_a_different_trajectory(tmp_path, toy_inputs, reference_run):
    other = _trajectory(_run(toy_inputs, tmp_path, config=_config(random_seed=12)))
    assert not np.allclose(_trajectory(reference_run[1])["positions"], other["positions"])


def test_a_complete_simulation_is_reused(toy_inputs, reference_run):
    directory, artifacts = reference_run
    stamp = os.path.getmtime(artifacts.trajectory_npz)
    again = _run(toy_inputs, directory)
    assert again.reused is True and again.n_frames == 9
    assert os.path.getmtime(artifacts.trajectory_npz) == stamp


def test_an_interrupted_simulation_continues_where_it_stopped(tmp_path, toy_inputs, reference_run, monkeypatch):
    progress_path = str(tmp_path / "wt_progress.json")
    original_step = app.Simulation.step
    production_calls = {"n": 0}

    def crashing_step(self, steps):
        if os.path.exists(progress_path):  # production has started
            production_calls["n"] += 1
            if production_calls["n"] == 6:  # dies while computing steps 51-60
                raise KeyboardInterrupt("simulated crash")
        return original_step(self, steps)

    monkeypatch.setattr(app.Simulation, "step", crashing_step)
    with pytest.raises(KeyboardInterrupt):
        _run(toy_inputs, tmp_path)
    monkeypatch.setattr(app.Simulation, "step", original_step)

    progress = json.loads(open(progress_path).read())
    assert progress["stage"] == "production" and progress["steps_done"] == 40  # last checkpoint
    assert not (tmp_path / "wt_traj.npz").exists()

    resumed = _run(toy_inputs, tmp_path)
    assert resumed.resumes == 1 and resumed.reused is False and resumed.n_frames == 9
    log = json.loads((tmp_path / "wt_md.json").read_text())
    assert log["last_restored_from"] == "checkpoint"
    trajectory, expected = _trajectory(resumed), _trajectory(reference_run[1])
    assert np.allclose(trajectory["time_ps"], expected["time_ps"])
    # the checkpoint restores the full dynamical state: same trajectory as without the crash
    assert np.allclose(trajectory["positions"], expected["positions"], atol=1e-9)
    assert _dcd_frames(resumed.trajectory_dcd) == 9


def test_a_finished_simulation_can_be_extended(tmp_path, toy_inputs, reference_run):
    short = _run(toy_inputs, tmp_path, config=_config(production_steps=40))
    assert short.n_frames == 5
    extended = _run(toy_inputs, tmp_path, config=_config(production_steps=80))
    assert extended.n_frames == 9 and extended.resumes == 1
    assert np.allclose(_trajectory(extended)["positions"], _trajectory(reference_run[1])["positions"], atol=1e-9)
    with pytest.raises(RuntimeError, match="Refusing to truncate"):
        _run(toy_inputs, tmp_path, config=_config(production_steps=40))


def test_files_from_other_settings_are_never_overwritten_silently(tmp_path, toy_inputs):
    _run(toy_inputs, tmp_path, config=_config(production_steps=20))
    hotter = _config(production_steps=20, temperature_k=310.0)
    assert md_signature(*toy_inputs, hotter) != md_signature(*toy_inputs, _config(production_steps=20))
    with pytest.raises(RuntimeError, match="different"):
        _run(toy_inputs, tmp_path, config=hotter)
    redone = _run(toy_inputs, tmp_path, config=hotter, resume=False)
    assert redone.reused is False and redone.resumes == 0


def test_signature_ignores_what_does_not_change_the_physics(toy_inputs):
    base = md_signature(*toy_inputs, _config())
    assert base == md_signature(*toy_inputs, _config(platform="CPU"))
    assert base == md_signature(*toy_inputs, _config(production_steps=10**6))
    assert base == md_signature(*toy_inputs, _config(checkpoint_interval_steps=0))
    assert base != md_signature(*toy_inputs, _config(random_seed=12))
    assert base != md_signature(*toy_inputs, _config(timestep_fs=1.0))


def test_restart_on_another_platform_uses_the_portable_state(tmp_path, toy_inputs):
    """A binary checkpoint only loads where it was written; the XML state is the fallback."""
    _run(toy_inputs, tmp_path, config=_config(production_steps=40))
    extended = _run(toy_inputs, tmp_path, config=_config(production_steps=80, platform="CPU"))
    log = json.loads((tmp_path / "wt_md.json").read_text())
    assert extended.n_frames == 9 and extended.resumes == 1
    assert log["last_restored_from"] == "state_xml" and log["platform_selected"] == "CPU"
    trajectory = _trajectory(extended)
    assert np.allclose(trajectory["time_ps"], np.arange(9) * 0.02)
    assert np.isfinite(trajectory["positions"]).all()
    # continuous with the first half: the frame at the restart point is the stored one
    assert np.abs(trajectory["positions"][5] - trajectory["positions"][4]).max() < 0.1


# --- metal-site protonation -------------------------------------------------------------------


def _site_atoms(prepared_pdb):
    """Residue name and hydrogen names of the four zinc ligands of 1ZNF."""
    out = {}
    for residue in app.PDBFile(prepared_pdb).topology.residues():
        if residue.id in ("3", "6", "19", "23") and residue.name not in ("ZN",):
            out[int(residue.id)] = (residue.name, {atom.name for atom in residue.atoms()})
    return out


def test_zinc_bound_residues_get_a_coordinating_protonation_state(reference_run):
    directory, artifacts = reference_run
    site = _site_atoms(artifacts.prepared_pdb)
    for resid in (3, 6):  # cysteines: thiolates, no thiol hydrogen
        name, atoms = site[resid]
        assert name == "CYM" and "HG" not in atoms
    for resid in (19, 23):  # histidines bind through NE2: the proton sits on ND1
        name, atoms = site[resid]
        assert name == "HIS" and "HD1" in atoms and "HE2" not in atoms
    log = json.loads((directory / "wt_md.json").read_text())
    assert log["metal_site_protonation"] == {"CYS3": "CYM", "CYS6": "CYM", "HIS19": "HID", "HIS23": "HID"}

    # the renamed cysteines are still protein residues for every analysis layer
    from pert.structure.residues import protein_ca_records

    sequence = "".join(record["aa1"] for record in protein_ca_records(artifacts.prepared_pdb))
    assert sequence == "YKCGLCERSFVEKSALSRHQRVHKN"


def test_metal_site_protonation_can_be_switched_off(tmp_path, toy_inputs):
    artifacts = _run(toy_inputs, tmp_path, config=_config(production_steps=10, metal_site_protonation=False))
    site = _site_atoms(artifacts.prepared_pdb)
    assert site[3][0] == "CYS" and "HG" in site[3][1]
    assert json.loads((tmp_path / "wt_md.json").read_text())["metal_site_protonation"] == {}
    assert md_signature(*toy_inputs, _config()) != md_signature(*toy_inputs, _config(metal_site_protonation=False))
