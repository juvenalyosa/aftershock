"""Run-directory path resolution, CLI dispatch and a structure-level smoke test."""

from __future__ import annotations

import importlib
import os

import pytest

from pert.cli import COMMANDS, main
from pert.utils.paths import find_run_dirs, resolve_run_path, scenario_from_run_dir


def _touch(path) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("x")
    return str(path)


def test_resolve_run_path_follows_relocated_runs(tmp_path):
    run_dir = tmp_path / "runs" / "panel" / "snp_k2r" / "rep1"
    real = _touch(run_dir / "md" / "wt_traj.npz")
    # recorded relative to the directory the run was launched from, before the run was moved
    recorded = os.path.join("panel", "snp_k2r", "rep1", "md", "wt_traj.npz")
    assert resolve_run_path(str(run_dir), recorded) == real
    # recorded as an absolute path on another machine
    assert resolve_run_path(str(run_dir), "/cluster/scratch/panel/snp_k2r/rep1/md/wt_traj.npz") == real
    # plain run-relative paths keep working
    assert resolve_run_path(str(run_dir), os.path.join("md", "wt_traj.npz")) == real


def test_resolve_run_path_keeps_the_historical_answer_for_missing_files(tmp_path):
    run_dir = str(tmp_path / "rep1")
    assert resolve_run_path(run_dir, "/nowhere/file.pdb") == "/nowhere/file.pdb"
    assert resolve_run_path(run_dir, "md/missing.pdb") == os.path.join(run_dir, "md", "missing.pdb")


def test_panel_layout_helpers(tmp_path):
    for scenario, rep in [("snp_k2r", "rep1"), ("snp_k2r", "rep2"), ("dup_q20", "rep1")]:
        _touch(tmp_path / scenario / rep / "run_summary.json")
    run_dirs = find_run_dirs(str(tmp_path))
    assert len(run_dirs) == 3
    assert sorted({scenario_from_run_dir(d) for d in run_dirs}) == ["dup_q20", "snp_k2r"]


def test_cli_lists_every_command(capsys):
    main([])
    usage = capsys.readouterr().out
    for command in COMMANDS:
        assert command in usage


def test_cli_rejects_unknown_commands():
    with pytest.raises(SystemExit):
        main(["no-such-command"])


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_every_command_module_exposes_main(command):
    pytest.importorskip("matplotlib")
    module = importlib.import_module(COMMANDS[command][0])
    assert callable(module.main)


def test_fast_scan_of_identical_structures(tmp_path, toy_reference_pdb):
    """WT vs itself: perfect superposition and no structural/contact signal."""
    pytest.importorskip("openmm")
    pytest.importorskip("matplotlib")
    from pert.allostery.fast_scan import run_fast_scan

    summary = run_fast_scan(
        wt_pdb=toy_reference_pdb,
        mut_pdb=toy_reference_pdb,
        ligand_resname="ZN",
        output_dir=str(tmp_path / "scan"),
        mutation_label="identity",
    )
    assert summary["aligned_ca_rmsd_ang"] == pytest.approx(0.0, abs=1e-6)
    assert summary["deleted_wt_residues"] == []
    assert summary["target_residues"], "the zinc site must be detected"
    for row in summary["top_residues"]:
        assert row["structural_component"] == 0.0
        assert row["contact_component"] == 0.0
        assert row["strain_component"] == 0.0
    for path in summary["outputs"].values():
        assert os.path.exists(path)
