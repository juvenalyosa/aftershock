"""Synthetic WT/"mutant" trajectories around the toy zinc finger, for analysis tests.

Atoms jitter around a reference structure with exponentially correlated noise,
so the trajectories are stationary, well mixed and statistically identical
unless the reference itself is changed.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def reference_positions(pdb_path: str) -> np.ndarray:
    from pert.deps import app, unit

    pdb = app.PDBFile(pdb_path)
    return np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=np.float64)


def write_shifted_pdb(pdb_path: str, out_path: str, shift_nm: float = 0.25, residues=range(3, 8)) -> str:
    """Copy of the structure with site-lining residues (Cys3..Glu7) displaced: a crude "mutant" with a real effect."""
    from pert.deps import Vec3, app, unit

    pdb = app.PDBFile(pdb_path)
    pos = reference_positions(pdb_path)
    wanted = {str(r) for r in residues}
    for atom in pdb.topology.atoms():
        if atom.residue.id.strip() in wanted:
            pos[atom.index] += np.array([shift_nm, 0.0, 0.0])
    with open(out_path, "w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(pdb.topology, [Vec3(*xyz) for xyz in pos] * unit.nanometer, handle, keepIds=True)
    return out_path


def make_trajectory(
    reference_nm: np.ndarray,
    seed: int,
    n_frames: int = 240,
    out_path: Optional[str] = None,
    noise_nm: float = 0.02,
    memory: float = 0.5,
):
    """Jitter around ``reference_nm``; returns ``(positions, velocities)`` and optionally saves the npz."""
    rng = np.random.default_rng(seed)
    offset = np.zeros_like(reference_nm)
    positions = np.zeros((n_frames,) + reference_nm.shape)
    for t in range(n_frames):
        offset = memory * offset + noise_nm * rng.normal(size=reference_nm.shape)
        positions[t] = reference_nm + offset
    velocities = 0.3 * rng.normal(size=positions.shape)
    if out_path is not None:
        save_trajectory(out_path, positions, velocities)
    return positions, velocities


def save_trajectory(path: str, positions: np.ndarray, velocities: np.ndarray, dt_ps: float = 1.0) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        positions=positions,
        velocities=velocities,
        time_ps=np.arange(positions.shape[0], dtype=float) * dt_ps,
        dt_ps=np.asarray([dt_ps]),
    )
    return path


def random_rotation(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def write_run(run_dir: str, wt_pdb: str, wt_npz: str, mut_pdb: str, mut_npz: str) -> str:
    """Minimal run directory (only what post-processing reads) pointing at existing files."""
    os.makedirs(run_dir, exist_ok=True)
    summary = {
        "md": {
            "wt": {"prepared_pdb": wt_pdb, "traj_npz": wt_npz, "traj_dcd": None},
            "mut": {"prepared_pdb": mut_pdb, "traj_npz": mut_npz, "traj_dcd": None},
        }
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle)
    return run_dir
