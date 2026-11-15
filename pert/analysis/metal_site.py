"""Does the metal site hold together during a simulation?

A comparison of two ensembles of a metal site only says something about a
mutation if the site exists in both.  Metal ions are modelled here as
non-bonded particles, so nothing but electrostatics keeps the ligating atoms
around the ion, and a site can lose a ligand, swap it for water or fall apart
within nanoseconds.  This module follows the coordination of every metal ion
through a trajectory:

* the *donors* are the protein N/O/S atoms within ``shell_cutoff_ang`` of the
  ion in the prepared structure (the coordination shell the run started from);
* for every frame, the distance of each donor to the ion, and the number of
  protein donor atoms and of water oxygens within ``bound_cutoff_ang``.

Status of a site:

* ``intact``    - every initial donor is bound in at least 95% of the frames;
* ``lost``      - some initial donor is bound in less than half of the last
                  fifth of the trajectory (it left and did not come back);
* ``distorted`` - anything in between;
* ``no_site``   - the ion had no protein donor to begin with.

Distances use the minimum image when the trajectory stores the box.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pert.constants import METALS
from pert.deps import app, ensure_openmm, unit
from pert.structure.residues import is_protein_residue
from pert.structure.selection import ligand_selection

DONOR_ELEMENTS = ("N", "O", "S")
WATER_RESNAMES = ("HOH", "WAT", "H2O", "TIP3", "TIP")

# A donor counts as kept if it is bound in at least this fraction of the frames.
INTACT_FRACTION = 0.95
# ... and as lost if it is bound in less than this fraction of the last fifth.
LOST_FRACTION = 0.5


def _minimum_image(delta: np.ndarray, box_lengths: Optional[np.ndarray]) -> np.ndarray:
    """Wrap displacement vectors into the central image of an orthorhombic box."""
    if box_lengths is None:
        return delta
    return delta - box_lengths * np.round(delta / box_lengths)


def metal_site_series(
    pdb_path: str,
    traj_npz: str,
    ligand_resnames: Optional[Sequence[str]] = None,
    shell_cutoff_ang: float = 2.6,
    bound_cutoff_ang: float = 3.0,
) -> List[Dict[str, Any]]:
    """Coordination of every metal ion of one simulation, frame by frame.

    Returns one entry per metal ion (see the module docstring for ``status``)
    with per-donor statistics and the time series, in Angstrom.  ``pdb_path``
    is the prepared (simulated) structure, whose atom order is the trajectory's.
    """
    ensure_openmm()
    selection = ligand_selection(ligand_resnames)
    if selection and not selection.resnames:
        return []  # the ligand is a chain (a peptide): there is no metal site to follow
    pdb = app.PDBFile(pdb_path)
    atoms = list(pdb.topology.atoms())
    reference = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=np.float64)

    with np.load(traj_npz) as data:
        positions = np.asarray(data["positions"], dtype=np.float64)
        time_ps = np.asarray(data["time_ps"], dtype=np.float64) if "time_ps" in data else None
        box = np.asarray(data["box_nm"], dtype=np.float64) if "box_nm" in data else None
    if positions.shape[1] != len(atoms):
        raise ValueError(
            f"'{traj_npz}' has {positions.shape[1]} atoms but '{pdb_path}' has {len(atoms)}: "
            "the structure is not the one that was simulated."
        )
    n_frames = positions.shape[0]
    if time_ps is None:
        time_ps = np.arange(n_frames, dtype=np.float64)
    # Orthorhombic boxes only (what the pipeline builds): the diagonal is enough.
    box_lengths = None if box is None else np.stack([box[:, i, i] for i in range(3)], axis=1)[:, None, :]

    wanted = set(selection.resnames)
    metals = [
        atom
        for atom in atoms
        if atom.element is not None
        and atom.element.symbol.upper() in METALS
        and not is_protein_residue(atom.residue.name)
        and (not wanted or atom.residue.name.upper() in wanted)
    ]
    donor_candidates = np.asarray(
        [
            atom.index
            for atom in atoms
            if is_protein_residue(atom.residue.name) and atom.element is not None and atom.element.symbol in DONOR_ELEMENTS
        ],
        dtype=np.int64,
    )
    water_oxygens = np.asarray(
        [
            atom.index
            for atom in atoms
            if atom.residue.name.upper() in WATER_RESNAMES and atom.element is not None and atom.element.symbol == "O"
        ],
        dtype=np.int64,
    )

    shell_nm = float(shell_cutoff_ang) * 0.1
    bound_nm = float(bound_cutoff_ang) * 0.1
    last_fifth = slice(n_frames - max(1, n_frames // 5), n_frames)

    def distances(indices: np.ndarray, metal_index: int) -> np.ndarray:
        """(frames, len(indices)) distances to the ion, nm."""
        delta = positions[:, indices, :] - positions[:, [metal_index], :]
        return np.linalg.norm(_minimum_image(delta, box_lengths), axis=2)

    results: List[Dict[str, Any]] = []
    for metal in metals:
        initial = np.linalg.norm(reference[donor_candidates] - reference[metal.index], axis=1)
        donors = donor_candidates[initial <= shell_nm]
        entry: Dict[str, Any] = {
            "metal": f"{metal.residue.name}{metal.residue.id}:{metal.name}",
            "frames": int(n_frames),
            "time_ns": (time_ps / 1000.0).tolist(),
            "donors": [],
        }
        protein_d = distances(donor_candidates, metal.index)
        entry["protein_donors_bound"] = np.sum(protein_d <= bound_nm, axis=1).astype(int).tolist()
        entry["waters_bound"] = (
            np.sum(distances(water_oxygens, metal.index) <= bound_nm, axis=1).astype(int).tolist()
            if water_oxygens.size
            else [0] * n_frames
        )
        entry["mean_protein_donors_bound"] = float(np.mean(entry["protein_donors_bound"]))
        entry["mean_waters_bound"] = float(np.mean(entry["waters_bound"]))

        kept, lost = True, False
        for donor in donors:
            atom = atoms[int(donor)]
            series = distances(np.asarray([donor]), metal.index)[:, 0]
            bound = series <= bound_nm
            fraction = float(np.mean(bound))
            fraction_end = float(np.mean(bound[last_fifth]))
            kept = kept and fraction >= INTACT_FRACTION
            lost = lost or fraction_end < LOST_FRACTION
            entry["donors"].append(
                {
                    "donor": f"{atom.residue.name}{atom.residue.id}:{atom.name}",
                    "initial_ang": float(np.linalg.norm(reference[donor] - reference[metal.index]) * 10.0),
                    "mean_ang": float(np.mean(series) * 10.0),
                    "max_ang": float(np.max(series) * 10.0),
                    "final_ang": float(np.mean(series[last_fifth]) * 10.0),
                    "fraction_bound": fraction,
                    "fraction_bound_last_fifth": fraction_end,
                    "distance_ang": (series * 10.0).tolist(),
                }
            )
        if not entry["donors"]:
            entry["status"] = "no_site"
        elif lost:
            entry["status"] = "lost"
        elif kept:
            entry["status"] = "intact"
        else:
            entry["status"] = "distorted"
        results.append(entry)
    return results
