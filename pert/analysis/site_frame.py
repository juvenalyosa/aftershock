"""Shared reference frame of the binding site for a WT/mutant pair.

Density fields are only comparable between two systems if both are expressed
in the *same* coordinate frame.  The frame used here is attached to the protein
scaffold of the site:

1. WT and mutant residues are paired by sequence alignment;
2. the *anchors* are the C-alpha atoms of paired residues that line the ligand
   in the WT structure (heavy atom within ``site_cutoff_ang``), topped up with
   the next-nearest paired residues until there are ``min_anchors`` of them;
3. every frame of either system is rigidly superposed, through its anchors,
   onto the anchor geometry of the WT reference structure;
4. the origin is the position of the ligand centroid in that reference.

Consequences:

* rotating or translating a whole trajectory changes nothing (the legacy frame
  was referenced to each system's own first frame, so the orientation of a
  simulation leaked into every estimand);
* the pocket region is tied to the site itself, and the motion of the ligand
  inside the site (including a monatomic ion) stays visible in the ``L`` fields.

This is the only place where a WT<->mutant correspondence is used, and only
for residues present in both systems; the fields and the estimands remain free
of any atom mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from pert.deps import app, ensure_openmm, unit
from pert.structure.comparison import aligned_ca_pair_records
from pert.structure.selection import ligand_selection

# Anchors closer to a straight line than this (second singular value of the
# centred reference, nm) cannot define an orientation.
_MIN_PLANAR_SPREAD_NM = 0.05


@dataclass
class SiteFrame:
    """Frame definition for one system of a WT/mutant pair.

    Attributes:
        anchor_indices: atoms of *this* system that are fitted to the reference.
        reference_coords_nm: reference geometry of the anchors (same array for
            both systems, taken from the WT structure).
        reference_origin_nm: site centre in the reference geometry (WT ligand centroid).
        anchor_labels: residue keys of the anchors in this system, for reports.
    """

    anchor_indices: np.ndarray
    reference_coords_nm: np.ndarray
    reference_origin_nm: np.ndarray
    anchor_labels: List[str]


def _planar_spread_nm(coords: np.ndarray) -> float:
    """Second singular value of the centred coordinates (0 for collinear points)."""
    if coords.shape[0] < 3:
        return 0.0
    singular = np.linalg.svd(coords - coords.mean(axis=0), compute_uv=False)
    return float(singular[1])


def build_shared_site_frames(
    wt_pdb: str,
    mut_pdb: str,
    ligand_resnames: Sequence[str],
    site_cutoff_ang: float = 6.0,
    min_anchors: int = 6,
) -> Tuple[SiteFrame, SiteFrame, Dict[str, object]]:
    """Build the common site frame of a WT/mutant pair.

    Returns ``(wt_frame, mut_frame, diagnostics)``.  Both frames share the same
    reference geometry; they differ only in which atoms of their own topology
    are the anchors.
    """
    ensure_openmm()
    selection = ligand_selection(ligand_resnames)
    wt = app.PDBFile(wt_pdb)
    wt_pos = np.asarray(wt.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    wt_atoms = list(wt.topology.atoms())

    ligand_idx = [
        atom.index
        for atom in wt_atoms
        if selection.matches(atom.residue) and (atom.element is None or atom.element.symbol != "H")
    ]
    if not ligand_idx:
        raise ValueError(
            f"No ligand atoms ({selection.describe()}) found in '{wt_pdb}'. The shared site frame "
            "needs the ligand to locate the site (use frame_mode='legacy_self' for ligand-free systems)."
        )
    ligand_xyz = wt_pos[ligand_idx]
    origin = ligand_xyz.mean(axis=0)

    # Distance of every paired residue to the ligand in the WT reference.
    pairs = aligned_ca_pair_records(wt_pdb, mut_pdb, selection)
    candidates = []
    for rec in pairs:
        residue = wt_atoms[int(rec["wt_atom_index"])].residue
        heavy = [a.index for a in residue.atoms() if (a.element is None or a.element.symbol != "H")]
        dist = np.linalg.norm(wt_pos[heavy][:, None, :] - ligand_xyz[None, :, :], axis=2)
        candidates.append((float(dist.min()) * 10.0, rec))
    candidates.sort(key=lambda item: item[0])

    cutoff = float(site_cutoff_ang)
    n_site = sum(1 for dist_ang, _ in candidates if dist_ang <= cutoff)
    n_take = min(len(candidates), max(n_site, int(min_anchors), 3))
    # Extend until the anchors can define an orientation.
    while n_take < len(candidates) and _planar_spread_nm(
        wt_pos[[int(rec["wt_atom_index"]) for _, rec in candidates[:n_take]]]
    ) < _MIN_PLANAR_SPREAD_NM:
        n_take += 1
    chosen = candidates[:n_take]

    wt_idx = np.asarray([int(rec["wt_atom_index"]) for _, rec in chosen], dtype=np.int64)
    mut_idx = np.asarray([int(rec["mut_atom_index"]) for _, rec in chosen], dtype=np.int64)
    reference = wt_pos[wt_idx].copy()
    spread = _planar_spread_nm(reference)
    if len(chosen) < 3 or spread < _MIN_PLANAR_SPREAD_NM:
        raise ValueError(
            f"Only {len(chosen)} usable site anchors (planar spread {spread:.3f} nm); "
            "cannot define the orientation of the site frame."
        )

    wt_labels = [str(rec["wt_residue_key"]) for _, rec in chosen]
    mut_labels = [str(rec["mut_residue_key"]) for _, rec in chosen]
    wt_frame = SiteFrame(wt_idx, reference, origin.copy(), wt_labels)
    mut_frame = SiteFrame(mut_idx, reference, origin.copy(), mut_labels)
    diagnostics: Dict[str, object] = {
        "anchor_count": int(len(chosen)),
        "anchors_within_site_cutoff": int(min(n_site, len(chosen))),
        "site_cutoff_ang": cutoff,
        "min_anchors": int(min_anchors),
        "anchor_residues_wt": wt_labels,
        "anchor_residues_mut": mut_labels,
        "anchor_max_ligand_distance_ang": float(chosen[-1][0]),
        "reference_planar_spread_ang": spread * 10.0,
    }
    return wt_frame, mut_frame, diagnostics


def compute_shared_frame_series(
    positions: np.ndarray,
    frame: SiteFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Origin and orientation of the shared site frame in every MD frame.

    Returns ``(centers, rotations, fit_rmsd_nm)`` with the same convention as
    the legacy frame: site-frame coordinates are
    ``rotations[t].T @ (x - centers[t])``.  ``fit_rmsd_nm`` is the RMSD of the
    anchors to the reference after superposition, i.e. how much the site
    scaffold itself is deformed in that frame.
    """
    reference = np.asarray(frame.reference_coords_nm, dtype=np.float64)
    ref_centroid = reference.mean(axis=0)
    ref_centered = reference - ref_centroid
    origin_offset = ref_centroid - np.asarray(frame.reference_origin_nm, dtype=np.float64)

    num_frames = positions.shape[0]
    centers = np.zeros((num_frames, 3), dtype=np.float64)
    rotations = np.zeros((num_frames, 3, 3), dtype=np.float64)
    fit_rmsd = np.zeros(num_frames, dtype=np.float64)

    for t in range(num_frames):
        anchors = positions[t, frame.anchor_indices, :]
        centroid = anchors.mean(axis=0)
        cur_centered = anchors - centroid
        h = cur_centered.T @ ref_centered
        u, _, vt = np.linalg.svd(h)
        q = vt.T @ u.T  # current -> reference
        if np.linalg.det(q) < 0:
            vt[-1, :] *= -1.0
            q = vt.T @ u.T
        # x_site = q @ (x - centroid) + origin_offset = q @ (x - centers[t])
        rotations[t] = q.T
        centers[t] = centroid - q.T @ origin_offset
        fitted = (q @ cur_centered.T).T
        fit_rmsd[t] = np.sqrt(np.mean(np.sum((fitted - ref_centered) ** 2, axis=1)))

    return centers, rotations, fit_rmsd
