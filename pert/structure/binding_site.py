"""Definition of the active/binding site.

The site is defined geometrically: every protein residue with a heavy atom
within ``cutoff_ang`` of the ligand (default 6 A).  It is the *target* of all
allosteric analyses: mutation sources are scored by how they couple to it.
"""

from __future__ import annotations

import csv
import os
from typing import List, Sequence, Tuple

import numpy as np

from pert.constants import AA3_TO_1, STANDARD_AA3
from pert.deps import app, ensure_openmm, unit


def binding_site_keys_from_pdb(
    pdb_path: str,
    ligand_resnames: Sequence[str],
    cutoff_ang: float,
) -> List[Tuple[str, str]]:
    """``(chain, resid)`` of protein residues with a heavy atom within ``cutoff_ang`` of any ligand."""
    ensure_openmm()
    ligand_set = {name.strip().upper() for name in ligand_resnames if name.strip()}
    if not ligand_set:
        return []
    pdb = app.PDBFile(pdb_path)
    pos_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    lig_idx = [atom.index for atom in pdb.topology.atoms() if atom.residue.name.upper() in ligand_set]
    if not lig_idx:
        return []
    lig_xyz = pos_nm[lig_idx]
    cutoff_nm = float(cutoff_ang) * 0.1
    keys: List[Tuple[str, str]] = []
    for chain in pdb.topology.chains():
        chain_id = (chain.id or "").strip()
        for residue in chain.residues():
            if residue.name.upper() not in AA3_TO_1:
                continue
            atom_idx = [a.index for a in residue.atoms() if (a.element is None or a.element.symbol != "H")]
            if not atom_idx:
                continue
            xyz = pos_nm[atom_idx]
            d = np.linalg.norm(xyz[:, None, :] - lig_xyz[None, :, :], axis=2)
            if float(np.min(d)) <= cutoff_nm:
                keys.append((chain_id, str(residue.id).strip()))
    return sorted(set(keys))


def binding_site_residues(
    pdb_path: str,
    ligand_resname: str,
    cutoff_ang: float,
) -> List[Tuple[str, str, float]]:
    """Binding-site residues of a structure.

    Returns ``[(chain, residue_key, min_heavy_atom_distance_A), ...]`` sorted by
    distance, for standard residues within ``cutoff_ang`` of the ligand.
    """
    pdb = app.PDBFile(pdb_path)
    pos_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    cutoff_nm = cutoff_ang * 0.1

    ligand_indices = []
    for atom in pdb.topology.atoms():
        if atom.residue.name.upper() == ligand_resname.upper():
            ligand_indices.append(atom.index)
    if not ligand_indices:
        return []

    lig_xyz = pos_nm[ligand_indices]
    result = []
    for residue in pdb.topology.residues():
        if residue.name.upper() not in STANDARD_AA3:
            continue
        atom_indices = [a.index for a in residue.atoms() if (a.element is None or a.element.symbol != "H")]
        if not atom_indices:
            continue
        xyz = pos_nm[atom_indices]
        d = np.linalg.norm(xyz[:, None, :] - lig_xyz[None, :, :], axis=2)
        min_d = float(np.min(d)) * 10.0
        if min_d <= cutoff_ang:
            result.append((residue.chain.id, f"{residue.name}{residue.id}", min_d))
    result.sort(key=lambda x: x[2])
    return result


def write_binding_csv(
    wt_pdb: str,
    mut_pdb: str,
    ligand_resname: str,
    cutoff_ang: float,
    out_csv: str,
) -> None:
    """Write the union of WT and mutant site residues with their ligand distances and the shift (MUT - WT)."""
    wt = binding_site_residues(wt_pdb, ligand_resname, cutoff_ang)
    mut = binding_site_residues(mut_pdb, ligand_resname, cutoff_ang)

    wt_map = {row[1]: row[2] for row in wt}
    mut_map = {row[1]: row[2] for row in mut}
    keys = sorted(set(wt_map) | set(mut_map))
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["residue", "wt_min_dist_ang", "mut_min_dist_ang", "delta_mut_minus_wt"])
        for key in keys:
            w = wt_map.get(key)
            m = mut_map.get(key)
            delta = None if (w is None or m is None) else (m - w)
            writer.writerow(
                [
                    key,
                    "" if w is None else f"{w:.3f}",
                    "" if m is None else f"{m:.3f}",
                    "" if delta is None else f"{delta:.3f}",
                ]
            )


def read_binding_delta_csv(path: str) -> List[Tuple[str, float]]:
    """``[(residue_key, delta_distance), ...]`` for site residues present in both WT and mutant."""
    rows: List[Tuple[str, float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = str(row.get("residue", "")).strip()
            raw = str(row.get("delta_mut_minus_wt", "")).strip()
            if not key or not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            rows.append((key, value))
    return rows


def read_binding_residue_keys(binding_csv: str) -> List[str]:
    """Residue keys listed in a binding-site CSV (empty list if the file does not exist)."""
    if not os.path.exists(binding_csv):
        return []
    keys: List[str] = []
    with open(binding_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = str(row.get("residue", "")).strip()
            if key:
                keys.append(key)
    return sorted(set(keys))
