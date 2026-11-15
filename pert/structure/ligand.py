"""Transfer of ligand coordinates from the WT reference to the mutant model.

MODELLER builds the mutant protein without the ligand.  The ligand is copied
from the WT reference after superposing the *local* environment of the site
(C-alpha atoms within 12 A of the ligand centroid, matched through sequence
alignment), which keeps the placement correct even when distant regions of the
mutant have moved or changed length.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pert.deps import app, ensure_openmm, unit, Vec3
from pert.sequence.alignment import align_index_pairs
from pert.structure.geometry import kabsch_fit
from pert.structure.residues import ca_records_from_topology
from pert.structure.selection import ligand_selection


def _extract_ligand_subset(
    topology: "app.Topology",
    positions: "unit.Quantity",
    ligand_resnames: Sequence[str],
) -> Tuple["app.Topology", List["unit.Quantity"], List[int]]:
    """Copy the ligand residues (atoms, bonds, positions) into a new topology."""
    ensure_openmm()
    selection = ligand_selection(ligand_resnames)
    lig_top = app.Topology()
    atom_map = {}
    selected_indices: List[int] = []
    lig_positions: List[unit.Quantity] = []
    chain_cache = {}

    for chain in topology.chains():
        for residue in chain.residues():
            if not selection.matches(residue):
                continue
            if chain not in chain_cache:
                chain_cache[chain] = lig_top.addChain(chain.id)
            new_chain = chain_cache[chain]
            new_res = lig_top.addResidue(
                residue.name,
                new_chain,
                id=residue.id,
                insertionCode=residue.insertionCode,
            )
            for atom in residue.atoms():
                new_atom = lig_top.addAtom(
                    atom.name,
                    atom.element,
                    new_res,
                    id=atom.id,
                    formalCharge=atom.formalCharge,
                )
                atom_map[atom] = new_atom
                selected_indices.append(atom.index)
                lig_positions.append(positions[atom.index])

    for atom1, atom2 in topology.bonds():
        if atom1 in atom_map and atom2 in atom_map:
            try:
                lig_top.addBond(atom_map[atom1], atom_map[atom2], type=None, order=None)
            except TypeError:
                lig_top.addBond(atom_map[atom1], atom_map[atom2])

    return lig_top, lig_positions, selected_indices


def transfer_ligand_by_local_alignment(
    reference_pdb: str,
    mutant_pdb: str,
    output_pdb: str,
    ligand_resnames: Sequence[str],
    chain_id: Optional[str],
    local_radius_ang: float = 12.0,
) -> Dict[str, float]:
    """Place the WT ligand into the mutant model.

    The mutant is superposed on the reference using C-alpha atoms close to the
    ligand (>= 6 aligned pairs within ``local_radius_ang``, otherwise all aligned
    pairs); the same transform is applied to the ligand, which replaces any ligand
    already present.  Returns alignment diagnostics (pair counts, local RMSD).
    """
    ensure_openmm()
    ref = app.PDBFile(reference_pdb)
    mut = app.PDBFile(mutant_pdb)
    lig_top, lig_positions, lig_indices = _extract_ligand_subset(
        ref.topology, ref.positions, ligand_resnames
    )
    if not lig_indices:
        raise ValueError(
            f"No ligand residues {ligand_resnames} found in reference PDB '{reference_pdb}'."
        )

    ref_pos_nm = np.asarray(ref.positions.value_in_unit(unit.nanometer), dtype=float)
    lig_pos_nm = ref_pos_nm[lig_indices]
    lig_centroid = lig_pos_nm.mean(axis=0)

    ref_records = ca_records_from_topology(ref.topology, ref.positions, chain_id)
    mut_records = ca_records_from_topology(mut.topology, mut.positions, chain_id)
    if len(ref_records) < 3 or len(mut_records) < 3:
        raise ValueError("Insufficient CA anchors for protein alignment.")

    ref_seq = "".join(aa for aa, _ in ref_records)
    mut_seq = "".join(aa for aa, _ in mut_records)
    pairs = align_index_pairs(ref_seq, mut_seq)
    if len(pairs) < 3:
        raise ValueError("Insufficient aligned residue pairs for ligand transfer.")

    local_radius_nm = local_radius_ang * 0.1
    ref_points_all = np.asarray([ref_records[i][1] for i, _ in pairs], dtype=float)
    mut_points_all = np.asarray([mut_records[j][1] for _, j in pairs], dtype=float)
    distances = np.linalg.norm(ref_points_all - lig_centroid[None, :], axis=1)
    local_mask = distances <= local_radius_nm

    if int(np.sum(local_mask)) >= 6:
        ref_points = ref_points_all[local_mask]
        mut_points = mut_points_all[local_mask]
        local_pairs = int(np.sum(local_mask))
    else:
        ref_points = ref_points_all
        mut_points = mut_points_all
        local_pairs = len(pairs)

    rot, trans, rmsd_nm = kabsch_fit(ref_points, mut_points)
    transformed_lig = (rot @ lig_pos_nm.T).T + trans
    transformed_lig_qty = [Vec3(*xyz) * unit.nanometer for xyz in transformed_lig]

    modeller = app.Modeller(mut.topology, mut.positions)
    existing_lig_residues = []
    selection = ligand_selection(ligand_resnames)
    for chain in modeller.topology.chains():
        for residue in chain.residues():
            if selection.matches(residue):
                existing_lig_residues.append(residue)
    if existing_lig_residues:
        modeller.delete(existing_lig_residues)

    modeller.add(lig_top, transformed_lig_qty)
    with open(output_pdb, "w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

    return {
        "alignment_pairs": float(len(pairs)),
        "local_alignment_pairs": float(local_pairs),
        "local_alignment_rmsd_ang": float(rmsd_nm * 10.0),
        "ligand_atoms": float(len(lig_indices)),
    }
