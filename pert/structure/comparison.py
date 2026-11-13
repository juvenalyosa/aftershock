"""Residue-by-residue correspondence between a WT and a mutant structure.

Sequences are aligned with affine-gap Needleman-Wunsch, so the mapping stays
valid for insertions, deletions and duplications.  Only residues present in
both structures become "pair records"; they are the nodes of the residue-level
dynamic analyses.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pert.sequence.alignment import needleman_wunsch_affine
from pert.structure.geometry import kabsch_align
from pert.structure.residues import protein_ca_records


def aligned_ca_pair_records(wt_pdb: str, mut_pdb: str) -> List[Dict[str, object]]:
    """Pair WT and mutant residues through sequence alignment.

    Returns one record per residue present in *both* structures, with the CA atom
    index in each topology, so the same residue can be followed in the WT and the
    mutant trajectories.  Inserted/deleted residues are left out.
    """
    wt_records = protein_ca_records(wt_pdb)
    mut_records = protein_ca_records(mut_pdb)
    if len(wt_records) < 3 or len(mut_records) < 3:
        raise ValueError("Need at least 3 CA residues in both WT and MUT.")

    wt_seq = "".join(str(rec["aa1"]) for rec in wt_records)
    mut_seq = "".join(str(rec["aa1"]) for rec in mut_records)
    aln_wt, aln_mut = needleman_wunsch_affine(wt_seq, mut_seq)

    wt_idx = 0
    mut_idx = 0
    out: List[Dict[str, object]] = []
    for aa_w, aa_m in zip(aln_wt, aln_mut):
        iw: Optional[int] = None
        im: Optional[int] = None
        if aa_w != "-":
            iw = wt_idx
            wt_idx += 1
        if aa_m != "-":
            im = mut_idx
            mut_idx += 1
        if iw is None or im is None:
            continue
        wt_rec = wt_records[iw]
        mut_rec = mut_records[im]
        chain_id = str(mut_rec["chain_id"]).strip()
        resid = str(mut_rec["resid"]).strip()
        mut_reskey = str(mut_rec["residue_key"]).strip()
        node_label = f"{chain_id}:{mut_reskey}" if chain_id else mut_reskey
        out.append(
            {
                "node_label": node_label,
                "chain_id": chain_id,
                "resid": resid,
                "mut_residue_key": mut_reskey,
                "wt_residue_key": str(wt_rec["residue_key"]).strip(),
                "mut_atom_index": int(mut_rec["atom_index"]),
                "wt_atom_index": int(wt_rec["atom_index"]),
                "mut_coord_nm": np.asarray(mut_rec["coord"], dtype=float),
            }
        )
    if len(out) < 4:
        raise ValueError("Insufficient aligned CA pairs for residue-flow analysis.")
    return out


def infer_mutation_anchor_from_structure(
    wt_pdb: str,
    mut_pdb: str,
    top_n: int = 3,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """Locate where the mutation perturbs the structure.

    After superposing the aligned C-alphas, residues are ranked by displacement.
    Returns ``(centre of the top_n residues, their keys, displacement per residue key)``.
    The top residues are used as *sources* of the perturbation.
    """
    wt_records = protein_ca_records(wt_pdb)
    mut_records = protein_ca_records(mut_pdb)
    if len(wt_records) < 3 or len(mut_records) < 3:
        raise ValueError("Need at least 3 CA records in WT and MUT for mutation anchor inference.")

    wt_seq = "".join(str(rec["aa1"]) for rec in wt_records)
    mut_seq = "".join(str(rec["aa1"]) for rec in mut_records)
    aln_wt, aln_mut = needleman_wunsch_affine(wt_seq, mut_seq)

    wt_idx = 0
    mut_idx = 0
    wt_pts: List[np.ndarray] = []
    mut_pts: List[np.ndarray] = []
    mut_keys: List[str] = []
    mut_coords: List[np.ndarray] = []

    for aa_w, aa_m in zip(aln_wt, aln_mut):
        i_w: Optional[int] = None
        i_m: Optional[int] = None
        if aa_w != "-":
            i_w = wt_idx
            wt_idx += 1
        if aa_m != "-":
            i_m = mut_idx
            mut_idx += 1
        if i_w is None or i_m is None:
            continue
        wt_rec = wt_records[i_w]
        mut_rec = mut_records[i_m]
        wt_pts.append(np.asarray(wt_rec["coord"], dtype=float))
        mut_pts.append(np.asarray(mut_rec["coord"], dtype=float))
        mut_keys.append(str(mut_rec["residue_key"]))
        mut_coords.append(np.asarray(mut_rec["coord"], dtype=float))

    if len(wt_pts) < 3:
        raise ValueError("Insufficient aligned CA pairs for mutation anchor inference.")

    wt_arr = np.stack(wt_pts, axis=0)
    mut_arr = np.stack(mut_pts, axis=0)
    rot, trans = kabsch_align(mut_arr, wt_arr)
    mut_aligned = (rot @ mut_arr.T).T + trans[None, :]
    disp = np.linalg.norm(mut_aligned - wt_arr, axis=1)

    rank = np.argsort(-disp)
    keep = rank[: max(1, min(top_n, len(rank)))]
    hotspot_keys = [mut_keys[i] for i in keep]
    hotspot_coords = np.stack([mut_coords[i] for i in keep], axis=0)
    center = np.mean(hotspot_coords, axis=0)
    disp_map = {mut_keys[i]: float(disp[i]) for i in range(len(mut_keys))}
    return center, hotspot_keys, disp_map
