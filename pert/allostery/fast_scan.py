"""Fast, MD-free mutation/allostery scanner (screening tier).

Scores a WT/mutant *structure pair* in seconds, in the spirit of FoldX/DDMut
but aimed at the question "does this mutation reach the binding site?".

Per-residue ingredients (C-alpha level, mutant numbering):

* structural shift after global superposition,
* gained/lost contacts in the C-alpha contact network,
* local strain (change of distances to contact neighbours),
* a path term: graph proximity to the mutation sources *and* to the site,
* pocket proximity and a flag for mutation-source residues.

They are normalised and combined with fixed weights into the
``fast_allostery_score``; a global score and a qualitative effect class
(minimal/low/moderate/high) summarise the variant.

Residue correspondence handles SNP, multi-SNP, INS, DEL, DELINS and DUP either
exactly (when mutation specs are given) or by sequence alignment.

This is a hypothesis generator: validate prioritised variants with MD.
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pert.sequence.alignment import needleman_wunsch_affine
from pert.sequence.mutations import mutation_reference_map
from pert.structure.binding_site import binding_site_residues
from pert.structure.geometry import kabsch_align
from pert.structure.residues import protein_ca_records
from pert.utils.io import load_json, write_csv
from pert.utils.numeric import normalize_nonnegative
from pert.utils.paths import resolve_run_path
from pert.visualization.allostery_plots import plot_fast_scores
from pert.visualization.vmd_allostery import write_fast_scan_vmd


# --- contact graph primitives -----------------------------------------------------


def _distance_matrix(coords: np.ndarray) -> np.ndarray:
    """All-against-all Euclidean distances."""
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


def _bfs_distances(adj: np.ndarray, sources: Sequence[int]) -> np.ndarray:
    """Number of contact-graph steps from the nearest source to every node (inf if unreachable)."""
    n = int(adj.shape[0])
    dist = np.full(n, np.inf, dtype=float)
    q: deque[int] = deque()
    for src in sources:
        if 0 <= int(src) < n and not np.isfinite(dist[int(src)]):
            dist[int(src)] = 0.0
            q.append(int(src))
    while q:
        cur = q.popleft()
        for nxt in np.where(adj[cur])[0].tolist():
            if not np.isfinite(dist[int(nxt)]):
                dist[int(nxt)] = dist[cur] + 1.0
                q.append(int(nxt))
    return dist


# --- WT <-> MUT residue correspondence -----------------------------------------------


def _make_pairs_with_indels(wt_pdb: str, mut_pdb: str, ligand=None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Residue rows of the mutant with their WT partner, inferred by sequence alignment.

    Also returns the WT residues that were deleted.  Rows flag insertions,
    substitutions and the neighbours of deletions.
    """
    wt_records = protein_ca_records(wt_pdb, ligand)
    mut_records = protein_ca_records(mut_pdb, ligand)
    if len(wt_records) < 3 or len(mut_records) < 3:
        raise ValueError("Need at least 3 CA residues in both WT and MUT.")

    wt_seq = "".join(str(rec["aa1"]) for rec in wt_records)
    mut_seq = "".join(str(rec["aa1"]) for rec in mut_records)
    aln_wt, aln_mut = needleman_wunsch_affine(wt_seq, mut_seq)

    wt_idx = 0
    mut_idx = 0
    rows: List[Dict[str, Any]] = []
    deleted_wt_residues: List[str] = []
    deletion_neighbor_indices: List[int] = []

    for aa_w, aa_m in zip(aln_wt, aln_mut):
        wt_rec = None
        mut_rec = None
        if aa_w != "-":
            wt_rec = wt_records[wt_idx]
            wt_idx += 1
        if aa_m != "-":
            mut_rec = mut_records[mut_idx]
            mut_idx += 1

        if mut_rec is None:
            if wt_rec is not None:
                deleted_wt_residues.append(str(wt_rec["residue_key"]))
                if rows:
                    deletion_neighbor_indices.append(len(rows) - 1)
                if mut_idx < len(mut_records):
                    deletion_neighbor_indices.append(len(rows))
            continue

        chain_id = str(mut_rec["chain_id"]).strip()
        resid = str(mut_rec["resid"]).strip()
        mut_key = str(mut_rec["residue_key"]).strip()
        wt_key = str(wt_rec["residue_key"]).strip() if wt_rec is not None else ""
        node_label = f"{chain_id}:{mut_key}" if chain_id else mut_key
        rows.append(
            {
                "index": len(rows),
                "node_label": node_label,
                "chain": chain_id,
                "resid": resid,
                "residue": mut_key,
                "wt_residue": wt_key,
                "aa_mut": str(mut_rec["aa1"]),
                "aa_wt": str(wt_rec["aa1"]) if wt_rec is not None else "-",
                "is_insertion": wt_rec is None,
                "is_substitution": wt_rec is not None and str(wt_rec["aa1"]) != str(mut_rec["aa1"]),
                "is_deletion_neighbor": False,
                "mut_coord_nm": np.asarray(mut_rec["coord"], dtype=float),
                "wt_coord_nm": None if wt_rec is None else np.asarray(wt_rec["coord"], dtype=float),
            }
        )

    for idx in deletion_neighbor_indices:
        if 0 <= idx < len(rows):
            rows[idx]["is_deletion_neighbor"] = True
    return rows, sorted(set(deleted_wt_residues))


def _reference_start_from_records(records: Sequence[Dict[str, Any]]) -> int:
    """Residue number of the first WT residue (1 if it is not an integer)."""
    if not records:
        return 1
    try:
        return int(str(records[0]["resid"]).strip())
    except ValueError:
        return 1


def _make_pairs_from_mutation_specs(
    wt_pdb: str,
    mut_pdb: str,
    mutation_specs: Sequence[str],
    ligand=None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Residue rows of the mutant with their WT partner, derived exactly from the mutation specs.

    Preferred over alignment for compound variants: every mutant residue knows
    which reference position and which mutation it came from.  Fails if the
    specs do not reproduce the sequence of the mutant structure.
    """
    wt_records = protein_ca_records(wt_pdb, ligand)
    mut_records = protein_ca_records(mut_pdb, ligand)
    if len(wt_records) < 3 or len(mut_records) < 3:
        raise ValueError("Need at least 3 CA residues in both WT and MUT.")

    wt_seq = "".join(str(rec["aa1"]) for rec in wt_records)
    mut_seq = "".join(str(rec["aa1"]) for rec in mut_records)
    ref_start = _reference_start_from_records(wt_records)
    ref_map = mutation_reference_map(wt_seq, mutation_specs, start_pos=ref_start)
    expected_mut_seq = str(ref_map["mutated_sequence"])
    if expected_mut_seq != mut_seq:
        raise ValueError(
            "Mutation specs do not match mutant structure sequence. "
            f"Expected {expected_mut_seq}, found {mut_seq}."
        )

    wt_by_ref_pos = {ref_start + idx: rec for idx, rec in enumerate(wt_records)}
    deleted_ref_positions = [int(pos) for pos in ref_map["deleted_ref_positions"]]
    deleted_set = set(deleted_ref_positions)
    deleted_wt_residues = [
        str(wt_by_ref_pos[pos]["residue_key"])
        for pos in deleted_ref_positions
        if pos in wt_by_ref_pos
    ]

    rows: List[Dict[str, Any]] = []
    for idx, (mut_rec, map_row) in enumerate(zip(mut_records, ref_map["residue_map"])):
        ref_pos = map_row.get("ref_pos")
        wt_rec = wt_by_ref_pos.get(int(ref_pos)) if ref_pos is not None else None
        chain_id = str(mut_rec["chain_id"]).strip()
        resid = str(mut_rec["resid"]).strip()
        mut_key = str(mut_rec["residue_key"]).strip()
        wt_key = str(wt_rec["residue_key"]).strip() if wt_rec is not None else ""
        node_label = f"{chain_id}:{mut_key}" if chain_id else mut_key
        status = str(map_row.get("status", "unknown"))
        is_deleted_neighbor = False
        if ref_pos is not None:
            ref_pos_int = int(ref_pos)
            is_deleted_neighbor = (ref_pos_int - 1 in deleted_set) or (ref_pos_int + 1 in deleted_set)
        rows.append(
            {
                "index": idx,
                "node_label": node_label,
                "chain": chain_id,
                "resid": resid,
                "residue": mut_key,
                "wt_residue": wt_key,
                "aa_mut": str(mut_rec["aa1"]),
                "aa_wt": str(wt_rec["aa1"]) if wt_rec is not None else "-",
                "is_insertion": status in {"insertion", "duplication"} or wt_rec is None,
                "is_substitution": status in {"substitution", "delins"},
                "is_deletion_neighbor": is_deleted_neighbor or status == "delins",
                "is_mutation_source": bool(map_row.get("is_source", False)),
                "mutation_status": status,
                "mutation_spec": str(map_row.get("mutation", "")),
                "mut_coord_nm": np.asarray(mut_rec["coord"], dtype=float),
                "wt_coord_nm": None if wt_rec is None else np.asarray(wt_rec["coord"], dtype=float),
            }
        )
    return rows, sorted(set(deleted_wt_residues))


def _align_mutant_coords(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Superpose the mutant on the WT using paired C-alphas; returns aligned mutant coords, WT coords, RMSD (A)."""
    paired = [row for row in rows if row["wt_coord_nm"] is not None]
    if len(paired) < 3:
        raise ValueError("Need at least 3 aligned WT/MUT residue pairs for structural alignment.")
    wt = np.stack([np.asarray(row["wt_coord_nm"], dtype=float) for row in paired], axis=0)
    mut = np.stack([np.asarray(row["mut_coord_nm"], dtype=float) for row in paired], axis=0)
    rot, trans = kabsch_align(mut, wt)
    all_mut = np.stack([np.asarray(row["mut_coord_nm"], dtype=float) for row in rows], axis=0)
    all_mut_aligned = (rot @ all_mut.T).T + trans[None, :]
    paired_aligned = (rot @ mut.T).T + trans[None, :]
    rmsd_ang = float(np.sqrt(np.mean(np.sum((paired_aligned - wt) ** 2, axis=1))) * 10.0)
    return all_mut_aligned, wt, rmsd_ang


def _binding_target_indices(
    rows: Sequence[Dict[str, Any]],
    wt_pdb: str,
    mut_pdb: str,
    ligand_resname: str,
    cutoff_ang: float,
) -> Tuple[List[int], List[str]]:
    """Row indices of the residues that line the ligand in WT or mutant, plus their keys."""
    wt_site = binding_site_residues(wt_pdb, ligand_resname, cutoff_ang)
    mut_site = binding_site_residues(mut_pdb, ligand_resname, cutoff_ang)
    site_keys = {row[1] for row in wt_site} | {row[1] for row in mut_site}
    indices: List[int] = []
    for idx, row in enumerate(rows):
        if str(row["residue"]) in site_keys or str(row["wt_residue"]) in site_keys:
            indices.append(idx)
    return sorted(set(indices)), sorted(site_keys)


# --- scoring ---------------------------------------------------------------------------


def run_fast_scan(
    wt_pdb: str,
    mut_pdb: str,
    ligand_resname: str,
    output_dir: str,
    mutation_label: str = "mutation",
    mutation_specs: Optional[Sequence[str]] = None,
    cutoff_ang: float = 6.0,
    contact_threshold_ang: float = 8.0,
    source_hotspots: int = 8,
    top_residues: int = 30,
) -> Dict[str, Any]:
    """Score a WT/mutant structure pair without MD (see module docstring).

    Score = 0.25 structure + 0.20 contacts + 0.20 strain + 0.20 path +
    0.10 pocket + 0.05 source (each component normalised to [0, 1]).  The global
    score mixes the mean of the top-10 residues (65%) with the mean over the
    binding-site residues (35%).

    Writes residue CSV/JSON, a summary JSON, a figure and a VMD scene to
    ``output_dir`` and returns the summary dict.
    """
    wt_pdb = os.path.abspath(wt_pdb)
    mut_pdb = os.path.abspath(mut_pdb)
    os.makedirs(output_dir, exist_ok=True)

    if mutation_specs:
        rows, deleted_wt_residues = _make_pairs_from_mutation_specs(wt_pdb, mut_pdb, mutation_specs, ligand_resname)
    else:
        rows, deleted_wt_residues = _make_pairs_with_indels(wt_pdb, mut_pdb, ligand_resname)
    mut_aligned, paired_wt_coords, ca_rmsd_ang = _align_mutant_coords(rows)
    mut_coords = np.stack([np.asarray(row["mut_coord_nm"], dtype=float) for row in rows], axis=0)
    mut_dist = _distance_matrix(mut_coords)
    mut_adj = (mut_dist <= contact_threshold_ang * 0.1) & (~np.eye(len(rows), dtype=bool))

    wt_coords_full = np.full_like(mut_aligned, np.nan)
    paired_indices = []
    for idx, row in enumerate(rows):
        if row["wt_coord_nm"] is not None:
            wt_coords_full[idx] = np.asarray(row["wt_coord_nm"], dtype=float)
            paired_indices.append(idx)
    structural_shift = np.zeros(len(rows), dtype=float)
    paired_mask = np.isfinite(wt_coords_full[:, 0])
    if np.any(paired_mask):
        structural_shift[paired_mask] = 10.0 * np.linalg.norm(
            mut_aligned[paired_mask] - wt_coords_full[paired_mask],
            axis=1,
        )
        insertion_default = max(float(np.percentile(structural_shift[paired_mask], 90.0)), 1.0)
    else:
        insertion_default = 1.0
    for idx, row in enumerate(rows):
        if row["wt_coord_nm"] is None:
            nearest = np.min(np.linalg.norm(mut_aligned[idx][None, :] - paired_wt_coords, axis=1))
            structural_shift[idx] = max(float(nearest * 10.0), insertion_default)

    wt_dist = np.full((len(rows), len(rows)), np.nan, dtype=float)
    valid = np.where(paired_mask)[0]
    if valid.size > 1:
        valid_wt = wt_coords_full[valid]
        wt_d_valid = _distance_matrix(valid_wt)
        wt_dist[np.ix_(valid, valid)] = wt_d_valid

    threshold_nm = contact_threshold_ang * 0.1
    wt_contact = np.isfinite(wt_dist) & (wt_dist <= threshold_nm) & (~np.eye(len(rows), dtype=bool))
    mut_contact = mut_adj
    contact_gain = np.sum(mut_contact & (~wt_contact), axis=1) / float(max(1, len(rows) - 1))
    contact_loss = np.sum(wt_contact & (~mut_contact), axis=1) / float(max(1, len(rows) - 1))
    contact_change = contact_gain + contact_loss

    union_contact = wt_contact | mut_contact
    strain = np.zeros(len(rows), dtype=float)
    abs_delta_d = np.abs(mut_dist - wt_dist)
    for idx in range(len(rows)):
        mask = union_contact[idx] & np.isfinite(abs_delta_d[idx])
        if np.any(mask):
            strain[idx] = 10.0 * float(np.mean(abs_delta_d[idx, mask]))
        elif rows[idx]["wt_coord_nm"] is None:
            strain[idx] = structural_shift[idx]

    target_idx, binding_keys = _binding_target_indices(
        rows=rows,
        wt_pdb=wt_pdb,
        mut_pdb=mut_pdb,
        ligand_resname=ligand_resname,
        cutoff_ang=cutoff_ang,
    )
    explicit_sources = [
        idx
        for idx, row in enumerate(rows)
        if bool(row.get("is_mutation_source", False))
        or bool(row["is_insertion"])
        or bool(row["is_substitution"])
        or bool(row["is_deletion_neighbor"])
    ]
    shifted_sources = np.argsort(-structural_shift)[: max(1, int(source_hotspots))].astype(int).tolist()
    source_idx = sorted(set(explicit_sources) | set(shifted_sources))
    if not source_idx:
        source_idx = [int(np.argmax(structural_shift))]
    if not target_idx:
        target_idx = [int(np.argmin(_bfs_distances(mut_adj, source_idx)))] if len(rows) else []

    dist_source = _bfs_distances(mut_adj, source_idx)
    dist_target = _bfs_distances(mut_adj, target_idx)
    graph_scale = 3.0
    path_bridge = np.exp(-np.where(np.isfinite(dist_source), dist_source, 99.0) / graph_scale) * np.exp(
        -np.where(np.isfinite(dist_target), dist_target, 99.0) / graph_scale
    )
    pocket_component = np.exp(-np.where(np.isfinite(dist_target), dist_target, 99.0) / graph_scale)
    source_component = np.zeros(len(rows), dtype=float)
    source_component[source_idx] = 1.0

    structural_component = normalize_nonnegative(structural_shift)
    contact_component = normalize_nonnegative(contact_change)
    strain_component = normalize_nonnegative(strain)
    path_component = normalize_nonnegative(path_bridge)
    pocket_component = normalize_nonnegative(pocket_component)

    fast_score = (
        0.25 * structural_component
        + 0.20 * contact_component
        + 0.20 * strain_component
        + 0.20 * path_component
        + 0.10 * pocket_component
        + 0.05 * source_component
    )
    destabilization_proxy = 0.45 * normalize_nonnegative(contact_loss) + 0.35 * strain_component + 0.20 * structural_component

    output_rows: List[Dict[str, Any]] = []
    source_set = set(source_idx)
    target_set = set(target_idx)
    for idx, row in enumerate(rows):
        out = {
            "index": idx,
            "node_label": row["node_label"],
            "chain": row["chain"],
            "resid": row["resid"],
            "residue": row["residue"],
            "wt_residue": row["wt_residue"],
            "aa_wt": row["aa_wt"],
            "aa_mut": row["aa_mut"],
            "is_source": idx in source_set,
            "is_target_site": idx in target_set,
            "is_insertion": bool(row["is_insertion"]),
            "is_substitution": bool(row["is_substitution"]),
            "is_deletion_neighbor": bool(row["is_deletion_neighbor"]),
            "is_mutation_source": bool(row.get("is_mutation_source", False)),
            "mutation_status": str(row.get("mutation_status", "")),
            "mutation_spec": str(row.get("mutation_spec", "")),
            "structural_shift_ang": float(structural_shift[idx]),
            "contact_gain_fraction": float(contact_gain[idx]),
            "contact_loss_fraction": float(contact_loss[idx]),
            "contact_change_fraction": float(contact_change[idx]),
            "local_strain_ang": float(strain[idx]),
            "graph_distance_from_source": None if not np.isfinite(dist_source[idx]) else float(dist_source[idx]),
            "graph_distance_to_target": None if not np.isfinite(dist_target[idx]) else float(dist_target[idx]),
            "structural_component": float(structural_component[idx]),
            "contact_component": float(contact_component[idx]),
            "strain_component": float(strain_component[idx]),
            "path_component": float(path_component[idx]),
            "pocket_component": float(pocket_component[idx]),
            "source_component": float(source_component[idx]),
            "destabilization_proxy": float(destabilization_proxy[idx]),
            "fast_allostery_score": float(fast_score[idx]),
            "mut_coord_nm": np.asarray(row["mut_coord_nm"], dtype=float).tolist(),
        }
        output_rows.append(out)

    ranked = sorted(output_rows, key=lambda r: float(r["fast_allostery_score"]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    csv_rows = []
    for row in ranked:
        csv_row = dict(row)
        csv_row.pop("mut_coord_nm", None)
        csv_rows.append(csv_row)

    residue_csv = os.path.join(output_dir, "fast_allostery_residue_scores.csv")
    residue_json = os.path.join(output_dir, "fast_allostery_residue_scores.json")
    summary_json = os.path.join(output_dir, "fast_allostery_summary.json")
    plot_png = os.path.join(output_dir, "plot_fast_allostery_rank.png")
    vmd_tcl = os.path.join(output_dir, "view_fast_allostery.vmd.tcl")
    write_csv(residue_csv, csv_rows)
    with open(residue_json, "w", encoding="utf-8") as handle:
        json.dump(ranked, handle, indent=2)
    plot_fast_scores(ranked, plot_png, top_n=top_residues)

    source_rows = [output_rows[i] for i in source_idx if 0 <= i < len(output_rows)]
    target_rows = [output_rows[i] for i in target_idx if 0 <= i < len(output_rows)]
    write_fast_scan_vmd(
        mut_pdb=mut_pdb,
        ligand_resname=ligand_resname,
        rows_ranked=ranked,
        source_rows=source_rows,
        target_rows=target_rows,
        out_tcl=vmd_tcl,
        top_n=top_residues,
    )

    top_scores = np.asarray([float(row["fast_allostery_score"]) for row in ranked[: min(10, len(ranked))]], dtype=float)
    target_scores = np.asarray([float(output_rows[i]["fast_allostery_score"]) for i in target_idx], dtype=float) if target_idx else np.asarray([], dtype=float)
    global_score = float(0.65 * (np.mean(top_scores) if top_scores.size else 0.0) + 0.35 * (np.mean(target_scores) if target_scores.size else 0.0))
    if global_score >= 0.55:
        predicted_effect = "high"
    elif global_score >= 0.35:
        predicted_effect = "moderate"
    elif global_score >= 0.20:
        predicted_effect = "low"
    else:
        predicted_effect = "minimal"

    summary = {
        "mutation_label": mutation_label,
        "wt_pdb": wt_pdb,
        "mut_pdb": mut_pdb,
        "ligand_resname": ligand_resname,
        "mutation_specs": list(mutation_specs or []),
        "binding_cutoff_ang": cutoff_ang,
        "contact_threshold_ang": contact_threshold_ang,
        "deleted_wt_residues": deleted_wt_residues,
        "mutation_source_residues": [
            row["node_label"] for row in output_rows if bool(row.get("is_mutation_source", False))
        ],
        "source_residues": [output_rows[i]["node_label"] for i in source_idx if 0 <= i < len(output_rows)],
        "target_residues": [output_rows[i]["node_label"] for i in target_idx if 0 <= i < len(output_rows)],
        "aligned_ca_rmsd_ang": ca_rmsd_ang,
        "global_fast_allostery_score": global_score,
        "predicted_allosteric_effect": predicted_effect,
        "mean_top10_destabilization_proxy": float(np.mean([float(row["destabilization_proxy"]) for row in ranked[: min(10, len(ranked))]])),
        "top_residues": [{k: v for k, v in row.items() if k != "mut_coord_nm"} for row in ranked[: max(1, int(top_residues))]],
        "outputs": {
            "residue_csv": residue_csv,
            "residue_json": residue_json,
            "summary_json": summary_json,
            "rank_plot": plot_png,
            "vmd_script": vmd_tcl,
        },
        "method_note": (
            "Fast structural/contact/allostery proxy. Use for screening and prioritization; "
            "validate prioritized designs with replicate MD or experimental data."
        ),
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def infer_structures_from_run_dir(run_dir: str) -> Tuple[str, str, str]:
    """``(wt_pdb, mut_pdb, label)`` of the ligand-containing structures of a finished run."""
    run_dir = os.path.abspath(run_dir)
    summary = load_json(os.path.join(run_dir, "run_summary.json"))
    wt_pdb = resolve_run_path(run_dir, summary["structures"]["wt_with_ligand"])
    mut_pdb = resolve_run_path(run_dir, summary["structures"]["mutant_with_ligand"])
    label = os.path.basename(os.path.dirname(run_dir)) + "/" + os.path.basename(run_dir)
    return wt_pdb, mut_pdb, label
