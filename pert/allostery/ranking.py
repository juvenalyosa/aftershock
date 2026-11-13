"""MD-based per-residue allostery ranking for one run (validation tier).

Combines, for every residue present in both WT and mutant:

=====================  ======  ==============================================
component              weight  source
=====================  ======  ==============================================
structural shift        0.15   C-alpha displacement between prepared models
dynamics change         0.20   |dRMSF| + 0.25 |dspeed|
contact change          0.20   change of contact probabilities over frames
TE change               0.25   total |dTE| in + out of the residue
relay                   0.15   lies on source -> site routes of positive dTE
target coupling         0.05   positive dTE exchanged with the site
=====================  ======  ==============================================

Outputs residue and region rankings, the WT/MUT contact and TE maps, a figure
and a VMD scene in ``<run_dir>/analysis``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pert.analysis.residue_flow import (
    align_positions_and_velocities,
    build_residue_motion_signals,
    find_top_delta_edges,
    find_top_paths,
    indices_from_mut_residue_keys,
    pairwise_te_matrix,
    relay_scores,
)
from pert.structure.binding_site import read_binding_residue_keys, write_binding_csv
from pert.structure.comparison import aligned_ca_pair_records, infer_mutation_anchor_from_structure
from pert.structure.residues import aa_code_from_residue_key
from pert.utils.io import load_json, write_csv
from pert.utils.numeric import normalize_nonnegative
from pert.utils.paths import optional_run_path, resolve_run_path
from pert.visualization.allostery_plots import plot_ranked_residues
from pert.visualization.vmd_allostery import write_hotspot_vmd


# --- per-residue observables ----------------------------------------------------------


def _sample_frame_indices(n_frames: int, max_frames: Optional[int]) -> np.ndarray:
    """Evenly spaced frame indices, at most ``max_frames`` of them."""
    if max_frames is None or max_frames <= 0 or n_frames <= max_frames:
        return np.arange(n_frames, dtype=int)
    return np.unique(np.linspace(0, n_frames - 1, int(max_frames), dtype=int))


def _rmsf_ang(positions_nm: np.ndarray) -> np.ndarray:
    """Root-mean-square fluctuation per residue (A)."""
    centered = positions_nm - np.mean(positions_nm, axis=0, keepdims=True)
    return 10.0 * np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=0))


def _mean_speed_ang_per_ps(velocities_nmps: np.ndarray) -> np.ndarray:
    """Mean speed per residue (A/ps)."""
    return 10.0 * np.mean(np.linalg.norm(velocities_nmps, axis=2), axis=0)


def _contact_change_score(
    wt_pos_nm: np.ndarray,
    mut_pos_nm: np.ndarray,
    threshold_ang: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change in C-alpha contact probabilities between WT and mutant.

    Returns ``(per-residue mean |delta probability|, WT contact map, mutant contact map)``.
    """
    n_frames = min(int(wt_pos_nm.shape[0]), int(mut_pos_nm.shape[0]))
    n_res = min(int(wt_pos_nm.shape[1]), int(mut_pos_nm.shape[1]))
    if n_frames < 1 or n_res < 2:
        zeros = np.zeros(n_res, dtype=float)
        return zeros, np.zeros((n_res, n_res), dtype=float), np.zeros((n_res, n_res), dtype=float)

    threshold_nm = float(threshold_ang) * 0.1
    wt_counts = np.zeros((n_res, n_res), dtype=float)
    mut_counts = np.zeros((n_res, n_res), dtype=float)
    eye = np.eye(n_res, dtype=bool)

    for frame in range(n_frames):
        wt_xyz = wt_pos_nm[frame, :n_res, :]
        mut_xyz = mut_pos_nm[frame, :n_res, :]
        wt_d = np.linalg.norm(wt_xyz[:, None, :] - wt_xyz[None, :, :], axis=2)
        mut_d = np.linalg.norm(mut_xyz[:, None, :] - mut_xyz[None, :, :], axis=2)
        wt_contact = (wt_d <= threshold_nm) & (~eye)
        mut_contact = (mut_d <= threshold_nm) & (~eye)
        wt_counts += wt_contact.astype(float)
        mut_counts += mut_contact.astype(float)

    wt_prob = wt_counts / float(n_frames)
    mut_prob = mut_counts / float(n_frames)
    delta = mut_prob - wt_prob
    per_residue = np.sum(np.abs(delta), axis=1) / float(max(1, n_res - 1))
    return per_residue, wt_prob, mut_prob


def _explicit_sequence_difference_indices(pair_records: Sequence[Dict[str, object]]) -> List[int]:
    """Paired residues whose amino acid differs between WT and mutant (substitutions)."""
    indices: List[int] = []
    for idx, rec in enumerate(pair_records):
        wt_key = str(rec.get("wt_residue_key", ""))
        mut_key = str(rec.get("mut_residue_key", ""))
        if aa_code_from_residue_key(wt_key) != aa_code_from_residue_key(mut_key):
            indices.append(idx)
    return indices


# --- contiguous regions ------------------------------------------------------------------


def _region_rows(
    residue_rows_sorted_by_index: Sequence[Dict[str, object]],
    region_window: int,
) -> List[Dict[str, object]]:
    """Group consecutive residues of each chain into windows and rank the windows by mean score."""
    if not residue_rows_sorted_by_index:
        return []
    window = max(1, int(region_window))
    out: List[Dict[str, object]] = []
    start = 0
    while start < len(residue_rows_sorted_by_index):
        chain = str(residue_rows_sorted_by_index[start]["chain"])
        end = start
        while end < len(residue_rows_sorted_by_index) and str(residue_rows_sorted_by_index[end]["chain"]) == chain:
            end += 1
        chain_rows = list(residue_rows_sorted_by_index[start:end])
        for chunk_start in range(0, len(chain_rows), window):
            chunk = chain_rows[chunk_start : chunk_start + window]
            if not chunk:
                continue
            scores = np.asarray([float(row["total_score"]) for row in chunk], dtype=float)
            top = max(chunk, key=lambda row: float(row["total_score"]))
            first = str(chunk[0]["residue"])
            last = str(chunk[-1]["residue"])
            out.append(
                {
                    "region": f"{chain}:{first}-{last}" if chain else f"{first}-{last}",
                    "chain": chain,
                    "residue_count": len(chunk),
                    "mean_total_score": float(np.mean(scores)),
                    "max_total_score": float(np.max(scores)),
                    "top_residue": str(top["residue"]),
                    "residues": ",".join(str(row["residue"]) for row in chunk),
                }
            )
        start = end
    out.sort(key=lambda row: (float(row["mean_total_score"]), float(row["max_total_score"])), reverse=True)
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    return out


# --- ranking -----------------------------------------------------------------------------


def run_allostery_ranking(
    run_dir: str,
    ligand_resname: str,
    cutoff_ang: float,
    source_hotspots: int,
    top_edges_n: int,
    top_paths_n: int,
    top_residues: int,
    region_window: int,
    max_frames: Optional[int],
    contact_threshold_ang: float,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Rank residues of one finished run by MD-derived allosteric evidence (see module docstring)."""
    run_dir = os.path.abspath(run_dir)
    summary = load_json(os.path.join(run_dir, "run_summary.json"))
    out_dir = output_dir or os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    wt_pdb = resolve_run_path(run_dir, summary["md"]["wt"]["prepared_pdb"])
    mut_pdb = resolve_run_path(run_dir, summary["md"]["mut"]["prepared_pdb"])
    wt_traj_npz = resolve_run_path(run_dir, summary["md"]["wt"]["traj_npz"])
    mut_traj_npz = resolve_run_path(run_dir, summary["md"]["mut"]["traj_npz"])
    mut_dcd = optional_run_path(run_dir, summary["md"]["mut"].get("traj_dcd"))

    binding_csv = os.path.join(out_dir, "binding_site_residues.csv")
    if not os.path.exists(binding_csv):
        write_binding_csv(
            wt_pdb=wt_pdb,
            mut_pdb=mut_pdb,
            ligand_resname=ligand_resname,
            cutoff_ang=cutoff_ang,
            out_csv=binding_csv,
        )

    pair_records = aligned_ca_pair_records(wt_pdb, mut_pdb)
    wt_idx = np.asarray([int(rec["wt_atom_index"]) for rec in pair_records], dtype=int)
    mut_idx = np.asarray([int(rec["mut_atom_index"]) for rec in pair_records], dtype=int)

    wt_data = np.load(wt_traj_npz)
    mut_data = np.load(mut_traj_npz)
    n_frames = min(int(wt_data["positions"].shape[0]), int(mut_data["positions"].shape[0]))
    frame_idx = _sample_frame_indices(n_frames, max_frames)
    if frame_idx.size < 10:
        raise ValueError("Need at least 10 sampled frames for allostery ranking.")

    wt_pos_raw = np.asarray(wt_data["positions"][frame_idx][:, wt_idx, :], dtype=float)
    wt_vel_raw = np.asarray(wt_data["velocities"][frame_idx][:, wt_idx, :], dtype=float)
    mut_pos_raw = np.asarray(mut_data["positions"][frame_idx][:, mut_idx, :], dtype=float)
    mut_vel_raw = np.asarray(mut_data["velocities"][frame_idx][:, mut_idx, :], dtype=float)

    wt_pos, wt_vel = align_positions_and_velocities(wt_pos_raw, wt_vel_raw)
    mut_pos, mut_vel = align_positions_and_velocities(mut_pos_raw, mut_vel_raw)
    signal_wt = build_residue_motion_signals(wt_pos, wt_vel, velocity_weight=0.35)
    signal_mut = build_residue_motion_signals(mut_pos, mut_vel, velocity_weight=0.35)

    _, hotspot_keys, disp_map_nm = infer_mutation_anchor_from_structure(
        wt_pdb,
        mut_pdb,
        top_n=max(1, int(source_hotspots)),
    )
    binding_keys = read_binding_residue_keys(binding_csv)
    source_idx = indices_from_mut_residue_keys(pair_records, hotspot_keys)
    explicit_source_idx = _explicit_sequence_difference_indices(pair_records)
    source_idx = sorted(set(source_idx) | set(explicit_source_idx))
    target_idx = indices_from_mut_residue_keys(pair_records, binding_keys)
    if not source_idx and pair_records:
        source_idx = [0]
    if not target_idx and len(pair_records) > 1:
        target_idx = [len(pair_records) - 1]
    target_idx = [idx for idx in target_idx if idx not in set(source_idx)] or target_idx

    te_wt = pairwise_te_matrix(signal_wt, lag=1)
    te_mut = pairwise_te_matrix(signal_mut, lag=1)
    te_delta = te_mut - te_wt
    te_delta_pos = np.clip(te_delta, 0.0, None)
    relay = relay_scores(te_delta_pos, source_idx=source_idx, target_idx=target_idx)
    top_edges = find_top_delta_edges(te_wt=te_wt, te_mut=te_mut, pair_records=pair_records, top_n=top_edges_n)
    top_paths = find_top_paths(
        te_delta_pos=te_delta_pos,
        source_idx=source_idx,
        target_idx=target_idx,
        pair_records=pair_records,
        top_n=top_paths_n,
    )

    structural_shift_ang = np.asarray(
        [10.0 * float(disp_map_nm.get(str(rec["mut_residue_key"]), 0.0)) for rec in pair_records],
        dtype=float,
    )
    rmsf_wt = _rmsf_ang(wt_pos)
    rmsf_mut = _rmsf_ang(mut_pos)
    rmsf_delta_abs = np.abs(rmsf_mut - rmsf_wt)
    speed_wt = _mean_speed_ang_per_ps(wt_vel)
    speed_mut = _mean_speed_ang_per_ps(mut_vel)
    speed_delta_abs = np.abs(speed_mut - speed_wt)
    dynamics_raw = rmsf_delta_abs + 0.25 * speed_delta_abs
    contact_change, contact_wt, contact_mut = _contact_change_score(
        wt_pos_nm=wt_pos,
        mut_pos_nm=mut_pos,
        threshold_ang=contact_threshold_ang,
    )
    te_out_gain = np.sum(te_delta_pos, axis=1)
    te_in_gain = np.sum(te_delta_pos, axis=0)
    te_out_abs = np.sum(np.abs(te_delta), axis=1)
    te_in_abs = np.sum(np.abs(te_delta), axis=0)
    target_coupling = np.zeros(len(pair_records), dtype=float)
    if target_idx:
        tgt = np.asarray(target_idx, dtype=int)
        target_coupling = np.sum(te_delta_pos[:, tgt], axis=1) + np.sum(te_delta_pos[tgt, :], axis=0)

    structural_component = normalize_nonnegative(structural_shift_ang)
    dynamics_component = normalize_nonnegative(dynamics_raw)
    contact_component = normalize_nonnegative(contact_change)
    te_component = normalize_nonnegative(te_out_abs + te_in_abs)
    relay_component = normalize_nonnegative(relay)
    target_coupling_component = normalize_nonnegative(target_coupling)

    total_score = (
        0.15 * structural_component
        + 0.20 * dynamics_component
        + 0.20 * contact_component
        + 0.25 * te_component
        + 0.15 * relay_component
        + 0.05 * target_coupling_component
    )

    rows_by_index: List[Dict[str, object]] = []
    source_set = set(source_idx)
    target_set = set(target_idx)
    for idx, rec in enumerate(pair_records):
        row = {
            "index": idx,
            "node_label": str(rec["node_label"]),
            "chain": str(rec["chain_id"]),
            "resid": str(rec["resid"]),
            "residue": str(rec["mut_residue_key"]),
            "wt_residue": str(rec["wt_residue_key"]),
            "is_source_hotspot": idx in source_set,
            "is_target_site": idx in target_set,
            "structural_shift_ang": float(structural_shift_ang[idx]),
            "rmsf_wt_ang": float(rmsf_wt[idx]),
            "rmsf_mut_ang": float(rmsf_mut[idx]),
            "rmsf_delta_abs_ang": float(rmsf_delta_abs[idx]),
            "speed_wt_ang_per_ps": float(speed_wt[idx]),
            "speed_mut_ang_per_ps": float(speed_mut[idx]),
            "speed_delta_abs_ang_per_ps": float(speed_delta_abs[idx]),
            "contact_change": float(contact_change[idx]),
            "te_out_gain": float(te_out_gain[idx]),
            "te_in_gain": float(te_in_gain[idx]),
            "te_out_abs_delta": float(te_out_abs[idx]),
            "te_in_abs_delta": float(te_in_abs[idx]),
            "target_coupling_gain": float(target_coupling[idx]),
            "relay_score": float(relay[idx]),
            "structural_component": float(structural_component[idx]),
            "dynamics_component": float(dynamics_component[idx]),
            "contact_component": float(contact_component[idx]),
            "te_component": float(te_component[idx]),
            "relay_component": float(relay_component[idx]),
            "target_coupling_component": float(target_coupling_component[idx]),
            "total_score": float(total_score[idx]),
            "mut_coord_nm": np.asarray(rec["mut_coord_nm"], dtype=float).tolist(),
        }
        rows_by_index.append(row)

    rows_ranked = sorted(rows_by_index, key=lambda row: float(row["total_score"]), reverse=True)
    for rank, row in enumerate(rows_ranked, start=1):
        row["rank"] = rank
    csv_rows = []
    for row in rows_ranked:
        csv_row = dict(row)
        csv_row.pop("mut_coord_nm", None)
        csv_rows.append(csv_row)

    residue_csv = os.path.join(out_dir, "allostery_residue_scores.csv")
    write_csv(residue_csv, csv_rows)

    residue_json = os.path.join(out_dir, "allostery_residue_scores.json")
    with open(residue_json, "w", encoding="utf-8") as handle:
        json.dump(rows_ranked, handle, indent=2)

    region_csv = os.path.join(out_dir, "allostery_region_scores.csv")
    region_rows = _region_rows(rows_by_index, region_window=region_window)
    write_csv(region_csv, region_rows)

    contact_npz = os.path.join(out_dir, "allostery_contact_maps.npz")
    np.savez_compressed(
        contact_npz,
        node_label=np.asarray([str(row["node_label"]) for row in rows_by_index], dtype=str),
        contact_wt=contact_wt,
        contact_mut=contact_mut,
        contact_delta=contact_mut - contact_wt,
        te_wt=te_wt,
        te_mut=te_mut,
        te_delta=te_delta,
    )

    rank_plot = os.path.join(out_dir, "plot_allostery_residue_rank.png")
    plot_ranked_residues(rows_ranked, rank_plot, top_n=top_residues)

    source_rows = [rows_by_index[i] for i in source_idx if 0 <= i < len(rows_by_index)]
    target_rows = [rows_by_index[i] for i in target_idx if 0 <= i < len(rows_by_index)]
    vmd_script = os.path.join(out_dir, "view_allostery_hotspots.vmd.tcl")
    write_hotspot_vmd(
        mut_pdb=mut_pdb,
        mut_dcd=mut_dcd,
        ligand_resname=ligand_resname,
        rows=rows_ranked,
        source_rows=source_rows,
        target_rows=target_rows,
        out_tcl=vmd_script,
        top_n=top_residues,
    )

    network_json = os.path.join(out_dir, "allostery_network.json")
    network = {
        "run_dir": run_dir,
        "ligand_resname": ligand_resname,
        "binding_cutoff_ang": cutoff_ang,
        "contact_threshold_ang": contact_threshold_ang,
        "sampled_frames": int(frame_idx.size),
        "source_hotspot_count_requested": int(source_hotspots),
        "source_residues": [str(rows_by_index[i]["node_label"]) for i in source_idx if 0 <= i < len(rows_by_index)],
        "target_residues": [str(rows_by_index[i]["node_label"]) for i in target_idx if 0 <= i < len(rows_by_index)],
        "score_weights": {
            "structural_component": 0.15,
            "dynamics_component": 0.20,
            "contact_component": 0.20,
            "te_component": 0.25,
            "relay_component": 0.15,
            "target_coupling_component": 0.05,
        },
        "top_residues": [
            {k: v for k, v in row.items() if k != "mut_coord_nm"}
            for row in rows_ranked[: max(1, int(top_residues))]
        ],
        "top_regions": region_rows[: max(1, min(20, len(region_rows)))],
        "top_delta_edges": top_edges,
        "top_mutation_to_binding_paths": top_paths,
        "outputs": {
            "residue_csv": residue_csv,
            "residue_json": residue_json,
            "region_csv": region_csv,
            "contact_npz": contact_npz,
            "rank_plot": rank_plot,
            "vmd_script": vmd_script,
        },
    }
    with open(network_json, "w", encoding="utf-8") as handle:
        json.dump(network, handle, indent=2)

    manifest = {
        "residue_scores": residue_csv,
        "region_scores": region_csv,
        "network_json": network_json,
        "contact_maps": contact_npz,
        "rank_plot": rank_plot,
        "vmd_script": vmd_script,
    }
    manifest_path = os.path.join(out_dir, "allostery_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest
