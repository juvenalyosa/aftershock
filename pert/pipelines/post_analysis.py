"""Post-processing of a finished run (everything here only needs ``run_dir``).

* ``run_standard_analysis``: standard figures, binding-site table,
  residue-flow network and VMD scenes;
* ``run_diagnostic_analysis``: convergence/time-course figures and the
  interpretation summary;
* ``run_residue_flow_analysis``: the residue-flow step on its own.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from pert.analysis.residue_flow import (
    align_positions_and_velocities,
    build_residue_motion_signals,
    find_top_delta_edges,
    find_top_paths,
    flow_score,
    pairwise_te_matrix,
    relay_scores,
    source_target_indices,
    windowed_flow_timeseries,
)
from pert.reporting.interpretation import write_interpretation_summary
from pert.structure.binding_site import write_binding_csv
from pert.structure.comparison import aligned_ca_pair_records
from pert.utils.io import load_json
from pert.utils.paths import resolve_run_path
from pert.visualization.diagnostic_plots import (
    plot_basin_convergence,
    plot_binding_site_shift,
    plot_information_dashboard,
    plot_latent_timecourse,
)
from pert.visualization.run_plots import (
    plot_basin_ddg,
    plot_flow_timeseries,
    plot_latent_trajectory,
    plot_susceptibility_quiver,
    plot_te,
)
from pert.visualization.vmd_comparison import write_vmd_comparison_script
from pert.visualization.vmd_flow import write_flow_network_vmd_script, write_mut_flow_vmd_script


def run_residue_flow_analysis(
    wt_pdb: str,
    mut_pdb: str,
    wt_traj_npz: str,
    mut_traj_npz: str,
    mut_dcd: Optional[str],
    binding_csv: str,
    ligand_resname: str,
    output_dir: str,
) -> Dict[str, str]:
    """Residue-flow analysis of a finished run.

    Writes ``residue_flow_graph.npz``, ``flow_summary.json``,
    ``plot_flow_timeseries.png`` and ``view_flow_network.vmd.tcl`` to ``output_dir``.
    """
    pair_records = aligned_ca_pair_records(wt_pdb, mut_pdb)
    wt_idx = np.asarray([int(rec["wt_atom_index"]) for rec in pair_records], dtype=int)
    mut_idx = np.asarray([int(rec["mut_atom_index"]) for rec in pair_records], dtype=int)

    wt_data = np.load(wt_traj_npz)
    mut_data = np.load(mut_traj_npz)
    n = min(int(wt_data["positions"].shape[0]), int(mut_data["positions"].shape[0]))
    if n < 10:
        raise ValueError("Not enough frames for residue-flow analysis.")

    wt_pos_raw = np.asarray(wt_data["positions"][:n, wt_idx, :], dtype=float)
    wt_vel_raw = np.asarray(wt_data["velocities"][:n, wt_idx, :], dtype=float)
    mut_pos_raw = np.asarray(mut_data["positions"][:n, mut_idx, :], dtype=float)
    mut_vel_raw = np.asarray(mut_data["velocities"][:n, mut_idx, :], dtype=float)

    wt_pos, wt_vel = align_positions_and_velocities(wt_pos_raw, wt_vel_raw)
    mut_pos, mut_vel = align_positions_and_velocities(mut_pos_raw, mut_vel_raw)

    signal_wt = build_residue_motion_signals(wt_pos, wt_vel, velocity_weight=0.35)
    signal_mut = build_residue_motion_signals(mut_pos, mut_vel, velocity_weight=0.35)

    source_idx, target_idx = source_target_indices(pair_records, wt_pdb=wt_pdb, mut_pdb=mut_pdb, binding_csv=binding_csv)
    te_wt = pairwise_te_matrix(signal_wt, lag=1)
    te_mut = pairwise_te_matrix(signal_mut, lag=1)
    te_delta = te_mut - te_wt
    te_delta_pos = np.clip(te_delta, 0.0, None)

    relay_score = relay_scores(te_delta_pos, source_idx=source_idx, target_idx=target_idx)
    top_edges = find_top_delta_edges(te_wt=te_wt, te_mut=te_mut, pair_records=pair_records, top_n=28)
    top_paths = find_top_paths(te_delta_pos=te_delta_pos, source_idx=source_idx, target_idx=target_idx, pair_records=pair_records, top_n=14)

    if "time_ps" in wt_data.files:
        time_ps = np.asarray(wt_data["time_ps"][:n], dtype=float)
    else:
        dt_ps = 1.0
        if "dt_ps" in wt_data.files:
            dt_ps = float(np.asarray(wt_data["dt_ps"], dtype=float).reshape(-1)[0])
        time_ps = np.arange(n, dtype=float) * dt_ps
    t_win, flow_wt, flow_mut, flow_delta, window_cfg = windowed_flow_timeseries(
        signal_wt=signal_wt,
        signal_mut=signal_mut,
        time_ps=time_ps,
        source_idx=source_idx,
        target_idx=target_idx,
    )

    total_wt = flow_score(te_wt, source_idx=source_idx, target_idx=target_idx)
    total_mut = flow_score(te_mut, source_idx=source_idx, target_idx=target_idx)

    graph_npz = os.path.join(output_dir, "residue_flow_graph.npz")
    np.savez_compressed(
        graph_npz,
        node_label=np.asarray([str(rec["node_label"]) for rec in pair_records], dtype=str),
        chain_id=np.asarray([str(rec["chain_id"]) for rec in pair_records], dtype=str),
        resid=np.asarray([str(rec["resid"]) for rec in pair_records], dtype=str),
        mut_residue_key=np.asarray([str(rec["mut_residue_key"]) for rec in pair_records], dtype=str),
        source_index=np.asarray(source_idx, dtype=int),
        target_index=np.asarray(target_idx, dtype=int),
        te_wt=te_wt,
        te_mut=te_mut,
        te_delta=te_delta,
        relay_score=relay_score,
        flow_time_ps=t_win,
        flow_wt=flow_wt,
        flow_mut=flow_mut,
        flow_delta=flow_delta,
    )

    flow_summary = {
        "nodes": int(len(pair_records)),
        "frames": int(n),
        "source_residues": [str(pair_records[i]["node_label"]) for i in source_idx],
        "target_residues": [str(pair_records[i]["node_label"]) for i in target_idx],
        "window_config": window_cfg,
        "total_flow_wt": float(total_wt),
        "total_flow_mut": float(total_mut),
        "total_flow_delta_mut_minus_wt": float(total_mut - total_wt),
        "top_delta_edges": top_edges,
        "top_mutation_to_binding_paths": top_paths,
        "time_series": {
            "time_ps": [float(v) for v in t_win.tolist()],
            "flow_wt": [float(v) for v in flow_wt.tolist()],
            "flow_mut": [float(v) for v in flow_mut.tolist()],
            "flow_delta": [float(v) for v in flow_delta.tolist()],
        },
    }
    summary_path = os.path.join(output_dir, "flow_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(flow_summary, handle, indent=2)

    flow_plot = os.path.join(output_dir, "plot_flow_timeseries.png")
    plot_flow_timeseries(t_win, flow_wt, flow_mut, flow_delta, out_png=flow_plot)

    flow_vmd = write_flow_network_vmd_script(
        mut_pdb=mut_pdb,
        mut_dcd=mut_dcd,
        ligand_resname=ligand_resname,
        pair_records=pair_records,
        top_edges=top_edges,
        source_idx=source_idx,
        target_idx=target_idx,
        relay_scores=relay_score,
        flow_delta_series=flow_delta,
        top_paths=top_paths,
        output_tcl=os.path.join(output_dir, "view_flow_network.vmd.tcl"),
    )

    return {
        "graph_npz": graph_npz,
        "summary_json": summary_path,
        "plot_png": flow_plot,
        "vmd_script": flow_vmd,
    }


def run_standard_analysis(run_dir: str, ligand_resname: str = "ZN", cutoff_ang: float = 6.0) -> Dict[str, List[str]]:
    """Standard post-processing of one finished run.

    Writes into ``<run_dir>/analysis``: the four region-level figures, the
    binding-site table, the residue-flow network artefacts and three VMD scenes,
    plus ``plot_manifest.json`` (returned).
    """
    summary = load_json(os.path.join(run_dir, "run_summary.json"))
    analysis = summary["analysis"]
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    plot_basin_ddg(analysis, os.path.join(analysis_dir, "plot_basin_ddg.png"))
    plot_latent_trajectory(os.path.join(analysis_dir, "latents.npz"), os.path.join(analysis_dir, "plot_latent_trajectory.png"))
    plot_te(analysis, os.path.join(analysis_dir, "plot_te.png"))
    plot_susceptibility_quiver(analysis, os.path.join(analysis_dir, "plot_susceptibility_quiver.png"))

    write_binding_csv(
        wt_pdb=resolve_run_path(run_dir, summary["md"]["wt"]["prepared_pdb"]),
        mut_pdb=resolve_run_path(run_dir, summary["md"]["mut"]["prepared_pdb"]),
        ligand_resname=ligand_resname,
        cutoff_ang=cutoff_ang,
        out_csv=os.path.join(analysis_dir, "binding_site_residues.csv"),
    )

    vmd_script = None
    if isinstance(summary.get("visualization"), dict):
        vmd_script = summary["visualization"].get("vmd_script")
    if vmd_script:
        vmd_script = resolve_run_path(run_dir, vmd_script)
    if not vmd_script or not os.path.exists(vmd_script):
        vmd_script = os.path.join(analysis_dir, "view_wt_mut.vmd.tcl")
        write_vmd_comparison_script(
            wt_pdb=resolve_run_path(run_dir, summary["md"]["wt"]["prepared_pdb"]),
            wt_dcd=resolve_run_path(run_dir, summary["md"]["wt"]["traj_dcd"])
            if summary["md"]["wt"].get("traj_dcd")
            else None,
            mut_pdb=resolve_run_path(run_dir, summary["md"]["mut"]["prepared_pdb"]),
            mut_dcd=resolve_run_path(run_dir, summary["md"]["mut"]["traj_dcd"])
            if summary["md"]["mut"].get("traj_dcd")
            else None,
            ligand_resnames=[ligand_resname],
            output_tcl=vmd_script,
            site_cutoff_ang=cutoff_ang,
            latents_npz=os.path.join(analysis_dir, "latents.npz"),
        )

    flow_outputs = run_residue_flow_analysis(
        wt_pdb=resolve_run_path(run_dir, summary["md"]["wt"]["prepared_pdb"]),
        mut_pdb=resolve_run_path(run_dir, summary["md"]["mut"]["prepared_pdb"]),
        wt_traj_npz=resolve_run_path(run_dir, summary["md"]["wt"]["traj_npz"]),
        mut_traj_npz=resolve_run_path(run_dir, summary["md"]["mut"]["traj_npz"]),
        mut_dcd=resolve_run_path(run_dir, summary["md"]["mut"]["traj_dcd"])
        if summary["md"]["mut"].get("traj_dcd")
        else None,
        binding_csv=os.path.join(analysis_dir, "binding_site_residues.csv"),
        ligand_resname=ligand_resname,
        output_dir=analysis_dir,
    )

    script_path = os.path.relpath(vmd_script, run_dir) if vmd_script else None
    mut_flow_script = write_mut_flow_vmd_script(
        wt_pdb=resolve_run_path(run_dir, summary["md"]["wt"]["prepared_pdb"]),
        mut_pdb=resolve_run_path(run_dir, summary["md"]["mut"]["prepared_pdb"]),
        mut_dcd=resolve_run_path(run_dir, summary["md"]["mut"]["traj_dcd"])
        if summary["md"]["mut"].get("traj_dcd")
        else None,
        mut_traj_npz=resolve_run_path(run_dir, summary["md"]["mut"]["traj_npz"])
        if summary["md"]["mut"].get("traj_npz")
        else None,
        latents_npz=os.path.join(analysis_dir, "latents.npz"),
        ligand_resname=ligand_resname,
        binding_csv=os.path.join(analysis_dir, "binding_site_residues.csv"),
        output_tcl=os.path.join(analysis_dir, "view_mut_flow.vmd.tcl"),
        cutoff_ang=cutoff_ang,
        analysis=analysis,
    )
    mut_flow_rel = os.path.relpath(mut_flow_script, run_dir)
    manifest = {
        "plots": [
            "analysis/plot_basin_ddg.png",
            "analysis/plot_latent_trajectory.png",
            "analysis/plot_te.png",
            "analysis/plot_susceptibility_quiver.png",
            os.path.relpath(flow_outputs["plot_png"], run_dir),
        ],
        "tables": [
            "analysis/binding_site_residues.csv",
            os.path.relpath(flow_outputs["summary_json"], run_dir),
        ],
        "artifacts": [
            os.path.relpath(flow_outputs["graph_npz"], run_dir),
        ],
    }
    scripts: List[str] = []
    if script_path is not None:
        scripts.append(script_path)
    scripts.append(mut_flow_rel)
    scripts.append(os.path.relpath(flow_outputs["vmd_script"], run_dir))
    manifest["scripts"] = scripts
    with open(os.path.join(analysis_dir, "plot_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def run_diagnostic_analysis(run_dir: str) -> Dict:
    """Extended diagnostics of one finished run.

    Writes convergence / time-course / information figures, the binding-site
    shift figure (when the table exists), ``interpretation_summary.json`` and
    ``enhanced_plot_manifest.json`` (returned) into ``<run_dir>/analysis``.
    """
    summary = load_json(os.path.join(run_dir, "run_summary.json"))
    analysis = load_json(os.path.join(run_dir, "analysis", "analysis_results.json"))
    latents = np.load(os.path.join(run_dir, "analysis", "latents.npz"))
    u_b_wt = np.asarray(latents["u_b_wt"], dtype=float)
    u_b_mut = np.asarray(latents["u_b_mut"], dtype=float)

    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    labels_wt, labels_mut = plot_basin_convergence(
        u_b_wt=u_b_wt,
        u_b_mut=u_b_mut,
        n_basins=int(analysis["basin_ddg"]["n_basins"]),
        out_png=os.path.join(analysis_dir, "plot_convergence_basin_prob.png"),
    )
    plot_latent_timecourse(
        u_b_wt=u_b_wt,
        u_b_mut=u_b_mut,
        out_png=os.path.join(analysis_dir, "plot_latent_timecourse.png"),
    )
    plot_information_dashboard(
        analysis=analysis,
        out_png=os.path.join(analysis_dir, "plot_information_dashboard.png"),
    )
    binding_csv = os.path.join(analysis_dir, "binding_site_residues.csv")
    if os.path.exists(binding_csv):
        plot_binding_site_shift(
            binding_csv=binding_csv,
            out_png=os.path.join(analysis_dir, "plot_binding_site_shift.png"),
        )

    write_interpretation_summary(
        summary=summary,
        analysis=analysis,
        labels_wt=labels_wt,
        labels_mut=labels_mut,
        out_json=os.path.join(analysis_dir, "interpretation_summary.json"),
    )

    manifest = {
        "plots": [
            "analysis/plot_convergence_basin_prob.png",
            "analysis/plot_latent_timecourse.png",
            "analysis/plot_information_dashboard.png",
            "analysis/plot_binding_site_shift.png",
        ],
        "summary": "analysis/interpretation_summary.json",
    }
    vmd_script = None
    if isinstance(summary.get("visualization"), dict):
        vmd_script = summary["visualization"].get("vmd_script")
    if vmd_script:
        abs_script = resolve_run_path(run_dir, vmd_script)
        if os.path.exists(abs_script):
            manifest["scripts"] = [os.path.relpath(abs_script, run_dir)]
    mut_flow = os.path.join(run_dir, "analysis", "view_mut_flow.vmd.tcl")
    if os.path.exists(mut_flow):
        scripts = list(manifest.get("scripts", []))
        rel = os.path.relpath(mut_flow, run_dir)
        if rel not in scripts:
            scripts.append(rel)
        manifest["scripts"] = scripts
    with open(os.path.join(analysis_dir, "enhanced_plot_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
