"""Panel-level consensus of the per-run rankings.

One MD run is one sample.  This layer aggregates replicas of each mutation
scenario: replicate support (how often a residue is in the top N), bootstrap
confidence intervals of the score, a conservative score (CI lower bound) and a
consensus score.  Optionally a control scenario is subtracted to separate
mutation-specific residues from residues that light up for any perturbation.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pert.allostery.ranking import run_allostery_ranking
from pert.utils.io import read_csv, safe_bool, safe_float, safe_int, write_csv
from pert.utils.numeric import bootstrap_mean_ci
from pert.utils.paths import find_run_dirs, replicate_from_run_dir, scenario_from_run_dir
from pert.visualization.allostery_plots import plot_consensus_scenario
from pert.visualization.vmd_allostery import write_consensus_vmd


def _collect_run_records(
    run_dirs: Sequence[str],
    rank_missing: bool,
    ligand_resname: str,
    cutoff_ang: float,
    source_hotspots: int,
    top_edges: int,
    top_paths: int,
    top_residues: int,
    max_frames: Optional[int],
    contact_threshold_ang: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """Load (and, if requested, compute) the per-run rankings of every run of a panel."""
    residue_records: List[Dict[str, Any]] = []
    region_records: List[Dict[str, Any]] = []
    scenario_runs: DefaultDict[str, List[str]] = defaultdict(list)
    representative_run: Dict[str, Dict[str, str]] = {}

    for run_dir_raw in run_dirs:
        run_dir = os.path.abspath(run_dir_raw)
        scenario = scenario_from_run_dir(run_dir)
        replicate = replicate_from_run_dir(run_dir)
        scenario_runs[scenario].append(run_dir)
        representative_run.setdefault(
            scenario,
            {"run_dir": run_dir, "scenario": scenario, "replicate": replicate},
        )

        analysis_dir = os.path.join(run_dir, "analysis")
        residue_csv = os.path.join(analysis_dir, "allostery_residue_scores.csv")
        region_csv = os.path.join(analysis_dir, "allostery_region_scores.csv")
        if rank_missing and not os.path.exists(residue_csv):
            run_allostery_ranking(
                run_dir=run_dir,
                ligand_resname=ligand_resname,
                cutoff_ang=cutoff_ang,
                source_hotspots=source_hotspots,
                top_edges_n=top_edges,
                top_paths_n=top_paths,
                top_residues=top_residues,
                region_window=5,
                max_frames=max_frames,
                contact_threshold_ang=contact_threshold_ang,
            )

        if os.path.exists(residue_csv):
            rows = read_csv(residue_csv)
            max_rank = max([safe_int(row.get("rank"), 0) for row in rows] or [0])
            for row in rows:
                rank = safe_int(row.get("rank"), max_rank)
                rank_percentile = 1.0
                if max_rank > 1:
                    rank_percentile = 1.0 - float(rank - 1) / float(max_rank - 1)
                rec: Dict[str, Any] = {
                    "scenario": scenario,
                    "replicate": replicate,
                    "run_dir": run_dir,
                    "max_rank_in_run": max_rank,
                    "rank_percentile": rank_percentile,
                }
                rec.update(row)
                residue_records.append(rec)

        if os.path.exists(region_csv):
            for row in read_csv(region_csv):
                rec = {"scenario": scenario, "replicate": replicate, "run_dir": run_dir}
                rec.update(row)
                region_records.append(rec)

    return residue_records, region_records, dict(scenario_runs), representative_run


# --- aggregation across replicas ---------------------------------------------------------


def _aggregate_residues(
    records: Sequence[Dict[str, Any]],
    scenario_runs: Dict[str, List[str]],
    top_support_n: int,
    rng: np.random.Generator,
    n_boot: int,
) -> List[Dict[str, Any]]:
    """Scenario-level statistics of every residue across replicas.

    ``consensus_score = presence * (0.50 mean score + 0.30 top-N support + 0.20 mean rank percentile)``;
    ``conservative_score`` is the lower bound of the bootstrap 95% CI of the score.
    """
    by_key: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_key[(str(row["scenario"]), str(row["node_label"]))].append(dict(row))

    component_cols = [
        "structural_component",
        "dynamics_component",
        "contact_component",
        "te_component",
        "relay_component",
        "target_coupling_component",
    ]
    metric_cols = [
        "total_score",
        "structural_shift_ang",
        "rmsf_delta_abs_ang",
        "speed_delta_abs_ang_per_ps",
        "contact_change",
        "te_out_gain",
        "te_in_gain",
        "te_out_abs_delta",
        "te_in_abs_delta",
        "target_coupling_gain",
        "relay_score",
        "rank_percentile",
    ]

    rows: List[Dict[str, Any]] = []
    for (scenario, node_label), items in by_key.items():
        total_reps = len(set(scenario_runs.get(scenario, [])))
        observed_reps = len({str(item["run_dir"]) for item in items})
        presence_fraction = float(observed_reps) / float(max(1, total_reps))
        ranks = np.asarray([safe_float(item.get("rank")) for item in items], dtype=float)
        scores = np.asarray([safe_float(item.get("total_score")) for item in items], dtype=float)
        support_top = float(np.sum(ranks <= float(top_support_n))) / float(max(1, total_reps))
        ci_low, ci_high = bootstrap_mean_ci(scores.tolist(), rng=rng, n_boot=n_boot)

        first = items[0]
        row: Dict[str, Any] = {
            "scenario": scenario,
            "node_label": node_label,
            "index": first.get("index", ""),
            "chain": first.get("chain", ""),
            "resid": first.get("resid", ""),
            "residue": first.get("residue", ""),
            "wt_residue": first.get("wt_residue", ""),
            "replicates_total": total_reps,
            "replicates_observed": observed_reps,
            "presence_fraction": presence_fraction,
            "support_top_n": int(top_support_n),
            "support_top_n_fraction": support_top,
            "rank_mean": float(np.nanmean(ranks)),
            "rank_median": float(np.nanmedian(ranks)),
            "rank_best": float(np.nanmin(ranks)),
            "rank_worst": float(np.nanmax(ranks)),
            "is_source_hotspot_fraction": float(np.mean([safe_bool(item.get("is_source_hotspot")) for item in items])),
            "is_target_site_fraction": float(np.mean([safe_bool(item.get("is_target_site")) for item in items])),
            "total_score_mean": float(np.nanmean(scores)),
            "total_score_std": float(np.nanstd(scores)),
            "total_score_ci95_low": ci_low,
            "total_score_ci95_high": ci_high,
            "conservative_score": ci_low,
        }
        for col in metric_cols:
            vals = np.asarray([safe_float(item.get(col)) for item in items], dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_std"] = float(np.nanstd(vals))
        for col in component_cols:
            vals = np.asarray([safe_float(item.get(col)) for item in items], dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(vals))

        rank_percentile_mean = float(row.get("rank_percentile_mean", 0.0))
        mean_total = float(row["total_score_mean"])
        row["consensus_score"] = presence_fraction * (
            0.50 * mean_total + 0.30 * support_top + 0.20 * rank_percentile_mean
        )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row["consensus_score"]),
            float(row["support_top_n_fraction"]),
            float(row["total_score_mean"]),
        ),
        reverse=True,
    )
    scenario_rank: DefaultDict[str, int] = defaultdict(int)
    for row in rows:
        scenario = str(row["scenario"])
        scenario_rank[scenario] += 1
        row["consensus_rank"] = scenario_rank[scenario]
    return rows


def _aggregate_regions(
    records: Sequence[Dict[str, Any]],
    scenario_runs: Dict[str, List[str]],
    top_support_n: int,
) -> List[Dict[str, Any]]:
    """Scenario-level statistics of every residue window across replicas."""
    by_key: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_key[(str(row["scenario"]), str(row["region"]))].append(dict(row))

    rows: List[Dict[str, Any]] = []
    for (scenario, region), items in by_key.items():
        total_reps = len(set(scenario_runs.get(scenario, [])))
        observed_reps = len({str(item["run_dir"]) for item in items})
        ranks = np.asarray([safe_float(item.get("rank")) for item in items], dtype=float)
        means = np.asarray([safe_float(item.get("mean_total_score")) for item in items], dtype=float)
        maxes = np.asarray([safe_float(item.get("max_total_score")) for item in items], dtype=float)
        support_top = float(np.sum(ranks <= float(top_support_n))) / float(max(1, total_reps))
        presence = float(observed_reps) / float(max(1, total_reps))
        row = {
            "scenario": scenario,
            "region": region,
            "chain": items[0].get("chain", ""),
            "replicates_total": total_reps,
            "replicates_observed": observed_reps,
            "presence_fraction": presence,
            "support_top_n": int(top_support_n),
            "support_top_n_fraction": support_top,
            "rank_mean": float(np.nanmean(ranks)),
            "rank_median": float(np.nanmedian(ranks)),
            "rank_best": float(np.nanmin(ranks)),
            "mean_total_score_mean": float(np.nanmean(means)),
            "mean_total_score_std": float(np.nanstd(means)),
            "max_total_score_mean": float(np.nanmean(maxes)),
            "max_total_score_std": float(np.nanstd(maxes)),
            "top_residue_modes": ",".join(sorted(set(str(item.get("top_residue", "")) for item in items if item.get("top_residue")))),
            "consensus_score": presence * (
                0.55 * float(np.nanmean(means)) + 0.30 * support_top + 0.15 * float(np.nanmean(maxes))
            ),
        }
        rows.append(row)

    rows.sort(key=lambda row: float(row["consensus_score"]), reverse=True)
    scenario_rank: DefaultDict[str, int] = defaultdict(int)
    for row in rows:
        scenario = str(row["scenario"])
        scenario_rank[scenario] += 1
        row["consensus_rank"] = scenario_rank[scenario]
    return rows


# --- control baseline -----------------------------------------------------------------------


def _apply_control_baseline(
    residue_rows: List[Dict[str, Any]],
    control_scenario: Optional[str],
    baseline_margin: float,
) -> List[Dict[str, Any]]:
    """Compare every scenario with a control scenario, residue by residue.

    Adds the control's scores, the differences and a ``mutation_specific_vs_control``
    flag to the rows, and returns the non-control rows ranked by specificity.
    """
    control_name = str(control_scenario or "").strip()
    for row in residue_rows:
        row["control_scenario"] = control_name
        row["control_consensus_score"] = ""
        row["control_total_score_mean"] = ""
        row["control_total_score_ci95_high"] = ""
        row["control_support_top_n_fraction"] = ""
        row["baseline_delta_consensus_score"] = ""
        row["baseline_delta_total_score_mean"] = ""
        row["baseline_specific_score"] = ""
        row["baseline_match_key"] = ""
        row["baseline_exceeds_control_ci"] = False
        row["mutation_specific_vs_control"] = False

    if not control_name:
        return []

    control_by_residue: Dict[str, Dict[str, Any]] = {
        str(row.get("node_label", "")): row
        for row in residue_rows
        if str(row.get("scenario", "")) == control_name
    }
    control_by_index: Dict[str, Dict[str, Any]] = {
        str(row.get("index", "")): row
        for row in residue_rows
        if str(row.get("scenario", "")) == control_name and str(row.get("index", "")).strip()
    }
    control_by_chain_resid: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(row.get("chain", "")), str(row.get("resid", ""))): row
        for row in residue_rows
        if str(row.get("scenario", "")) == control_name
    }
    baseline_rows: List[Dict[str, Any]] = []
    for row in residue_rows:
        residue = str(row.get("node_label", ""))
        index_key = str(row.get("index", ""))
        chain_resid_key = (str(row.get("chain", "")), str(row.get("resid", "")))
        control = control_by_index.get(index_key) if index_key else None
        match_key = f"index:{index_key}" if control is not None and index_key else ""
        if control is None:
            control = control_by_residue.get(residue)
            match_key = f"node_label:{residue}" if control is not None else ""
        if control is None:
            control = control_by_chain_resid.get(chain_resid_key)
            match_key = (
                f"chain_resid:{chain_resid_key[0]}:{chain_resid_key[1]}"
                if control is not None
                else ""
            )
        if control is None:
            control_consensus = 0.0
            control_mean = 0.0
            control_ci_high = 0.0
            control_support = 0.0
        else:
            control_consensus = safe_float(control.get("consensus_score"), 0.0)
            control_mean = safe_float(control.get("total_score_mean"), 0.0)
            control_ci_high = safe_float(control.get("total_score_ci95_high"), 0.0)
            control_support = safe_float(control.get("support_top_n_fraction"), 0.0)

        delta_consensus = safe_float(row.get("consensus_score"), 0.0) - control_consensus
        delta_mean = safe_float(row.get("total_score_mean"), 0.0) - control_mean
        scenario_ci_low = safe_float(row.get("total_score_ci95_low"), 0.0)
        exceeds_ci = bool(scenario_ci_low > control_ci_high)
        is_control = str(row.get("scenario", "")) == control_name
        mutation_specific = bool(
            (not is_control)
            and delta_consensus >= float(baseline_margin)
            and delta_mean > 0.0
        )

        row["control_consensus_score"] = control_consensus
        row["control_total_score_mean"] = control_mean
        row["control_total_score_ci95_high"] = control_ci_high
        row["control_support_top_n_fraction"] = control_support
        row["baseline_delta_consensus_score"] = delta_consensus
        row["baseline_delta_total_score_mean"] = delta_mean
        row["baseline_specific_score"] = max(0.0, delta_consensus)
        row["baseline_match_key"] = match_key
        row["baseline_exceeds_control_ci"] = exceeds_ci
        row["mutation_specific_vs_control"] = mutation_specific
        if not is_control:
            baseline_rows.append(dict(row))

    baseline_rows.sort(
        key=lambda row: (
            str(row.get("scenario", "")),
            -safe_float(row.get("baseline_specific_score"), 0.0),
            -safe_float(row.get("consensus_score"), 0.0),
        )
    )
    scenario_rank: DefaultDict[str, int] = defaultdict(int)
    for row in baseline_rows:
        scenario = str(row.get("scenario", ""))
        scenario_rank[scenario] += 1
        row["baseline_specific_rank"] = scenario_rank[scenario]
    return baseline_rows


def run_consensus(
    panel_dir: str,
    run_dirs: Optional[Sequence[str]],
    rank_missing: bool,
    ligand_resname: str,
    cutoff_ang: float,
    source_hotspots: int,
    top_edges: int,
    top_paths: int,
    top_residues: int,
    top_regions: int,
    max_frames: Optional[int],
    contact_threshold_ang: float,
    output_dir: Optional[str],
    bootstrap: int,
    seed: int,
    control_scenario: Optional[str],
    baseline_margin: float,
) -> Dict[str, Any]:
    """Build all consensus tables, figures and VMD scenes of a panel directory."""
    panel_dir = os.path.abspath(panel_dir)
    selected_run_dirs = list(run_dirs) if run_dirs else find_run_dirs(panel_dir)
    if not selected_run_dirs:
        raise ValueError(f"No run_summary.json files found under {panel_dir}.")

    out_dir = os.path.abspath(output_dir or panel_dir)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    residue_records, region_records, scenario_runs, representative_run = _collect_run_records(
        run_dirs=selected_run_dirs,
        rank_missing=rank_missing,
        ligand_resname=ligand_resname,
        cutoff_ang=cutoff_ang,
        source_hotspots=source_hotspots,
        top_edges=top_edges,
        top_paths=top_paths,
        top_residues=top_residues,
        max_frames=max_frames,
        contact_threshold_ang=contact_threshold_ang,
    )
    if not residue_records:
        raise ValueError(
            "No per-run allostery_residue_scores.csv files found. "
            "Run `pert rank` first, or pass --rank-missing."
        )

    residue_rows = _aggregate_residues(
        residue_records,
        scenario_runs=scenario_runs,
        top_support_n=top_residues,
        rng=rng,
        n_boot=bootstrap,
    )
    region_rows = _aggregate_regions(region_records, scenario_runs=scenario_runs, top_support_n=top_regions)
    baseline_rows = _apply_control_baseline(
        residue_rows=residue_rows,
        control_scenario=control_scenario,
        baseline_margin=baseline_margin,
    )

    residue_csv = os.path.join(out_dir, "allostery_consensus_residues.csv")
    region_csv = os.path.join(out_dir, "allostery_consensus_regions.csv")
    baseline_csv = os.path.join(out_dir, "allostery_baseline_specific_residues.csv")
    write_csv(residue_csv, residue_rows)
    write_csv(region_csv, region_rows)
    if control_scenario:
        write_csv(baseline_csv, baseline_rows)

    scenarios = sorted(scenario_runs.keys())
    plot_paths: Dict[str, str] = {}
    vmd_paths: Dict[str, str] = {}
    for scenario in scenarios:
        plot_path = os.path.join(out_dir, f"plot_allostery_consensus_{scenario}.png")
        plot_consensus_scenario(scenario, residue_rows, plot_path, top_n=top_residues)
        if os.path.exists(plot_path):
            plot_paths[scenario] = plot_path
        vmd_path = os.path.join(out_dir, f"view_allostery_consensus_{scenario}.vmd.tcl")
        written = write_consensus_vmd(
            scenario=scenario,
            representative_run=representative_run.get(scenario, {}),
            consensus_rows=residue_rows,
            ligand_resname=ligand_resname,
            out_tcl=vmd_path,
            top_n=top_residues,
        )
        if written:
            vmd_paths[scenario] = written

    consensus = {
        "panel_dir": panel_dir,
        "run_dirs": selected_run_dirs,
        "scenarios": scenarios,
        "replicates_by_scenario": {k: len(set(v)) for k, v in scenario_runs.items()},
        "top_residues_by_scenario": {
            scenario: [
                row
                for row in residue_rows
                if str(row["scenario"]) == scenario and int(row["consensus_rank"]) <= int(top_residues)
            ]
            for scenario in scenarios
        },
        "top_regions_by_scenario": {
            scenario: [
                row
                for row in region_rows
                if str(row["scenario"]) == scenario and int(row["consensus_rank"]) <= int(top_regions)
            ]
            for scenario in scenarios
        },
        "outputs": {
            "residue_consensus_csv": residue_csv,
            "region_consensus_csv": region_csv,
            "baseline_specific_residue_csv": baseline_csv if control_scenario else None,
            "plots": plot_paths,
            "vmd_scripts": vmd_paths,
        },
        "control_baseline": {
            "control_scenario": control_scenario,
            "baseline_margin": baseline_margin,
            "baseline_specific_residue_csv": baseline_csv if control_scenario else None,
        },
    }
    consensus_json = os.path.join(out_dir, "allostery_consensus.json")
    with open(consensus_json, "w", encoding="utf-8") as handle:
        json.dump(consensus, handle, indent=2)

    manifest = {
        "consensus_json": consensus_json,
        "residue_consensus_csv": residue_csv,
        "region_consensus_csv": region_csv,
        "baseline_specific_residue_csv": baseline_csv if control_scenario else None,
        "plots": plot_paths,
        "vmd_scripts": vmd_paths,
    }
    manifest_path = os.path.join(out_dir, "allostery_consensus_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
