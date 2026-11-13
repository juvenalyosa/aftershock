"""Fast (MD-free) mutation panel: build each mutant and score it with the fast scanner.

Use it to prioritise designs before spending GPU time on the MD panel.
Scenarios come from the built-in toy panel and/or user-supplied mutation sets
(see ``pert.pipelines.scenarios``).
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Sequence

import numpy as np

from pert.allostery.fast_scan import run_fast_scan
from pert.config import BuildConfig
from pert.pipelines.scenarios import clean_scenarios, MUTATION_SCENARIOS, ScenarioDef
from pert.structure.prepare import prepare_structures
from pert.utils.io import write_csv
from pert.visualization.panel_plots import plot_fast_panel_summary


# --- panel ---------------------------------------------------------------------------------------


def _scenario_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """Mean/std of the fast-scan metrics per scenario."""
    by_name: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_name.setdefault(str(row["scenario"]), []).append(dict(row))
    out: Dict[str, Dict[str, float]] = {}
    metrics = [
        "global_fast_allostery_score",
        "mean_top10_destabilization_proxy",
        "aligned_ca_rmsd_ang",
    ]
    for name, items in by_name.items():
        entry: Dict[str, float] = {"replicas": float(len(items))}
        for metric in metrics:
            vals = np.asarray([float(item[metric]) for item in items], dtype=float)
            entry[f"{metric}_mean"] = float(np.mean(vals))
            entry[f"{metric}_std"] = float(np.std(vals))
        out[name] = entry
    return out


def run_fast_panel(
    reference_pdb: str,
    template_pdb: str,
    ligand_xml: str,
    ligand_resname: str,
    template: str,
    chain_id: str,
    replicas: int,
    output_dir: str,
    num_models: int,
    num_processors: int,
    selected_scenarios: Sequence[str],
    cutoff_ang: float,
    contact_threshold_ang: float,
    source_hotspots: int,
    top_residues: int,
    scenario_defs: Sequence[ScenarioDef] | None = None,
) -> Dict[str, object]:
    """Build every scenario's mutant structure and fast-score it; writes panel tables and a figure."""
    os.makedirs(output_dir, exist_ok=True)
    template_code = template[:-1]
    scenarios: List[ScenarioDef] = clean_scenarios(
        list(scenario_defs) if scenario_defs is not None else [(name, list(mutations)) for name, mutations in MUTATION_SCENARIOS]
    )
    if selected_scenarios:
        selected = set(selected_scenarios)
        scenarios = [item for item in scenarios if item[0] in selected]
        if not scenarios:
            raise ValueError("No valid scenarios selected.")

    rows: List[Dict[str, object]] = []
    for scenario_index, (scenario_name, mutations) in enumerate(scenarios):
        for rep in range(1, int(replicas) + 1):
            run_dir = os.path.join(output_dir, scenario_name, f"rep{rep}")
            os.makedirs(run_dir, exist_ok=True)
            seed_dir = os.path.join(run_dir, "model_build", "modeling_job")
            os.makedirs(seed_dir, exist_ok=True)
            if template_pdb:
                shutil.copy2(template_pdb, os.path.join(seed_dir, f"{template_code}.pdb"))

            seed_modeller = 17 + scenario_index * 100 + rep
            build_cfg = BuildConfig(
                reference_pdb=reference_pdb,
                ligand_xml=ligand_xml,
                ligand_resnames=[ligand_resname],
                mutations=mutations,
                templates=[template],
                chain_id=chain_id,
                num_models=num_models,
                num_processors=num_processors,
                modeller_seed=seed_modeller,
            )
            prep = prepare_structures(build_cfg, output_dir=run_dir)
            scan = run_fast_scan(
                wt_pdb=prep.wt_pdb,
                mut_pdb=prep.mutant_ligand_pdb,
                ligand_resname=ligand_resname,
                output_dir=os.path.join(run_dir, "fast_scan"),
                mutation_label=scenario_name,
                mutation_specs=mutations,
                cutoff_ang=cutoff_ang,
                contact_threshold_ang=contact_threshold_ang,
                source_hotspots=source_hotspots,
                top_residues=top_residues,
            )
            top_res = ",".join(row["node_label"] for row in scan["top_residues"][: min(10, len(scan["top_residues"]))])
            rows.append(
                {
                    "scenario": scenario_name,
                    "replica": rep,
                    "mutations": ",".join(mutations),
                    "predicted_allosteric_effect": scan["predicted_allosteric_effect"],
                    "global_fast_allostery_score": float(scan["global_fast_allostery_score"]),
                    "mean_top10_destabilization_proxy": float(scan["mean_top10_destabilization_proxy"]),
                    "aligned_ca_rmsd_ang": float(scan["aligned_ca_rmsd_ang"]),
                    "mutation_source_residues": ",".join(scan.get("mutation_source_residues", [])),
                    "source_residues": ",".join(scan["source_residues"]),
                    "target_residues": ",".join(scan["target_residues"]),
                    "deleted_wt_residues": ",".join(scan["deleted_wt_residues"]),
                    "top_residues": top_res,
                    "summary_json": scan["outputs"]["summary_json"],
                }
            )

    metrics_csv = os.path.join(output_dir, "fast_panel_metrics.csv")
    write_csv(metrics_csv, rows)
    summary = _scenario_summary(rows)
    summary_json = os.path.join(output_dir, "fast_panel_summary.json")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_png = os.path.join(output_dir, "fast_panel_summary.png")
    plot_fast_panel_summary(summary, plot_png)

    manifest = {
        "runs": len(rows),
        "scenarios": [name for name, _ in scenarios],
        "metrics_csv": metrics_csv,
        "summary_json": summary_json,
        "summary_plot": plot_png,
        "method_note": (
            "No production MD. Supports SNP/SNPs, INS, DEL, DELINS/INSDEL, DUP, and "
            "reference-coordinate combinations through named mutation sets. Use for fast "
            "prioritization before targeted validation."
        ),
    }
    manifest_path = os.path.join(output_dir, "fast_panel_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
