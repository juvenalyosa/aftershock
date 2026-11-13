"""MD mutation panel: several scenarios x replicas through the full pipeline.

Scenarios are arbitrary named mutation sets for any protein (see
``pert.pipelines.scenarios``); the built-in toy panel on the zinc finger 1ZNF
is only the default.  Before anything is built or simulated the panel checks
its inputs: files exist, the ligand is in the reference structure and every
mutation fits the reference sequence.

Seeds are derived from the scenario *name* and the replica number, so a
scenario keeps its seeds (and its finished simulations stay valid) when other
scenarios are added to, removed from or filtered out of the panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from typing import Dict, List, Sequence

import numpy as np

from pert.analysis.topology_safe import comparison_metrics
from pert.config import BuildConfig, FieldConfig, LatentConfig, MDConfig, TEConfig
from pert.deps import app, ensure_openmm
from pert.pipelines.null_control import NullControlUnavailable, run_null_control
from pert.pipelines.scenarios import collect_scenarios, scenario_seed_offsets, ScenarioDef, validate_scenarios
from pert.pipelines.single_run import run_full_pipeline
from pert.structure.residues import is_protein_residue, parse_residue_number, sequence_and_start_from_pdb
from pert.visualization.panel_plots import plot_md_panel


def validate_panel_inputs(args: argparse.Namespace, scenarios: Sequence[ScenarioDef]) -> Dict[str, str]:
    """Fail before the first simulation rather than hours into the panel.

    Checks that the input files exist, that the ligand is present in the
    reference structure and that every scenario fits the reference sequence.
    Returns ``{scenario name: mutated sequence}``.
    """
    for label, path in (
        ("reference PDB", args.reference_pdb),
        ("template PDB", args.template_pdb),
        ("ligand force-field XML", args.ligand_xml),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The {label} does not exist: {path}")
    if len(args.template) < 2:
        raise ValueError(f"--template must be a template code followed by its chain (e.g. 1ZNFA), got '{args.template}'.")
    if not 1 <= args.replicas <= 30:
        raise ValueError("--replicas must be between 1 and 30.")

    ensure_openmm()
    residue_names = {residue.name.upper() for residue in app.PDBFile(args.reference_pdb).topology.residues()}
    if args.ligand_resname.upper() not in residue_names:
        raise ValueError(
            f"Ligand residue '{args.ligand_resname}' is not in {args.reference_pdb}. "
            f"Non-protein residues found: {sorted(n for n in residue_names if not is_protein_residue(n))}"
        )
    sequence, start_pos = sequence_and_start_from_pdb(args.reference_pdb, args.chain_id)

    # Mutation positions are mapped as "first residue number + index".  That is
    # only right when the chain is numbered without gaps or insertion codes;
    # otherwise every position after the first break would silently be shifted.
    numbers = [
        parse_residue_number(residue.id)
        for chain in app.PDBFile(args.reference_pdb).topology.chains()
        if not args.chain_id or chain.id == args.chain_id
        for residue in chain.residues()
        if is_protein_residue(residue.name)
    ][: len(sequence)]
    breaks = [
        f"{previous}->{current}"
        for previous, current in zip(numbers[:-1], numbers[1:])
        if previous is None or current is None or current != previous + 1
    ]
    if breaks:
        raise ValueError(
            f"Chain '{args.chain_id}' of {args.reference_pdb} is not numbered consecutively (breaks at "
            f"{', '.join(breaks[:5])}{' ...' if len(breaks) > 5 else ''}). Mutation positions after a break would be "
            "assigned to the wrong residues. Use a reference structure without missing residues or insertion "
            "codes (model the gaps first), numbered consecutively."
        )
    return validate_scenarios(scenarios, sequence, start_pos)



def _scenario_table(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation of the panel metrics per scenario (over replicas)."""
    by_name: Dict[str, List[Dict]] = {}
    for row in rows:
        by_name.setdefault(row["scenario"], []).append(row)

    metrics = [
        "te_mut_p_to_b",
        "te_delta_mut_minus_wt",
        "te_mut_pvalue",
        "js_divergence",
        "classifier_auc",
        "basin_population_tv",
        "gmm_kl_wt_to_mut",
        "susceptibility_delta_fro_norm",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for name, items in by_name.items():
        entry: Dict[str, float] = {"replicas": float(len(items))}
        for metric in metrics:
            vals = np.asarray([float(item.get(metric, np.nan)) for item in items], dtype=float)
            entry[f"{metric}_mean"] = float(np.nanmean(vals))
            entry[f"{metric}_std"] = float(np.nanstd(vals))
        out[name] = entry
    return out


def _write_panel_csv(rows: List[Dict], out_csv: str) -> None:
    """Write the per-run metrics table of the MD panel (no file when there are no rows)."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_md_panel(args: argparse.Namespace) -> Dict[str, object]:
    """Run every selected scenario x replica through the full pipeline and aggregate.

    Scenarios come from ``--mutation-set`` / ``--mutation-file`` (any protein)
    or, by default, from the built-in toy panel; ``--scenarios`` filters them.
    Seeds (MODELLER, MD, TE) are derived from the scenario name and the replica
    number so that every run is reproducible and independent.  Writes
    ``mutation_panel_metrics.csv``, ``mutation_panel_summary.json``,
    ``mutation_panel_comparison.png`` and ``panel_manifest.json`` to
    ``args.output_dir`` and returns the manifest.  Unless ``--no-null-control``
    is given, the panel then calibrates itself against the WT-vs-WT null
    (``pert.pipelines.null_control``) using the WT trajectories it just produced.  ``args`` is the namespace built
    by ``pert.cli.md_panel.build_arg_parser``.
    """
    scenarios = collect_scenarios(
        mutation_sets=args.mutation_set,
        mutation_files=args.mutation_file,
        include_builtins=bool(args.include_builtins),
        selected=args.scenarios,
    )
    mutated_sequences = validate_panel_inputs(args, scenarios)
    seed_offsets = scenario_seed_offsets([name for name, _ in scenarios])

    os.makedirs(args.output_dir, exist_ok=True)
    template_code = args.template[:-1]
    all_rows: List[Dict] = []

    for scenario_name, mutations in scenarios:
        offset = seed_offsets[scenario_name]
        for rep in range(1, args.replicas + 1):
            run_dir = os.path.join(args.output_dir, scenario_name, f"rep{rep}")
            os.makedirs(run_dir, exist_ok=True)
            seed_dir = os.path.join(run_dir, "model_build", "modeling_job")
            os.makedirs(seed_dir, exist_ok=True)
            shutil.copy2(args.template_pdb, os.path.join(seed_dir, f"{template_code}.pdb"))

            # The mutant simulation uses seed_md + 1, hence the spacing between replicas.
            seed_modeller = 17 + (offset % 400) * 100 + rep
            seed_md = args.md_seed_base + offset * 100 + rep * 3
            seed_te = args.te_seed_base + offset * 100 + rep

            nonbonded_method = args.nonbonded_method
            if args.no_solvent and nonbonded_method.strip().upper() in {
                "PME",
                "EWALD",
                "CUTOFFPERIODIC",
            }:
                nonbonded_method = "CutoffNonPeriodic"

            build_cfg = BuildConfig(
                reference_pdb=args.reference_pdb,
                ligand_xml=args.ligand_xml,
                ligand_resnames=[args.ligand_resname],
                mutations=mutations,
                templates=[args.template],
                chain_id=args.chain_id,
                num_models=args.num_models,
                num_processors=args.num_processors,
                modeller_seed=seed_modeller,
            )
            md_cfg = MDConfig(
                temperature_k=args.temperature_k,
                add_solvent=not args.no_solvent,
                nonbonded_method=nonbonded_method,
                nonbonded_cutoff_nm=args.nonbonded_cutoff_nm,
                equilibration_steps=args.equil_steps,
                production_steps=args.prod_steps,
                report_interval=args.report_interval,
                platform=args.platform,
                random_seed=seed_md,
                checkpoint_interval_steps=args.checkpoint_interval_steps,
            )
            field_cfg = FieldConfig(
                rb_ang=args.rb_ang,
                rp_ang=args.rp_ang,
                grid_spacing_ang=args.grid_spacing_ang,
                sigma_ang=args.sigma_ang,
                max_grid_points=args.max_grid_points,
                include_velocity=not args.occupancy_only,
                remove_com=True,
                frame_mode=args.frame_mode,
            )
            latent_cfg = LatentConfig(
                lag=args.vamp_lag,
                m_b=args.latent_b,
                m_p=args.latent_p,
                m_l=args.latent_l,
                reg=1e-6,
                solver=args.vamp_solver,
            )
            te_cfg = TEConfig(
                max_order=args.var_max_order,
                surrogates=args.te_surrogates,
                random_seed=seed_te,
            )

            summary = run_full_pipeline(
                build_cfg=build_cfg,
                md_cfg=md_cfg,
                field_cfg=field_cfg,
                latent_cfg=latent_cfg,
                te_cfg=te_cfg,
                output_dir=run_dir,
                resume=not args.no_resume,
            )
            analysis = summary["analysis"]
            dist = analysis["distribution_shift"]
            basin_block = analysis["basin_ddg"]
            basin = basin_block["basins"]
            headline = comparison_metrics(analysis)
            dominant = max(basin, key=lambda row: row["p_mut"])

            all_rows.append(
                {
                    "scenario": scenario_name,
                    "replica": rep,
                    "mutations": ",".join(mutations),
                    "te_mut_p_to_b": float(analysis["te_mut_p_to_b_given_l"]["te_driver_to_target"]),
                    "te_mut_pvalue": float(analysis["te_mut_p_to_b_given_l"]["surrogate_pvalue"]),
                    "te_delta_mut_minus_wt": float(analysis["te_delta_mut_minus_wt"]),
                    "js_divergence": headline["js_divergence"],
                    "classifier_auc": headline["classifier_auc"],
                    "basin_population_tv": headline["basin_population_tv"],
                    "gmm_kl_wt_to_mut": float(dist.get("gmm_kl_wt_to_mut", np.nan)),
                    "susceptibility_delta_fro_norm": float(analysis["susceptibility_delta_fro_norm"]),
                    "dominant_basin": int(dominant["basin"]),
                    "dominant_basin_p_mut": float(dominant["p_mut"]),
                    # None when the dominant mutant basin is never visited by the WT.
                    "dominant_basin_ddg_kcal_mol": (
                        float("nan") if dominant["ddg_kcal_mol"] is None else float(dominant["ddg_kcal_mol"])
                    ),
                    "effective_samples_min": float(
                        min(basin_block["effective_samples_wt"], basin_block["effective_samples_mut"])
                    ),
                }
            )

    out_csv = os.path.join(args.output_dir, "mutation_panel_metrics.csv")
    _write_panel_csv(all_rows, out_csv)

    agg = _scenario_table(all_rows)
    out_json = os.path.join(args.output_dir, "mutation_panel_summary.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(agg, handle, indent=2)

    out_plot = os.path.join(args.output_dir, "mutation_panel_comparison.png")
    plot_md_panel(agg, out_plot)

    payload = {
        "runs": len(all_rows),
        "scenarios": [name for name, _ in scenarios],
        "scenario_definitions": {
            name: {
                "mutations": list(mutations),
                "mutated_sequence": mutated_sequences[name],
                "seed_offset": seed_offsets[name],
            }
            for name, mutations in scenarios
        },
        "metrics_csv": out_csv,
        "summary_json": out_json,
        "comparison_plot": out_plot,
    }

    # Null control: the panel's own WT trajectories, compared with each other,
    # say how large every metric is when nothing was mutated.
    if args.no_null_control:
        payload["null_control"] = {"status": "skipped (--no-null-control)"}
    else:
        try:
            null_summary = run_null_control(
                panel_dir=args.output_dir,
                ligand_resnames=[args.ligand_resname],
                field_cfg=field_cfg,
                latent_cfg=latent_cfg,
                te_cfg=TEConfig(
                    max_order=args.var_max_order,
                    surrogates=args.te_surrogates,
                    random_seed=args.te_seed_base,
                ),
                temperature_k=args.temperature_k,
                max_pairs=args.max_null_pairs,
            )
            payload["null_control"] = {
                "status": "done",
                "null_pairs": null_summary["null_pairs"],
                "evidence_by_scenario": {
                    name: entry["evidence"] for name, entry in null_summary["scenarios"].items()
                },
                **null_summary["outputs"],
            }
        except NullControlUnavailable as exc:
            # A panel with a single run has no second WT trajectory to compare with.
            payload["null_control"] = {"status": "unavailable", "reason": str(exc)}
            print(f"[null control] unavailable: {exc}")
    with open(os.path.join(args.output_dir, "panel_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
