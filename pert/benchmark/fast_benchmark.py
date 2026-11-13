"""Benchmark of the fast scanner against labels, measured effects or other tools.

Each row of the input CSV is one variant.  Its structures are taken from
``wt_pdb``/``mut_pdb``, from a finished ``run_dir``, or built on the fly.  The
global fast score is then compared with binary labels (ROC AUC, average
precision, best threshold), continuous effects (Pearson/Spearman) and optional
external predictors such as FoldX or DDMut.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from typing import Any, Dict, List, Sequence, Tuple

from pert.allostery.fast_scan import infer_structures_from_run_dir, run_fast_scan
from pert.benchmark.metrics import (
    average_precision,
    best_binary_threshold,
    binary_label_column,
    numeric_column,
    pearson,
    roc_auc,
    spearman,
)
from pert.benchmark.plots import plot_benchmark
from pert.config import BuildConfig
from pert.structure.prepare import prepare_structures
from pert.utils.io import write_csv


def _split_mutation_tokens(value: object) -> List[str]:
    """Split one CSV cell into mutation tokens (comma or semicolon separated)."""
    out: List[str] = []
    for chunk in str(value or "").replace(";", ",").split(","):
        token = chunk.strip()
        if token:
            out.append(token)
    return out


def _load_rows(path: str) -> List[Dict[str, str]]:
    """Read and sanity-check the benchmark CSV."""
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Benchmark CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Benchmark CSV contains no rows: {path}")
    if "name" not in rows[0]:
        raise ValueError("Benchmark CSV must contain a 'name' column.")
    return rows


def _evaluate_predictions(
    rows: Sequence[Dict[str, object]],
    label_column: str,
    effect_column: str,
    external_score_columns: Sequence[str],
) -> Dict[str, Any]:
    """Agreement of one score column with the labels and the measured effects."""
    score_columns = ["global_fast_allostery_score", "mean_top10_destabilization_proxy", *external_score_columns]
    summary: Dict[str, Any] = {"n": len(rows), "scores": {}}
    labels = binary_label_column(rows, label_column) if label_column else None
    effects = numeric_column(rows, effect_column) if effect_column else []
    has_effects = bool(effect_column) and any(math.isfinite(v) for v in effects)

    for score_col in score_columns:
        scores = numeric_column(rows, score_col)
        entry: Dict[str, Any] = {
            "available": int(sum(math.isfinite(v) for v in scores)),
        }
        if labels is not None:
            entry["roc_auc"] = roc_auc(labels, scores)
            entry["average_precision"] = average_precision(labels, scores)
            entry["best_threshold"] = best_binary_threshold(labels, scores)
        if has_effects:
            entry["pearson_vs_effect"] = pearson(scores, effects)
            entry["spearman_vs_effect"] = spearman(scores, effects)
        summary["scores"][score_col] = entry
    return summary


def _resolve_structure_pair(
    row: Dict[str, str],
    output_dir: str,
    index: int,
    args: argparse.Namespace,
    mutations: Sequence[str],
) -> Tuple[str, str]:
    """Find or build the WT/mutant structures of one benchmark row."""
    run_dir = str(row.get("run_dir", "")).strip()
    if run_dir:
        wt_pdb, mut_pdb, _ = infer_structures_from_run_dir(run_dir)
        return wt_pdb, mut_pdb

    wt_pdb = str(row.get("wt_pdb", "")).strip()
    mut_pdb = str(row.get("mut_pdb", "")).strip()
    if wt_pdb and mut_pdb:
        return wt_pdb, mut_pdb

    if not args.reference_pdb:
        raise ValueError(
            f"Row {index + 1} has no run_dir/wt_pdb/mut_pdb. Provide --reference-pdb to build structures."
        )
    if not args.template_pdb:
        raise ValueError("Structure building requires --template-pdb.")

    name = str(row.get("name", f"row_{index + 1}")).strip() or f"row_{index + 1}"
    run_dir = os.path.join(output_dir, "built_structures", name)
    template_code = args.template[:-1]
    seed_dir = os.path.join(run_dir, "model_build", "modeling_job")
    os.makedirs(seed_dir, exist_ok=True)
    shutil.copy2(args.template_pdb, os.path.join(seed_dir, f"{template_code}.pdb"))
    build_cfg = BuildConfig(
        reference_pdb=args.reference_pdb,
        ligand_xml=args.ligand_xml,
        ligand_resnames=[args.ligand_resname],
        mutations=list(mutations),
        templates=[args.template],
        chain_id=args.chain_id,
        num_models=args.num_models,
        num_processors=args.num_processors,
        modeller_seed=args.modeller_seed_base + index,
    )
    prep = prepare_structures(build_cfg, output_dir=run_dir)
    return prep.wt_pdb, prep.mutant_ligand_pdb


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Score every variant of the input table and evaluate the predictions."""
    input_rows = _load_rows(args.input_csv)
    os.makedirs(args.output_dir, exist_ok=True)
    prediction_rows: List[Dict[str, object]] = []

    for index, row in enumerate(input_rows):
        name = str(row.get("name", f"row_{index + 1}")).strip() or f"row_{index + 1}"
        mutations = _split_mutation_tokens(row.get("mutations") or row.get("mutation_set"))
        if not mutations:
            raise ValueError(f"Row {index + 1} ({name}) has no mutations/mutation_set value.")
        wt_pdb, mut_pdb = _resolve_structure_pair(row, args.output_dir, index, args, mutations)
        scan_dir = os.path.join(args.output_dir, "scans", name)
        ligand_resname = str(row.get("ligand_resname", "")).strip() or args.ligand_resname
        scan = run_fast_scan(
            wt_pdb=wt_pdb,
            mut_pdb=mut_pdb,
            ligand_resname=ligand_resname,
            output_dir=scan_dir,
            mutation_label=name,
            mutation_specs=mutations,
            cutoff_ang=args.cutoff_ang,
            contact_threshold_ang=args.contact_threshold_ang,
            source_hotspots=args.source_hotspots,
            top_residues=args.top_residues,
        )
        pred: Dict[str, object] = dict(row)
        pred.update(
            {
                "name": name,
                "mutations": ",".join(mutations),
                "wt_pdb": wt_pdb,
                "mut_pdb": mut_pdb,
                "predicted_allosteric_effect": scan["predicted_allosteric_effect"],
                "global_fast_allostery_score": float(scan["global_fast_allostery_score"]),
                "mean_top10_destabilization_proxy": float(scan["mean_top10_destabilization_proxy"]),
                "aligned_ca_rmsd_ang": float(scan["aligned_ca_rmsd_ang"]),
                "mutation_source_residues": ",".join(scan.get("mutation_source_residues", [])),
                "source_residues": ",".join(scan["source_residues"]),
                "target_residues": ",".join(scan["target_residues"]),
                "deleted_wt_residues": ",".join(scan["deleted_wt_residues"]),
                "top_residues": ",".join(
                    item["node_label"] for item in scan["top_residues"][: min(args.top_residues, len(scan["top_residues"]))]
                ),
                "summary_json": scan["outputs"]["summary_json"],
            }
        )
        prediction_rows.append(pred)

    predictions_csv = os.path.join(args.output_dir, "benchmark_predictions.csv")
    write_csv(predictions_csv, prediction_rows)
    summary = _evaluate_predictions(
        prediction_rows,
        label_column=args.label_column,
        effect_column=args.effect_column,
        external_score_columns=args.external_score_column or [],
    )
    summary["input_csv"] = args.input_csv
    summary["predictions_csv"] = predictions_csv
    summary["method_note"] = (
        "Benchmark summary reports association/calibration only. Use independent held-out data "
        "before claiming superiority over FoldX/DDMut or any other method."
    )
    summary_json = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_png = os.path.join(args.output_dir, "benchmark_summary.png")
    plot_benchmark(prediction_rows, plot_png, args.label_column, args.effect_column)
    manifest = {
        "rows": len(prediction_rows),
        "predictions_csv": predictions_csv,
        "summary_json": summary_json,
        "summary_plot": plot_png,
    }
    manifest_path = os.path.join(args.output_dir, "benchmark_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
