"""WT-vs-WT null control: how different do two simulations look with no mutation?

Every estimand of the WT-vs-MUT comparison is non-zero even when nothing was
mutated, because two finite simulations never sample exactly the same
ensemble.  A value therefore means nothing until it is compared with that
baseline.

Each run of a panel contains its own, independently seeded WT simulation, so a
panel with N runs holds N WT replicas for free.  This stage

1. runs the *same* analysis, with the *same* settings, on pairs of WT
   trajectories (WT_i as "WT", WT_j as "mutant") to obtain the null
   distribution of every comparison metric;
2. places each real WT-vs-MUT result on that distribution: z-score and
   empirical p-value ``(1 + #null >= observed) / (1 + n_null)``.

Outputs, in ``<panel_dir>/null_control`` by default:

* ``null_control_pairs.csv``      metrics of every WT-vs-WT pair
* ``null_calibrated_metrics.csv`` every run with its z-scores and p-values
* ``null_control_summary.json``   settings, null statistics, per-scenario verdicts
* ``null_control.png``            null vs observed, one panel per metric
* ``null/`` and ``observed/``     the underlying analysis results

With few WT replicas the test is coarse: the smallest attainable p-value is
``1 / (1 + n_null)``.  More replicas give a sharper null.  Each scenario gets
an ``evidence`` level that says so explicitly:

* ``none``       - no metric separates the scenario from the WT-vs-WT pairs;
* ``suggestive`` - in at least one metric every replica exceeds every null
  pair, but there are too few null pairs or replicas to call it significant;
* ``supported``  - the same, with at least two replicas and enough null pairs
  for a per-run p <= 0.05.  Five metrics are examined and no multiple-testing
  correction is applied, so treat a single supporting metric with care.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pert.analysis.topology_safe import COMPARISON_METRICS, comparison_metrics, run_topology_safe_analysis
from pert.config import FieldConfig, LatentConfig, TEConfig
from pert.structure.residues import protein_ca_records
from pert.utils.io import load_json, write_csv
from pert.utils.paths import find_run_dirs, replicate_from_run_dir, resolve_run_path, scenario_from_run_dir
from pert.visualization.panel_plots import plot_null_control

# A run can only reach p <= 0.05 against the null if there are at least this many pairs.
MIN_NULL_PAIRS_FOR_SIGNIFICANCE = 19

# TE settings assumed for panels analysed before ``te_config`` was recorded
# (they are the defaults of ``pert md-panel``).
_LEGACY_PANEL_TE = {"max_order": 3, "surrogates": 64}


class NullControlUnavailable(RuntimeError):
    """The panel does not contain two comparable WT trajectories."""


@dataclass
class _Run:
    """Inputs of one finished run that the null control needs."""

    scenario: str
    replicate: str
    run_dir: str
    wt_pdb: str
    wt_npz: str
    mut_pdb: str
    mut_npz: str
    wt_sequence: str

    @property
    def label(self) -> str:
        return f"{self.scenario}__{self.replicate}"


# --- inputs ----------------------------------------------------------------------------------


def _collect_runs(panel_dir: str) -> Tuple[List[_Run], List[Dict[str, str]]]:
    """Runs of the panel that still have both trajectories, and the ones that do not."""
    runs: List[_Run] = []
    skipped: List[Dict[str, str]] = []
    for run_dir in find_run_dirs(panel_dir):
        run_dir = os.path.abspath(run_dir)
        try:
            md = load_json(os.path.join(run_dir, "run_summary.json"))["md"]
            paths = {
                "wt_pdb": resolve_run_path(run_dir, md["wt"]["prepared_pdb"]),
                "wt_npz": resolve_run_path(run_dir, md["wt"]["traj_npz"]),
                "mut_pdb": resolve_run_path(run_dir, md["mut"]["prepared_pdb"]),
                "mut_npz": resolve_run_path(run_dir, md["mut"]["traj_npz"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append({"run_dir": run_dir, "reason": f"unreadable run_summary.json ({exc})"})
            continue
        missing = [name for name, path in paths.items() if not os.path.exists(path)]
        if missing:
            skipped.append({"run_dir": run_dir, "reason": f"missing files: {', '.join(missing)}"})
            continue
        sequence = "".join(str(rec["aa1"]) for rec in protein_ca_records(paths["wt_pdb"]))
        runs.append(
            _Run(
                scenario=scenario_from_run_dir(run_dir),
                replicate=replicate_from_run_dir(run_dir),
                run_dir=run_dir,
                wt_sequence=sequence,
                **paths,
            )
        )
    return runs, skipped


def _dataclass_from_recorded(cls, recorded: Optional[Dict[str, Any]], **overrides):
    """Rebuild a config dataclass from the dict stored in ``analysis_results.json``."""
    known = {f.name for f in fields(cls)}
    values = {k: v for k, v in (recorded or {}).items() if k in known}
    values.update({k: v for k, v in overrides.items() if v is not None})
    if "ridge_values" in values:
        values["ridge_values"] = tuple(float(v) for v in values["ridge_values"])
    return cls(**values)


def _te_signature(te_cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """TE settings that define the method (the random seed does not)."""
    out = {k: v for k, v in te_cfg_dict.items() if k != "random_seed"}
    out["ridge_values"] = [float(v) for v in out.get("ridge_values", [])]
    return out


def _same_method(
    results: Dict[str, Any],
    field_cfg: FieldConfig,
    latent_cfg: LatentConfig,
    te_cfg: TEConfig,
    ligand_resnames: Sequence[str],
) -> bool:
    """Was this stored result computed with exactly the settings of the null control?"""
    try:
        return (
            results.get("field_config") == asdict(field_cfg)
            and results.get("latent_config") == asdict(latent_cfg)
            and _te_signature(results["te_config"]) == _te_signature(asdict(te_cfg))
            and [str(n).upper() for n in results["ligand_resnames"]] == [str(n).upper() for n in ligand_resnames]
            and "classifier_auc" in results["distribution_shift"]
        )
    except (KeyError, TypeError):
        return False


# --- statistics --------------------------------------------------------------------------------


def calibrate_against_null(observed: float, null_values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Place one observed value on the null distribution (larger = more different).

    Returns the null mean/std/max, the z-score (None when the null has no
    spread) and the empirical one-sided p-value
    ``(1 + #null >= observed) / (1 + n_null)``.
    """
    null = np.asarray([v for v in null_values if np.isfinite(v)], dtype=float)
    if null.size == 0 or not np.isfinite(observed):
        return {"n_null": int(null.size), "null_mean": None, "null_std": None, "null_max": None, "z": None, "p_empirical": None}
    mean = float(np.mean(null))
    std = float(np.std(null, ddof=1)) if null.size > 1 else float("nan")
    z = float((observed - mean) / std) if np.isfinite(std) and std > 1e-12 else None
    p = float((1 + int(np.sum(null >= observed))) / (1 + null.size))
    return {
        "n_null": int(null.size),
        "null_mean": mean,
        "null_std": None if not np.isfinite(std) else std,
        "null_max": float(np.max(null)),
        "z": z,
        "p_empirical": p,
    }


def _select_pairs(runs: Sequence[_Run], max_pairs: int, rng: np.random.Generator) -> List[Tuple[_Run, _Run]]:
    """Unordered pairs of WT trajectories of the same protein (a random subset if too many)."""
    pairs = [(a, b) for a, b in combinations(runs, 2) if a.wt_sequence == b.wt_sequence]
    if max_pairs > 0 and len(pairs) > max_pairs:
        keep = sorted(rng.choice(len(pairs), size=max_pairs, replace=False).tolist())
        pairs = [pairs[i] for i in keep]
    return pairs


# --- stage ---------------------------------------------------------------------------------------


def run_null_control(
    panel_dir: str,
    ligand_resnames: Optional[Sequence[str]] = None,
    field_cfg: Optional[FieldConfig] = None,
    latent_cfg: Optional[LatentConfig] = None,
    te_cfg: Optional[TEConfig] = None,
    temperature_k: Optional[float] = None,
    output_dir: Optional[str] = None,
    max_pairs: int = 30,
    reuse_run_analysis: bool = True,
    random_seed: int = 13,
) -> Dict[str, Any]:
    """Calibrate every run of a panel against the WT-vs-WT null (see module docstring).

    Settings left as ``None`` are taken from the ``analysis_results.json`` of
    the first run, so the null is computed the way the panel was analysed.  A
    run whose stored result used different settings (for instance a panel
    analysed with an older version) is re-analysed into ``observed/`` rather
    than compared across methods; its own ``analysis/`` directory is never
    modified.
    """
    panel_dir = os.path.abspath(panel_dir)
    out_dir = os.path.abspath(output_dir or os.path.join(panel_dir, "null_control"))
    runs, skipped = _collect_runs(panel_dir)
    if not runs:
        raise NullControlUnavailable(f"No run with WT and mutant trajectories under {panel_dir}.")

    # --- settings: explicit arguments win, then what the panel recorded -------------------
    notes: List[str] = []
    recorded: Dict[str, Any] = {}
    first_results = os.path.join(runs[0].run_dir, "analysis", "analysis_results.json")
    if os.path.exists(first_results):
        recorded = load_json(first_results)
    if field_cfg is None:
        field_cfg = _dataclass_from_recorded(FieldConfig, recorded.get("field_config"))
        if "frame_mode" not in (recorded.get("field_config") or {}):
            notes.append(
                "The panel was analysed before the shared site frame existed; observed results are "
                "re-analysed with frame_mode='shared_site' so that they are comparable with the null."
            )
    if latent_cfg is None:
        latent_cfg = _dataclass_from_recorded(LatentConfig, recorded.get("latent_config"))
    if te_cfg is None:
        if recorded.get("te_config"):
            te_cfg = _dataclass_from_recorded(TEConfig, recorded["te_config"])
        else:
            te_cfg = TEConfig(**_LEGACY_PANEL_TE)
            notes.append(
                "TE settings were not recorded by this panel; assumed the `pert md-panel` defaults "
                f"({_LEGACY_PANEL_TE}). Pass them explicitly if the panel used others."
            )
    if temperature_k is None:
        temperature_k = float(recorded.get("temperature_k", 300.0))
    if not ligand_resnames:
        ligand_resnames = recorded.get("ligand_resnames")
    if not ligand_resnames:
        raise ValueError("The ligand residue name is not recorded in this panel; pass ligand_resnames.")
    ligand_resnames = [str(name) for name in ligand_resnames]

    def analyse(wt: _Run, mut_pdb: str, mut_npz: str, target_dir: str) -> Dict[str, Any]:
        return run_topology_safe_analysis(
            wt_pdb=wt.wt_pdb,
            wt_traj_npz=wt.wt_npz,
            mut_pdb=mut_pdb,
            mut_traj_npz=mut_npz,
            ligand_resnames=ligand_resnames,
            temperature_k=temperature_k,
            field_cfg=field_cfg,
            latent_cfg=latent_cfg,
            te_cfg=te_cfg,
            output_dir=target_dir,
        )

    # --- null: WT_i vs WT_j ----------------------------------------------------------------
    rng = np.random.default_rng(random_seed)
    pairs = _select_pairs(runs, max_pairs, rng)
    if not pairs:
        raise NullControlUnavailable(
            f"Need at least two WT trajectories of the same protein; found {len(runs)} usable run(s). "
            "Run more scenarios or replicas."
        )
    pair_rows: List[Dict[str, Any]] = []
    for wt_a, wt_b in pairs:
        name = f"{wt_a.label}__vs__{wt_b.label}"
        results = analyse(wt_a, wt_b.wt_pdb, wt_b.wt_npz, os.path.join(out_dir, "null", name))
        pair_rows.append({"pair": name, "wt_a": wt_a.label, "wt_b": wt_b.label, **comparison_metrics(results)})
    null_values = {m: [float(row[m]) for row in pair_rows] for m in COMPARISON_METRICS}

    # --- observed: WT_i vs MUT_i -----------------------------------------------------------
    run_rows: List[Dict[str, Any]] = []
    reanalysed = 0
    for run in runs:
        stored_path = os.path.join(run.run_dir, "analysis", "analysis_results.json")
        stored = load_json(stored_path) if os.path.exists(stored_path) else None
        if reuse_run_analysis and stored and _same_method(stored, field_cfg, latent_cfg, te_cfg, ligand_resnames):
            results, source = stored, "run analysis"
        else:
            results = analyse(run, run.mut_pdb, run.mut_npz, os.path.join(out_dir, "observed", run.label))
            source, reanalysed = "re-analysed", reanalysed + 1
        metrics = comparison_metrics(results)
        row: Dict[str, Any] = {
            "scenario": run.scenario,
            "replicate": run.replicate,
            "observed_from": source,
            "te_delta": metrics["te_delta"],
            "effective_samples_min": float(
                min(results["basin_ddg"]["effective_samples_wt"], results["basin_ddg"]["effective_samples_mut"])
            ),
        }
        for m in COMPARISON_METRICS:
            cal = calibrate_against_null(metrics[m], null_values[m])
            row[m] = metrics[m]
            row[f"{m}_z"] = cal["z"]
            row[f"{m}_p"] = cal["p_empirical"]
            row[f"{m}_beyond_null"] = bool(cal["null_max"] is not None and metrics[m] > cal["null_max"])
        run_rows.append(row)

    # --- per-scenario verdicts ---------------------------------------------------------------
    scenarios: Dict[str, Dict[str, Any]] = {}
    for scenario in sorted({row["scenario"] for row in run_rows}):
        rows = [row for row in run_rows if row["scenario"] == scenario]
        entry: Dict[str, Any] = {"replicas": len(rows), "metrics": {}}
        beyond = []
        for m in COMPARISON_METRICS:
            values = np.asarray([row[m] for row in rows], dtype=float)
            z = np.asarray([np.nan if row[f"{m}_z"] is None else row[f"{m}_z"] for row in rows], dtype=float)
            p = [row[f"{m}_p"] for row in rows if row[f"{m}_p"] is not None]
            all_beyond = bool(all(row[f"{m}_beyond_null"] for row in rows))
            entry["metrics"][m] = {
                "mean": float(np.nanmean(values)) if np.any(np.isfinite(values)) else None,
                "mean_z": float(np.nanmean(z)) if np.any(np.isfinite(z)) else None,
                "max_p_empirical": max(p) if p else None,  # the least convincing replica
                "replicas_beyond_null": int(sum(row[f"{m}_beyond_null"] for row in rows)),
                "all_replicas_beyond_null": all_beyond,
            }
            if all_beyond:
                beyond.append(m)
        entry["metrics_beyond_null"] = beyond
        if not beyond:
            entry["evidence"] = "none"
            entry["evidence_note"] = "No metric separates this scenario from pairs of WT simulations."
        elif len(rows) >= 2 and len(pair_rows) >= MIN_NULL_PAIRS_FOR_SIGNIFICANCE:
            entry["evidence"] = "supported"
            entry["evidence_note"] = (
                f"All {len(rows)} replicas exceed all {len(pair_rows)} WT-vs-WT pairs in: {', '.join(beyond)} "
                f"(per-run p <= {1.0 / (1 + len(pair_rows)):.3f}, uncorrected for {len(COMPARISON_METRICS)} metrics)."
            )
        else:
            entry["evidence"] = "suggestive"
            entry["evidence_note"] = (
                f"Exceeds every WT-vs-WT pair in: {', '.join(beyond)}, but with {len(rows)} replica(s) and "
                f"{len(pair_rows)} null pair(s) this is not significant (need >= 2 replicas and "
                f">= {MIN_NULL_PAIRS_FOR_SIGNIFICANCE} pairs)."
            )
        scenarios[scenario] = entry

    n_null = len(pair_rows)
    notes.append(
        f"{n_null} WT-vs-WT pair(s) from {len(runs)} WT trajectories: the smallest attainable "
        f"empirical p-value is {1.0 / (1 + n_null):.3f}."
    )
    if n_null < MIN_NULL_PAIRS_FOR_SIGNIFICANCE:
        notes.append(
            f"Fewer than {MIN_NULL_PAIRS_FOR_SIGNIFICANCE} null pairs: no single run can reach p <= 0.05 and "
            "z-scores are unstable. Seven WT trajectories (e.g. 4 scenarios x 2 replicas) are enough."
        )

    os.makedirs(out_dir, exist_ok=True)
    pairs_csv = os.path.join(out_dir, "null_control_pairs.csv")
    runs_csv = os.path.join(out_dir, "null_calibrated_metrics.csv")
    summary_json = os.path.join(out_dir, "null_control_summary.json")
    plot_png = os.path.join(out_dir, "null_control.png")
    write_csv(pairs_csv, pair_rows)
    write_csv(runs_csv, run_rows)
    observed_by_scenario = {
        scenario: {m: [float(row[m]) for row in run_rows if row["scenario"] == scenario] for m in COMPARISON_METRICS}
        for scenario in scenarios
    }
    plot_null_control(COMPARISON_METRICS, null_values, observed_by_scenario, plot_png)

    summary: Dict[str, Any] = {
        "panel_dir": panel_dir,
        "wt_trajectories": len(runs),
        "null_pairs": n_null,
        "runs_reanalysed": reanalysed,
        "ligand_resnames": ligand_resnames,
        "temperature_k": temperature_k,
        "field_config": asdict(field_cfg),
        "latent_config": asdict(latent_cfg),
        "te_config": {**asdict(te_cfg), "ridge_values": [float(v) for v in te_cfg.ridge_values]},
        "null": {
            m: {
                "values": null_values[m],
                "mean": float(np.nanmean(null_values[m])),
                "max": float(np.nanmax(null_values[m])),
            }
            for m in COMPARISON_METRICS
        },
        "scenarios": scenarios,
        "skipped_runs": skipped,
        "notes": notes,
        "outputs": {
            "pairs_csv": pairs_csv,
            "calibrated_csv": runs_csv,
            "summary_json": summary_json,
            "plot": plot_png,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
