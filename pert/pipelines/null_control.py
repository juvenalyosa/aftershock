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
``1 / (1 + n_null)``.  More replicas give a sharper null.

A metric *counts* for a scenario when every replica exceeds every WT-vs-WT
pair, and in addition

* the metric is not saturated in the null: if two WT simulations already reach
  the ceiling of a bounded metric (a total-variation distance of 0.999 out of
  1), exceeding them says nothing;
* every replica also exceeds the modelled-WT control, when the panel has one.
  A scenario whose "mutant" has the WT sequence (a silent substitution such as
  ``K2K``) is recognised as that control: it differs from the WT only by having
  been rebuilt by the modeller, so whatever it reaches is not a mutation effect.

Each scenario then gets an ``evidence`` level:

* ``none``       - no metric counts;
* ``suggestive`` - some metric counts, but there are too few replicas or null
  pairs for significance, or the simulations are not converged (fewer
  effective samples than ``MIN_EFFECTIVE_SAMPLES``: each trajectory mostly
  shows where it happened to sit);
* ``supported``  - some metric counts, with at least two replicas, enough null
  pairs for a per-run p <= 0.05, and converged simulations.  Five metrics are
  examined and no multiple-testing correction is applied, so treat a single
  supporting metric with care.

``recalibrate_null_control`` re-derives the verdicts of a panel from the pairs
already computed, without the trajectories (after a change of rules, or on a
copy of the results that does not carry them).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pert.analysis.ensemble import MIN_EFFECTIVE_SAMPLES
from pert.analysis.topology_safe import COMPARISON_METRICS, comparison_metrics, run_topology_safe_analysis
from pert.config import FieldConfig, LatentConfig, TEConfig
from pert.structure.residues import protein_ca_records
from pert.utils.io import load_json, read_csv, write_csv
from pert.utils.paths import find_run_dirs, replicate_from_run_dir, resolve_run_path, scenario_from_run_dir
from pert.visualization.panel_plots import plot_null_control

# A run can only reach p <= 0.05 against the null if there are at least this many pairs.
MIN_NULL_PAIRS_FOR_SIGNIFICANCE = 19

# Range of the bounded metrics, and how close to the ceiling the WT-vs-WT pairs
# may come (as a fraction of the range) before the metric is called saturated.
METRIC_RANGE = {
    "js_divergence": (0.0, math.log(2.0)),
    "classifier_auc": (0.5, 1.0),
    "basin_population_tv": (0.0, 1.0),
}
SATURATION_FRACTION = 0.01

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
    mut_sequence: str = ""

    @property
    def is_control(self) -> bool:
        """The "mutant" has the WT sequence: a modelled-WT control, not a mutation."""
        return bool(self.mut_sequence) and self.mut_sequence == self.wt_sequence

    @property
    def label(self) -> str:
        return f"{self.scenario}__{self.replicate}"


# --- inputs ----------------------------------------------------------------------------------


def _sequence(pdb_path: str) -> str:
    return "".join(str(rec["aa1"]) for rec in protein_ca_records(pdb_path))


def _collect_runs(panel_dir: str, require_trajectories: bool = True) -> Tuple[List[_Run], List[Dict[str, str]]]:
    """Runs of the panel that still have their files, and the ones that do not.

    The prepared structures are always needed (they give the sequences); the
    trajectories only when something has to be analysed.
    """
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
        needed = [name for name in paths if require_trajectories or name.endswith("_pdb")]
        missing = [name for name in needed if not os.path.exists(paths[name])]
        if missing:
            skipped.append({"run_dir": run_dir, "reason": f"missing files: {', '.join(missing)}"})
            continue
        runs.append(
            _Run(
                scenario=scenario_from_run_dir(run_dir),
                replicate=replicate_from_run_dir(run_dir),
                run_dir=run_dir,
                wt_sequence=_sequence(paths["wt_pdb"]),
                mut_sequence=_sequence(paths["mut_pdb"]),
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

    # --- observed: WT_i vs MUT_i -----------------------------------------------------------
    observed: List[Tuple[_Run, Dict[str, Any], str]] = []
    for run in runs:
        stored_path = os.path.join(run.run_dir, "analysis", "analysis_results.json")
        stored = load_json(stored_path) if os.path.exists(stored_path) else None
        if reuse_run_analysis and stored and _same_method(stored, field_cfg, latent_cfg, te_cfg, ligand_resnames):
            observed.append((run, stored, "run analysis"))
        else:
            results = analyse(run, run.mut_pdb, run.mut_npz, os.path.join(out_dir, "observed", run.label))
            observed.append((run, results, "re-analysed"))

    settings = {
        "ligand_resnames": ligand_resnames,
        "temperature_k": temperature_k,
        "field_config": asdict(field_cfg),
        "latent_config": asdict(latent_cfg),
        "te_config": {**asdict(te_cfg), "ridge_values": [float(v) for v in te_cfg.ridge_values]},
    }
    return _calibrate_and_write(panel_dir, out_dir, pair_rows, observed, settings, skipped, notes)


def recalibrate_null_control(panel_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Re-derive the verdicts of a panel from the WT-vs-WT pairs already computed.

    Nothing is re-analysed and no trajectory is read: the pairs come from
    ``null_control_pairs.csv`` and the observed values from the stored analysis
    of each run, exactly as the null control used them.  Use it after the
    rules for the verdicts change, or on a copy of the results without
    trajectories.
    """
    panel_dir = os.path.abspath(panel_dir)
    out_dir = os.path.abspath(output_dir or os.path.join(panel_dir, "null_control"))
    pairs_csv = os.path.join(out_dir, "null_control_pairs.csv")
    if not os.path.exists(pairs_csv):
        raise NullControlUnavailable(f"No null control to recalibrate: '{pairs_csv}' does not exist. Run `pert null-control` first.")
    pair_rows: List[Dict[str, Any]] = []
    for raw in read_csv(pairs_csv):
        row: Dict[str, Any] = dict(raw)
        for key in ("te_delta",) + tuple(COMPARISON_METRICS):
            if key in row:
                row[key] = float(row[key])
        pair_rows.append(row)
    if not pair_rows:
        raise NullControlUnavailable(f"'{pairs_csv}' holds no WT-vs-WT pair.")

    previous: Dict[str, Any] = {}
    previous_path = os.path.join(out_dir, "null_control_summary.json")
    if os.path.exists(previous_path):
        previous = load_json(previous_path)
    settings = {
        key: previous.get(key)
        for key in ("ligand_resnames", "temperature_k", "field_config", "latent_config", "te_config")
    }

    runs, skipped = _collect_runs(panel_dir, require_trajectories=False)
    observed: List[Tuple[_Run, Dict[str, Any], str]] = []
    for run in runs:
        # A run the null control had to re-analyse keeps that result in observed/.
        candidates = [
            (os.path.join(out_dir, "observed", run.label, "analysis_results.json"), "re-analysed"),
            (os.path.join(run.run_dir, "analysis", "analysis_results.json"), "run analysis"),
        ]
        for path, source in candidates:
            if os.path.exists(path):
                observed.append((run, load_json(path), source))
                break
        else:
            skipped.append({"run_dir": run.run_dir, "reason": "no stored analysis_results.json"})
    if not observed:
        raise NullControlUnavailable(f"No analysed run under {panel_dir}.")
    notes = ["Verdicts recalibrated from the stored WT-vs-WT pairs; nothing was re-analysed."]
    return _calibrate_and_write(panel_dir, out_dir, pair_rows, observed, settings, skipped, notes)


# --- verdicts ------------------------------------------------------------------------------------


def _saturated_in_null(metric: str, null_max: Optional[float]) -> bool:
    """Do two WT simulations already reach the ceiling of this (bounded) metric?"""
    if metric not in METRIC_RANGE or null_max is None or not np.isfinite(null_max):
        return False
    low, high = METRIC_RANGE[metric]
    return bool(null_max >= high - SATURATION_FRACTION * (high - low))


def scenario_verdicts(run_rows: Sequence[Dict[str, Any]], null_values: Dict[str, Sequence[float]]) -> Dict[str, Dict[str, Any]]:
    """Evidence level of every scenario (rules in the module docstring).

    ``run_rows`` are the calibrated runs: ``scenario``, ``control``,
    ``sampling_ok``, ``effective_samples_min`` and, per metric, the value and
    its ``_z``, ``_p`` and ``_beyond_null``.
    """
    n_null = max((len(values) for values in null_values.values()), default=0)
    null_max = {
        m: (float(np.nanmax(values)) if len(values) and np.any(np.isfinite(values)) else None)
        for m, values in null_values.items()
    }
    controls = sorted({str(row["scenario"]) for row in run_rows if row.get("control")})
    control_max = {
        m: max((float(row[m]) for row in run_rows if row.get("control") and np.isfinite(row[m])), default=None)
        for m in COMPARISON_METRICS
    }

    scenarios: Dict[str, Dict[str, Any]] = {}
    for scenario in sorted({str(row["scenario"]) for row in run_rows}):
        rows = [row for row in run_rows if row["scenario"] == scenario]
        is_control = scenario in controls
        entry: Dict[str, Any] = {"replicas": len(rows), "is_control": is_control, "metrics": {}}
        beyond: List[str] = []
        supporting: List[str] = []
        excluded: Dict[str, str] = {}
        for m in COMPARISON_METRICS:
            values = np.asarray([row[m] for row in rows], dtype=float)
            z = np.asarray([np.nan if row[f"{m}_z"] is None else row[f"{m}_z"] for row in rows], dtype=float)
            p = [row[f"{m}_p"] for row in rows if row[f"{m}_p"] is not None]
            all_beyond = bool(all(row[f"{m}_beyond_null"] for row in rows))
            saturated = _saturated_in_null(m, null_max.get(m))
            beyond_control: Optional[bool] = None
            if controls and not is_control and control_max[m] is not None:
                beyond_control = bool(all(float(row[m]) > control_max[m] for row in rows))
            entry["metrics"][m] = {
                "mean": float(np.nanmean(values)) if np.any(np.isfinite(values)) else None,
                "mean_z": float(np.nanmean(z)) if np.any(np.isfinite(z)) else None,
                "max_p_empirical": max(p) if p else None,  # the least convincing replica
                "replicas_beyond_null": int(sum(row[f"{m}_beyond_null"] for row in rows)),
                "all_replicas_beyond_null": all_beyond,
                "saturated_in_null": saturated,
                "control_max": None if is_control else control_max[m] if controls else None,
                "all_replicas_beyond_control": beyond_control,
            }
            if not all_beyond:
                continue
            beyond.append(m)
            if saturated:
                excluded[m] = (
                    f"saturated: two WT simulations already reach {null_max[m]:.3f} "
                    f"of a possible {METRIC_RANGE[m][1]:.3f}"
                )
            elif beyond_control is False:
                excluded[m] = f"the modelled-WT control ({', '.join(controls)}) reaches {control_max[m]:.3g}"
            else:
                supporting.append(m)

        n_eff = min((float(row["effective_samples_min"]) for row in rows), default=float("nan"))
        converged = bool(all(row.get("sampling_ok", False) for row in rows))
        entry["metrics_beyond_null"] = beyond
        entry["metrics_supporting"] = supporting
        entry["metrics_excluded"] = excluded
        entry["sampling_converged"] = converged
        entry["effective_samples_min"] = n_eff

        prefix = "Control (the 'mutant' has the WT sequence): " if is_control else ""
        if not beyond:
            entry["evidence"] = "none"
            note = "No metric separates this scenario from pairs of WT simulations."
        elif not supporting:
            entry["evidence"] = "none"
            reasons = "; ".join(f"{m}: {why}" for m, why in excluded.items())
            note = f"Exceeds every WT-vs-WT pair in {', '.join(beyond)}, but that does not count ({reasons})."
        else:
            shortfalls: List[str] = []
            if len(rows) < 2 or n_null < MIN_NULL_PAIRS_FOR_SIGNIFICANCE:
                shortfalls.append(
                    f"{len(rows)} replica(s) and {n_null} null pair(s) are not enough for significance "
                    f"(need >= 2 replicas and >= {MIN_NULL_PAIRS_FOR_SIGNIFICANCE} pairs)"
                )
            if not converged:
                shortfalls.append(
                    f"the simulations are not converged (as few as {n_eff:.1f} effective samples, "
                    f"need >= {MIN_EFFECTIVE_SAMPLES:g})"
                )
            where = f"all {len(rows)} replica(s) exceed all {n_null} WT-vs-WT pairs"
            if controls and not is_control:
                where += f" and the modelled-WT control ({', '.join(controls)})"
            if shortfalls:
                entry["evidence"] = "suggestive"
                note = f"In {', '.join(supporting)}, {where}, but {'; and '.join(shortfalls)}."
            else:
                entry["evidence"] = "supported"
                note = (
                    f"In {', '.join(supporting)}, {where} (per-run p <= {1.0 / (1 + n_null):.3f}, "
                    f"uncorrected for {len(COMPARISON_METRICS)} metrics)."
                )
            if excluded:
                note += " Not counted: " + "; ".join(f"{m} ({why})" for m, why in excluded.items()) + "."
            if is_control:
                note += " Model building alone produces a difference here: read the mutants against this scenario."
        entry["evidence_note"] = prefix + note
        scenarios[scenario] = entry
    return scenarios


def _calibrate_and_write(
    panel_dir: str,
    out_dir: str,
    pair_rows: List[Dict[str, Any]],
    observed: Sequence[Tuple[_Run, Dict[str, Any], str]],
    settings: Dict[str, Any],
    skipped: List[Dict[str, str]],
    notes: List[str],
) -> Dict[str, Any]:
    """Place every observed run on the null, derive the verdicts and write the outputs."""
    null_values = {m: [float(row[m]) for row in pair_rows] for m in COMPARISON_METRICS}

    run_rows: List[Dict[str, Any]] = []
    for run, results, source in observed:
        metrics = comparison_metrics(results)
        basins = results["basin_ddg"]
        n_eff = float(min(basins["effective_samples_wt"], basins["effective_samples_mut"]))
        row: Dict[str, Any] = {
            "scenario": run.scenario,
            "replicate": run.replicate,
            "control": run.is_control,
            "observed_from": source,
            "te_delta": metrics["te_delta"],
            "effective_samples_min": n_eff,
            "sampling_ok": bool(basins.get("sampling_ok", n_eff >= MIN_EFFECTIVE_SAMPLES)),
        }
        for m in COMPARISON_METRICS:
            cal = calibrate_against_null(metrics[m], null_values[m])
            row[m] = metrics[m]
            row[f"{m}_z"] = cal["z"]
            row[f"{m}_p"] = cal["p_empirical"]
            row[f"{m}_beyond_null"] = bool(cal["null_max"] is not None and metrics[m] > cal["null_max"])
        run_rows.append(row)

    scenarios = scenario_verdicts(run_rows, null_values)
    controls = [name for name, entry in scenarios.items() if entry["is_control"]]

    n_null = len(pair_rows)
    n_wt = len({label for row in pair_rows for label in (row.get("wt_a"), row.get("wt_b")) if label}) or len(run_rows)
    notes = list(notes)
    notes.append(
        f"{n_null} WT-vs-WT pair(s) from {n_wt} WT trajectories: the smallest attainable "
        f"empirical p-value is {1.0 / (1 + n_null):.3f}."
    )
    if n_null < MIN_NULL_PAIRS_FOR_SIGNIFICANCE:
        notes.append(
            f"Fewer than {MIN_NULL_PAIRS_FOR_SIGNIFICANCE} null pairs: no single run can reach p <= 0.05 and "
            "z-scores are unstable. Seven WT trajectories (e.g. 4 scenarios x 2 replicas) are enough."
        )
    saturated = [m for m in COMPARISON_METRICS if _saturated_in_null(m, float(np.nanmax(null_values[m])))]
    if saturated:
        notes.append(
            f"Saturated in the null (two WT simulations already reach the ceiling), so not usable: {', '.join(saturated)}."
        )
    unconverged = sum(1 for row in run_rows if not row["sampling_ok"])
    if unconverged:
        notes.append(
            f"{unconverged} of {len(run_rows)} runs are not converged (fewer than {MIN_EFFECTIVE_SAMPLES:g} effective "
            "samples): no scenario can be more than suggestive. Longer simulations are needed."
        )
    if controls:
        notes.append(f"Modelled-WT control: {', '.join(controls)}. A metric counts for a mutant only above it.")
    else:
        notes.append(
            "No modelled-WT control in this panel (a silent substitution such as K2K): effects of model "
            "building cannot be told from effects of the mutation."
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
        "wt_trajectories": n_wt,
        "null_pairs": n_null,
        "runs_reanalysed": sum(1 for _, _, source in observed if source == "re-analysed"),
        **settings,
        "null": {
            m: {
                "values": null_values[m],
                "mean": float(np.nanmean(null_values[m])),
                "max": float(np.nanmax(null_values[m])),
                "saturated": m in saturated,
            }
            for m in COMPARISON_METRICS
        },
        "control_scenarios": controls,
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
