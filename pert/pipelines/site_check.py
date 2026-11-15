"""Metal-site integrity of every simulation of a panel (or of one run).

Run it before reading any WT-vs-MUT result of a metalloprotein: if the site
falls apart in some simulations, the comparison measures that, not the
mutation.  See ``pert.analysis.metal_site`` for the definitions.

Outputs, in ``<panel_dir>/site_check`` by default:

* ``site_check_simulations.csv``  one row per simulation and metal ion
* ``site_check_donors.csv``       one row per ligating atom
* ``site_check_summary.json``     counts per status and the settings used
* ``site_check.png``              donor-metal distances against time
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from pert.analysis.metal_site import metal_site_series
from pert.utils.io import load_json, write_csv
from pert.utils.paths import find_run_dirs, replicate_from_run_dir, resolve_run_path, scenario_from_run_dir
from pert.visualization.panel_plots import plot_site_check


def run_site_check(
    panel_dir: str,
    ligand_resnames: Optional[Sequence[str]] = None,
    shell_cutoff_ang: float = 2.6,
    bound_cutoff_ang: float = 3.0,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Follow the metal coordination in every WT and mutant trajectory under ``panel_dir``."""
    panel_dir = os.path.abspath(panel_dir)
    out_dir = os.path.abspath(output_dir or os.path.join(panel_dir, "site_check"))

    simulations: List[Dict[str, Any]] = []
    donor_rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for run_dir in find_run_dirs(panel_dir):
        run_dir = os.path.abspath(run_dir)
        scenario = scenario_from_run_dir(run_dir)
        replicate = replicate_from_run_dir(run_dir)
        try:
            md = load_json(os.path.join(run_dir, "run_summary.json"))["md"]
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append({"run_dir": run_dir, "reason": f"unreadable run_summary.json ({exc})"})
            continue
        for system in ("wt", "mut"):
            try:
                pdb = resolve_run_path(run_dir, md[system]["prepared_pdb"])
                npz = resolve_run_path(run_dir, md[system]["traj_npz"])
            except (KeyError, TypeError) as exc:
                skipped.append({"run_dir": run_dir, "reason": f"{system}: paths not recorded ({exc})"})
                continue
            if not (os.path.exists(pdb) and os.path.exists(npz)):
                skipped.append({"run_dir": run_dir, "reason": f"{system}: structure or trajectory file missing"})
                continue
            label = f"{scenario}/{replicate}/{system}"
            for site in metal_site_series(pdb, npz, ligand_resnames, shell_cutoff_ang, bound_cutoff_ang):
                simulations.append(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "system": system,
                        "metal": site["metal"],
                        "status": site["status"],
                        "initial_donors": len(site["donors"]),
                        "donors_kept": sum(1 for d in site["donors"] if d["fraction_bound_last_fifth"] >= 0.5),
                        "mean_protein_donors_bound": site["mean_protein_donors_bound"],
                        "mean_waters_bound": site["mean_waters_bound"],
                        "frames": site["frames"],
                        "length_ns": site["time_ns"][-1] if site["time_ns"] else 0.0,
                    }
                )
                for donor in site["donors"]:
                    donor_rows.append(
                        {
                            "scenario": scenario,
                            "replicate": replicate,
                            "system": system,
                            "metal": site["metal"],
                            **{key: value for key, value in donor.items() if key != "distance_ang"},
                        }
                    )
                traces.append({"label": label, "metal": site["metal"], "status": site["status"], "time_ns": site["time_ns"],
                               "donors": {d["donor"]: d["distance_ang"] for d in site["donors"]}})

    if not simulations:
        reason = "; ".join(sorted({item["reason"] for item in skipped})) or "no run_summary.json found"
        raise RuntimeError(f"No metal site could be followed under {panel_dir} ({reason}).")

    os.makedirs(out_dir, exist_ok=True)
    simulations_csv = os.path.join(out_dir, "site_check_simulations.csv")
    donors_csv = os.path.join(out_dir, "site_check_donors.csv")
    summary_json = os.path.join(out_dir, "site_check_summary.json")
    plot_png = os.path.join(out_dir, "site_check.png")
    write_csv(simulations_csv, simulations)
    write_csv(donors_csv, donor_rows)
    plot_site_check(traces, bound_cutoff_ang, plot_png)

    counts: Dict[str, int] = {}
    for row in simulations:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    by_system = {
        system: {
            status: sum(1 for row in simulations if row["system"] == system and row["status"] == status)
            for status in sorted(counts)
        }
        for system in ("wt", "mut")
    }
    summary: Dict[str, Any] = {
        "panel_dir": panel_dir,
        "simulations": len(simulations),
        "status_counts": counts,
        "status_counts_by_system": by_system,
        "all_intact": bool(counts.get("intact", 0) == len(simulations)),
        "shell_cutoff_ang": float(shell_cutoff_ang),
        "bound_cutoff_ang": float(bound_cutoff_ang),
        "ligand_resnames": list(ligand_resnames or []),
        "skipped": skipped,
        "outputs": {
            "simulations_csv": simulations_csv,
            "donors_csv": donors_csv,
            "summary_json": summary_json,
            "plot": plot_png,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
