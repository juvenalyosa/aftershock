"""VMD scenes that highlight ranked allosteric residues.

Colour convention in all three scenes: top-ranked residues, mutation sources
and binding-site targets are drawn as separate representations on the mutant
structure.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pert.utils.io import load_json
from pert.utils.paths import resolve_run_path
from pert.visualization.vmd_common import (
    selection_from_consensus_rows,
    selection_from_rows,
    tcl_braced_path,
)


def write_hotspot_vmd(
    mut_pdb: str,
    mut_dcd: Optional[str],
    ligand_resname: str,
    rows: Sequence[Dict[str, object]],
    source_rows: Sequence[Dict[str, object]],
    target_rows: Sequence[Dict[str, object]],
    out_tcl: str,
    top_n: int,
) -> str:
    """Write the scene of the MD-ranked residues of one run."""
    draw_rows = list(rows[: max(1, int(top_n))])
    hotspot_sel = selection_from_rows(draw_rows)
    source_sel = selection_from_rows(source_rows)
    target_sel = selection_from_rows(target_rows)

    label_lines: List[str] = []
    for rank, row in enumerate(draw_rows[: min(20, len(draw_rows))], start=1):
        coord_nm = np.asarray(row["mut_coord_nm"], dtype=float)
        coord_ang = 10.0 * coord_nm
        label = f"{rank}. {row['node_label']} {float(row['total_score']):.3f}"
        label = label.replace("{", "").replace("}", "")
        label_lines.append(
            f"graphics $mid text {{{coord_ang[0]:.3f} {coord_ang[1]:.3f} {coord_ang[2]:.3f}}} {{{label}}} size 0.9 thickness 2"
        )

    lines = [
        "# Auto-generated allostery hotspot view for VMD",
        "display projection Orthographic",
        "display depthcue off",
        "axes location Off",
        "color Display Background white",
        f"set mut_pdb {tcl_braced_path(mut_pdb)}",
        f"set mut_dcd {tcl_braced_path(mut_dcd)}",
        f"set ligand_sel {{resname {ligand_resname.upper()}}}",
        f"set hotspot_sel {{{hotspot_sel}}}",
        f"set source_sel {{{source_sel}}}",
        f"set target_sel {{{target_sel}}}",
        "",
        "mol new $mut_pdb type pdb waitfor all",
        "set mid [molinfo top]",
        "if {[string length $mut_dcd] > 0 && [file exists $mut_dcd]} {",
        "    mol addfile $mut_dcd type dcd waitfor all molid $mid",
        "}",
        "mol delrep 0 $mid",
        "mol representation NewCartoon 0.22 24.0 4.2 0",
        "mol color ColorID 8",
        "mol selection {protein}",
        "mol material AOChalky",
        "mol addrep $mid",
        "",
        "mol representation Licorice 0.24 20.0 20.0",
        "mol color Name",
        "mol selection $ligand_sel",
        "mol material Glossy",
        "mol addrep $mid",
    ]
    if hotspot_sel != "none":
        lines += [
            "",
            "mol representation VDW 0.52 24.0",
            "mol color ColorID 1",
            "mol selection ($hotspot_sel) and name CA",
            "mol material Glossy",
            "mol addrep $mid",
            "mol representation Licorice 0.20 18.0 18.0",
            "mol color ColorID 1",
            "mol selection ($hotspot_sel) and not hydrogen",
            "mol material Glossy",
            "mol addrep $mid",
        ]
    if source_sel != "none":
        lines += [
            "",
            "mol representation VDW 0.68 24.0",
            "mol color ColorID 4",
            "mol selection ($source_sel) and name CA",
            "mol material Glossy",
            "mol addrep $mid",
        ]
    if target_sel != "none":
        lines += [
            "",
            "mol representation QuickSurf 1.0 0.5 1.0 1.0",
            "mol color ColorID 10",
            "mol selection $target_sel",
            "mol material Transparent",
            "mol addrep $mid",
        ]
    lines += [
        "",
        "graphics $mid color black",
        *label_lines,
        "display resetview",
        "puts \"Loaded allostery hotspot view.\"",
        "puts \"Red: top ranked allosteric residues. Yellow: mutation source hotspots. Cyan surface: target/binding site.\"",
    ]

    os.makedirs(os.path.dirname(out_tcl) or ".", exist_ok=True)
    with open(out_tcl, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return out_tcl


def write_fast_scan_vmd(
    mut_pdb: str,
    ligand_resname: str,
    rows_ranked: Sequence[Dict[str, Any]],
    source_rows: Sequence[Dict[str, Any]],
    target_rows: Sequence[Dict[str, Any]],
    out_tcl: str,
    top_n: int,
) -> str:
    """Write the scene of the fast-scan ranking."""
    top_rows = list(rows_ranked[: max(1, int(top_n))])
    hotspot_sel = selection_from_rows(top_rows)
    source_sel = selection_from_rows(source_rows)
    target_sel = selection_from_rows(target_rows)
    lines = [
        "# Auto-generated fast allostery scan view for VMD",
        "display projection Orthographic",
        "display depthcue off",
        "axes location Off",
        "color Display Background white",
        f"set mut_pdb {tcl_braced_path(mut_pdb)}",
        f"set ligand_sel {{resname {ligand_resname.upper()}}}",
        f"set hotspot_sel {{{hotspot_sel}}}",
        f"set source_sel {{{source_sel}}}",
        f"set target_sel {{{target_sel}}}",
        "mol new $mut_pdb type pdb waitfor all",
        "set mid [molinfo top]",
        "mol delrep 0 $mid",
        "mol representation NewCartoon 0.22 24.0 4.2 0",
        "mol color ColorID 8",
        "mol selection {protein}",
        "mol material AOChalky",
        "mol addrep $mid",
        "mol representation Licorice 0.24 20.0 20.0",
        "mol color Name",
        "mol selection $ligand_sel",
        "mol material Glossy",
        "mol addrep $mid",
        "mol representation VDW 0.55 24.0",
        "mol color ColorID 1",
        "mol selection ($hotspot_sel) and name CA",
        "mol material Glossy",
        "mol addrep $mid",
        "mol representation VDW 0.70 24.0",
        "mol color ColorID 4",
        "mol selection ($source_sel) and name CA",
        "mol material Glossy",
        "mol addrep $mid",
        "mol representation QuickSurf 1.0 0.5 1.0 1.0",
        "mol color ColorID 10",
        "mol selection $target_sel",
        "mol material Transparent",
        "mol addrep $mid",
        "graphics $mid color black",
    ]
    for row in top_rows[: min(20, len(top_rows))]:
        coord_ang = 10.0 * np.asarray(row["mut_coord_nm"], dtype=float)
        label = f"{row['rank']}. {row['node_label']} {float(row['fast_allostery_score']):.2f}"
        label = label.replace("{", "").replace("}", "")
        lines.append(
            f"graphics $mid text {{{coord_ang[0]:.3f} {coord_ang[1]:.3f} {coord_ang[2]:.3f}}} {{{label}}} size 0.85 thickness 2"
        )
    lines += [
        "display resetview",
        "puts \"Loaded fast allostery scan view.\"",
        "puts \"Red: top fast-ranked residues. Yellow: mutation source. Cyan: target/binding site.\"",
    ]
    os.makedirs(os.path.dirname(out_tcl) or ".", exist_ok=True)
    with open(out_tcl, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return out_tcl


def write_consensus_vmd(
    scenario: str,
    representative_run: Dict[str, str],
    consensus_rows: Sequence[Dict[str, Any]],
    ligand_resname: str,
    out_tcl: str,
    top_n: int,
) -> Optional[str]:
    """Write the scene of the consensus ranking of one scenario."""
    if not representative_run:
        return None
    run_dir = representative_run["run_dir"]
    summary_path = os.path.join(run_dir, "run_summary.json")
    if not os.path.exists(summary_path):
        return None
    summary = load_json(summary_path)
    mut_pdb = resolve_run_path(run_dir, summary["md"]["mut"]["prepared_pdb"])
    mut_dcd_raw = summary["md"]["mut"].get("traj_dcd")
    mut_dcd = resolve_run_path(run_dir, mut_dcd_raw) if mut_dcd_raw else None
    rows = [row for row in consensus_rows if str(row["scenario"]) == scenario][: max(1, int(top_n))]
    hotspot_sel = selection_from_consensus_rows(rows)
    label_lines = []
    for row in rows[: min(24, len(rows))]:
        resid = str(row.get("resid", "")).strip()
        chain = str(row.get("chain", "")).strip()
        label = f"{row['consensus_rank']}. {row['node_label']} {float(row['consensus_score']):.3f}"
        label = label.replace("{", "").replace("}", "")
        selection = f"chain {chain} and resid {resid} and name CA" if chain else f"resid {resid} and name CA"
        label_lines += [
            f"set sel [atomselect $mid {{{selection}}}]",
            "if {[$sel num] > 0} {",
            "    set c [measure center $sel]",
            f"    graphics $mid text $c {{{label}}} size 0.85 thickness 2",
            "}",
            "$sel delete",
        ]

    lines = [
        "# Auto-generated consensus allostery hotspot view for VMD",
        f"# Scenario: {scenario}",
        "display projection Orthographic",
        "display depthcue off",
        "axes location Off",
        "color Display Background white",
        f"set mut_pdb {tcl_braced_path(mut_pdb)}",
        f"set mut_dcd {tcl_braced_path(mut_dcd)}",
        f"set ligand_sel {{resname {ligand_resname.upper()}}}",
        f"set hotspot_sel {{{hotspot_sel}}}",
        "mol new $mut_pdb type pdb waitfor all",
        "set mid [molinfo top]",
        "if {[string length $mut_dcd] > 0 && [file exists $mut_dcd]} {",
        "    mol addfile $mut_dcd type dcd waitfor all molid $mid",
        "}",
        "mol delrep 0 $mid",
        "mol representation NewCartoon 0.22 24.0 4.2 0",
        "mol color ColorID 8",
        "mol selection {protein}",
        "mol material AOChalky",
        "mol addrep $mid",
        "mol representation Licorice 0.24 20.0 20.0",
        "mol color Name",
        "mol selection $ligand_sel",
        "mol material Glossy",
        "mol addrep $mid",
        "",
        "mol representation VDW 0.55 24.0",
        "mol color ColorID 1",
        "mol selection ($hotspot_sel) and name CA",
        "mol material Glossy",
        "mol addrep $mid",
        "graphics $mid color black",
        *label_lines,
        "display resetview",
        f"puts \"Loaded consensus allostery hotspots for {scenario}.\"",
    ]
    os.makedirs(os.path.dirname(out_tcl) or ".", exist_ok=True)
    with open(out_tcl, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return out_tcl
