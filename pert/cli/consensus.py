"""``pert consensus`` - aggregate per-run rankings across replicas of a panel.
"""

from __future__ import annotations

import argparse
import json

from pert.allostery.consensus import run_consensus


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate allostery rankings across replicas/scenarios.")
    parser.add_argument("--panel-dir", default=".", help="Panel directory containing scenario/rep*/run_summary.json.")
    parser.add_argument("--run-dir", action="append", default=None, help="Specific run dir; repeat to bypass panel scan.")
    parser.add_argument("--rank-missing", action="store_true", help="Run the `pert rank` logic for runs missing residue scores.")
    parser.add_argument("--ligand-resname", default="ZN")
    parser.add_argument("--cutoff-ang", type=float, default=6.0)
    parser.add_argument("--source-hotspots", type=int, default=10)
    parser.add_argument("--top-edges", type=int, default=80)
    parser.add_argument("--top-paths", type=int, default=30)
    parser.add_argument("--top-residues", type=int, default=30)
    parser.add_argument("--top-regions", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--contact-threshold-ang", type=float, default=8.0)
    parser.add_argument("--output-dir", default=None, help="Default: panel dir.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--control-scenario", default=None, help="Scenario to use as a control/baseline for mutation-specific calls.")
    parser.add_argument("--baseline-margin", type=float, default=0.05, help="Minimum consensus-score excess over control.")
    args = parser.parse_args()

    manifest = run_consensus(
        panel_dir=args.panel_dir,
        run_dirs=args.run_dir,
        rank_missing=args.rank_missing,
        ligand_resname=args.ligand_resname,
        cutoff_ang=args.cutoff_ang,
        source_hotspots=args.source_hotspots,
        top_edges=args.top_edges,
        top_paths=args.top_paths,
        top_residues=args.top_residues,
        top_regions=args.top_regions,
        max_frames=None if args.max_frames <= 0 else args.max_frames,
        contact_threshold_ang=args.contact_threshold_ang,
        output_dir=args.output_dir,
        bootstrap=args.bootstrap,
        seed=args.seed,
        control_scenario=args.control_scenario,
        baseline_margin=args.baseline_margin,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
