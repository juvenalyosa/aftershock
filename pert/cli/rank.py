"""``pert rank`` - MD-based allosteric residue/region ranking for a finished run.
"""

from __future__ import annotations

import argparse
import json

from pert.allostery.ranking import run_allostery_ranking


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank allosteric residue effects for a completed run.")
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory containing run_summary.json.")
    parser.add_argument("--ligand-resname", default="ZN")
    parser.add_argument("--cutoff-ang", type=float, default=6.0, help="Binding-site cutoff around ligand.")
    parser.add_argument("--source-hotspots", type=int, default=10, help="Number of mutation-proximal shifted residues to use as sources.")
    parser.add_argument("--top-edges", type=int, default=80, help="Number of positive TE-delta edges to store.")
    parser.add_argument("--top-paths", type=int, default=30, help="Number of mutation-to-site paths to store.")
    parser.add_argument("--top-residues", type=int, default=30, help="Number of residues to show in plot/VMD.")
    parser.add_argument("--region-window", type=int, default=5, help="Contiguous residue window size for region ranking.")
    parser.add_argument("--max-frames", type=int, default=2000, help="Evenly sample at most this many frames; <=0 uses all frames.")
    parser.add_argument("--contact-threshold-ang", type=float, default=8.0, help="C-alpha contact threshold.")
    parser.add_argument("--output-dir", default=None, help="Default: <run-dir>/analysis.")
    args = parser.parse_args()

    manifest = run_allostery_ranking(
        run_dir=args.run_dir,
        ligand_resname=args.ligand_resname,
        cutoff_ang=args.cutoff_ang,
        source_hotspots=args.source_hotspots,
        top_edges_n=args.top_edges,
        top_paths_n=args.top_paths,
        top_residues=args.top_residues,
        region_window=args.region_window,
        max_frames=None if args.max_frames <= 0 else args.max_frames,
        contact_threshold_ang=args.contact_threshold_ang,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
