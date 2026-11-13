"""``pert fast-scan`` - score one WT/mutant structure pair without MD.
"""

from __future__ import annotations

import argparse
import json
import os

from pert.allostery.fast_scan import infer_structures_from_run_dir, run_fast_scan
from pert.utils.io import parse_csv_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast topology-flexible mutation/allostery scan without MD.")
    parser.add_argument("--run-dir", default=None, help="Completed run directory; infers WT/MUT structures.")
    parser.add_argument("--wt-pdb", default=None)
    parser.add_argument("--mut-pdb", default=None)
    parser.add_argument("--mutation-label", default=None)
    parser.add_argument(
        "--mutation-spec",
        action="append",
        default=None,
        help="Reference-coordinate mutation spec, repeat/comma separated. Enables exact source mapping for compound variants.",
    )
    parser.add_argument("--ligand-resname", default="ZN")
    parser.add_argument("--cutoff-ang", type=float, default=6.0)
    parser.add_argument("--contact-threshold-ang", type=float, default=8.0)
    parser.add_argument("--source-hotspots", type=int, default=8)
    parser.add_argument("--top-residues", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.run_dir:
        wt_pdb, mut_pdb, inferred_label = infer_structures_from_run_dir(args.run_dir)
        mutation_label = args.mutation_label or inferred_label
        output_dir = args.output_dir or os.path.join(os.path.abspath(args.run_dir), "fast_scan")
    else:
        if not args.wt_pdb or not args.mut_pdb:
            raise SystemExit("Provide --run-dir or both --wt-pdb and --mut-pdb.")
        wt_pdb = args.wt_pdb
        mut_pdb = args.mut_pdb
        mutation_label = args.mutation_label or "mutation"
        output_dir = args.output_dir or "runs/fast_allostery_scan"

    summary = run_fast_scan(
        wt_pdb=wt_pdb,
        mut_pdb=mut_pdb,
        ligand_resname=args.ligand_resname,
        output_dir=os.path.abspath(output_dir),
        mutation_label=mutation_label,
        mutation_specs=parse_csv_tokens(args.mutation_spec),
        cutoff_ang=args.cutoff_ang,
        contact_threshold_ang=args.contact_threshold_ang,
        source_hotspots=args.source_hotspots,
        top_residues=args.top_residues,
    )
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
