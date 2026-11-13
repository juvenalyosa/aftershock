"""``pert analyze-run`` - standard plots, binding-site table, residue-flow
network and VMD scenes for a finished run.
"""

from __future__ import annotations

import argparse
import json

from pert.pipelines.post_analysis import run_standard_analysis


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser of ``pert analyze-run``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory (contains run_summary.json).")
    parser.add_argument("--ligand-resname", default="ZN", help="Ligand residue name to define binding-site shell.")
    parser.add_argument("--cutoff-ang", type=float, default=6.0, help="Binding-site cutoff around ligand.")
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = run_standard_analysis(
        args.run_dir,
        ligand_resname=args.ligand_resname,
        cutoff_ang=args.cutoff_ang,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
