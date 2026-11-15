"""``pert site-check`` - does the metal site hold together in every simulation?
"""

from __future__ import annotations

import argparse
import json

from pert.pipelines.site_check import NoMetalSite, run_site_check
from pert.utils.io import parse_csv_tokens


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser of ``pert site-check``."""
    parser = argparse.ArgumentParser(
        description=(
            "Follow the coordination of the metal ion(s) through every WT and mutant trajectory of a panel "
            "(or of a single run): which ligating atoms stay, which leave, and how much water gets in."
        )
    )
    parser.add_argument("--panel-dir", required=True, help="Panel directory, or one run directory.")
    parser.add_argument(
        "--ligand-resname",
        action="append",
        default=None,
        help="Residue name(s) of the metal; repeat or comma-separate. Default: every metal ion.",
    )
    parser.add_argument("--shell-cutoff-ang", type=float, default=2.6, help="Initial coordination shell (prepared structure).")
    parser.add_argument("--bound-cutoff-ang", type=float, default=3.0, help="A donor within this distance is bound.")
    parser.add_argument("--output-dir", default=None, help="Default: <panel-dir>/site_check.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    names = parse_csv_tokens(args.ligand_resname) if args.ligand_resname else None
    try:
        summary = run_site_check(
            args.panel_dir,
            ligand_resnames=names or None,
            shell_cutoff_ang=args.shell_cutoff_ang,
            bound_cutoff_ang=args.bound_cutoff_ang,
            output_dir=args.output_dir,
        )
    except NoMetalSite as exc:
        print(f"Nothing to check: {exc}")   # not an error: most ligands are not metal ions
        return
    print(json.dumps({k: summary[k] for k in ("simulations", "status_counts", "status_counts_by_system", "all_intact", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
