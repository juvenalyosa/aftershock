"""``pert null-control`` - calibrate a panel against the WT-vs-WT null.
"""

from __future__ import annotations

import argparse
import json

from pert.config import TEConfig
from pert.pipelines.null_control import NullControlUnavailable, recalibrate_null_control, run_null_control
from pert.utils.io import parse_csv_tokens


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser of ``pert null-control``."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare every WT-vs-MUT result of a panel with the distribution obtained from "
            "pairs of WT simulations (no mutation), using the WT trajectories the panel already has."
        )
    )
    parser.add_argument("--panel-dir", required=True, help="Panel directory containing <scenario>/<replicate>/run_summary.json.")
    parser.add_argument(
        "--ligand-resname",
        action="append",
        default=None,
        help="Ligand residue name(s); repeat or comma-separate. Default: as recorded by the panel.",
    )
    parser.add_argument("--max-pairs", type=int, default=30, help="Maximum number of WT-vs-WT pairs (<=0: all).")
    parser.add_argument("--output-dir", default=None, help="Default: <panel-dir>/null_control.")
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Re-analyse every run instead of reusing stored results computed with the same settings.",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help=(
            "Only re-derive the verdicts from the WT-vs-WT pairs already computed (no trajectory is read). "
            "For results produced under older rules, or copied without their trajectories."
        ),
    )
    parser.add_argument("--var-max-order", type=int, default=None, help="TE setting; default: as recorded by the panel.")
    parser.add_argument("--te-surrogates", type=int, default=None, help="TE setting; default: as recorded by the panel.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for the choice of pairs.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    te_cfg = None
    if args.var_max_order is not None or args.te_surrogates is not None:
        if args.var_max_order is None or args.te_surrogates is None:
            raise SystemExit("Give both --var-max-order and --te-surrogates, or neither.")
        te_cfg = TEConfig(max_order=args.var_max_order, surrogates=args.te_surrogates)
    try:
        if args.recalibrate:
            summary = recalibrate_null_control(args.panel_dir, output_dir=args.output_dir)
            print(json.dumps({k: summary[k] for k in ("wt_trajectories", "null_pairs", "scenarios", "notes", "outputs")}, indent=2))
            return
        summary = run_null_control(
            panel_dir=args.panel_dir,
            ligand_resnames=parse_csv_tokens(args.ligand_resname) or None,
            te_cfg=te_cfg,
            output_dir=args.output_dir,
            max_pairs=args.max_pairs,
            reuse_run_analysis=not args.no_reuse,
            random_seed=args.seed,
        )
    except NullControlUnavailable as exc:
        raise SystemExit(f"Null control unavailable: {exc}")
    print(json.dumps({k: summary[k] for k in ("wt_trajectories", "null_pairs", "scenarios", "notes", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
