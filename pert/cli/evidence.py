"""``pert evidence`` - write the panel-level evidence report.
"""

from __future__ import annotations

import argparse
import json

from pert.allostery.evidence import write_evidence_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a panel-level allostery evidence report.")
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--top-residues", type=int, default=10)
    parser.add_argument("--top-regions", type=int, default=5)
    parser.add_argument("--specificity-top-n", type=int, default=10)
    parser.add_argument("--control-scenario", default=None, help="Override/inject control scenario name for report interpretation.")
    parser.add_argument("--baseline-specific-top", type=int, default=10)
    parser.add_argument("--min-replicates-strong", type=int, default=3)
    parser.add_argument("--min-replicates-moderate", type=int, default=2)
    parser.add_argument("--min-support-strong", type=float, default=0.67)
    parser.add_argument("--min-support-moderate", type=float, default=0.50)
    parser.add_argument("--min-conservative-strong", type=float, default=0.35)
    parser.add_argument("--min-conservative-moderate", type=float, default=0.25)
    args = parser.parse_args()

    payload = write_evidence_report(
        panel_dir=args.panel_dir,
        output_md=args.out_md,
        output_json=args.out_json,
        top_residues=args.top_residues,
        top_regions=args.top_regions,
        specificity_top_n=args.specificity_top_n,
        control_scenario=args.control_scenario,
        baseline_specific_top=args.baseline_specific_top,
        min_replicates_strong=args.min_replicates_strong,
        min_replicates_moderate=args.min_replicates_moderate,
        min_support_strong=args.min_support_strong,
        min_support_moderate=args.min_support_moderate,
        min_conservative_strong=args.min_conservative_strong,
        min_conservative_moderate=args.min_conservative_moderate,
    )
    print(json.dumps(payload["outputs"], indent=2))


if __name__ == "__main__":
    main()
