"""``pert diagnostics`` - extended diagnostic plots and interpretation summary
for a finished run.
"""

from __future__ import annotations

import argparse
import json

from pert.pipelines.post_analysis import run_diagnostic_analysis


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser of ``pert diagnostics``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(run_diagnostic_analysis(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
