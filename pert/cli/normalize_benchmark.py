"""``pert normalize-benchmark`` - convert an external table to the benchmark format.
"""

from __future__ import annotations

import argparse
import json

from pert.benchmark.normalize import normalize_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize mutation datasets for `pert benchmark`.")
    parser.add_argument("--input", required=True, help="Input CSV/TSV table.")
    parser.add_argument("--output", required=True, help="Canonical benchmark CSV to write.")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--name-column", default=None)
    parser.add_argument("--mutations-column", default=None)
    parser.add_argument(
        "--mutation-separators",
        default=",;",
        help="Characters that separate multiple mutation specs. Use ',;:' for ProteinGym substitution rows.",
    )
    parser.add_argument("--mutated-sequence-column", default=None)
    parser.add_argument("--reference-sequence-column", default=None)
    parser.add_argument("--reference-sequence", default="")
    parser.add_argument("--sequence-start", type=int, default=1)
    parser.add_argument("--wt-pdb-column", default=None)
    parser.add_argument("--mut-pdb-column", default=None)
    parser.add_argument("--run-dir-column", default=None)
    parser.add_argument("--ligand-resname-column", default=None)
    parser.add_argument("--default-ligand-resname", default="")
    parser.add_argument("--label-source-column", default=None)
    parser.add_argument("--effect-source-column", default=None)
    parser.add_argument("--effect-mode", default="raw", choices=["raw", "abs", "neg"])
    parser.add_argument("--derive-label-from-effect", action="store_true")
    parser.add_argument("--label-threshold", type=float, default=1.0)
    parser.add_argument("--label-direction", default="abs-ge", choices=["ge", "le", "abs-ge"])
    parser.add_argument("--wt-pdb-pattern", default=None, help="Format pattern, e.g. structures/{name}/wt.pdb")
    parser.add_argument("--mut-pdb-pattern", default=None, help="Format pattern, e.g. structures/{name}/mut.pdb")
    parser.add_argument("--run-dir-pattern", default=None, help="Format pattern, e.g. runs/{name}")
    parser.add_argument("--rename-column", action="append", default=None, help="Copy SOURCE to DEST, repeatable.")
    parser.add_argument("--keep-column", action="append", default=None, help="Pass through an input column.")
    parser.add_argument("--drop-missing-structure", action="store_true")
    parser.add_argument("--allow-invalid-mutations", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(normalize_rows(args), indent=2))


if __name__ == "__main__":
    main()
