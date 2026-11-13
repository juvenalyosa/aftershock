"""``pert benchmark`` - benchmark the fast scanner on a table of variants.
"""

from __future__ import annotations

import argparse
import json

from pert.benchmark.fast_benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark/calibrate fast allostery mutation scores.")
    parser.add_argument("--input-csv", required=True, help="Benchmark CSV with name and mutations columns.")
    parser.add_argument("--output-dir", default="runs/fast_allostery_benchmark")
    parser.add_argument("--reference-pdb", default=None, help="Reference PDB used when rows do not provide structures.")
    parser.add_argument("--template-pdb", default=None, help="Template PDB copied for MODELLER when building structures.")
    parser.add_argument("--ligand-xml", default="data/toy/empty_ligand.xml")
    parser.add_argument("--ligand-resname", default="ZN")
    parser.add_argument("--template", default="1ZNFA")
    parser.add_argument("--chain-id", default="A")
    parser.add_argument("--num-models", type=int, default=1)
    parser.add_argument("--num-processors", type=int, default=1)
    parser.add_argument("--modeller-seed-base", type=int, default=7000)
    parser.add_argument("--cutoff-ang", type=float, default=6.0)
    parser.add_argument("--contact-threshold-ang", type=float, default=8.0)
    parser.add_argument("--source-hotspots", type=int, default=8)
    parser.add_argument("--top-residues", type=int, default=10)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--effect-column", default="effect_score")
    parser.add_argument(
        "--external-score-column",
        action="append",
        default=None,
        help="Optional comparator score column, for example foldx_ddg or ddmut_score. Repeat as needed.",
    )
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args), indent=2))


if __name__ == "__main__":
    main()
