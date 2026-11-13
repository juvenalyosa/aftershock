"""``pert md-panel`` - run the MD mutation panel (scenarios x replicas).
"""

from __future__ import annotations

import argparse
import json

from pert.pipelines.md_panel import run_md_panel


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser of ``pert md-panel``."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a panel of mutation scenarios x replicas through build, MD and analysis, then calibrate it "
            "against the WT-vs-WT null. Works for any protein: give --reference-pdb, --template-pdb, --template, "
            "--ligand-xml, --ligand-resname and the scenarios (--mutation-set / --mutation-file). "
            "The defaults run the toy zinc-finger panel."
        )
    )
    parser.add_argument("--reference-pdb", default="data/toy/1ZNF.pdb")
    parser.add_argument("--template-pdb", default="data/toy/1ZNF.pdb")
    parser.add_argument("--ligand-xml", default="data/toy/empty_ligand.xml")
    parser.add_argument("--ligand-resname", default="ZN")
    parser.add_argument("--template", default="1ZNFA")
    parser.add_argument("--chain-id", default="A")
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--output-dir", default="runs/mutation_panel_1znf")
    parser.add_argument("--num-models", type=int, default=4)
    parser.add_argument("--num-processors", type=int, default=1)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--equil-steps", type=int, default=2000)
    parser.add_argument("--prod-steps", type=int, default=20000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--no-solvent", action="store_true", help="Disable solvent and ions for dry/vacuum tests.")
    parser.add_argument(
        "--nonbonded-method",
        default="PME",
        help="PME/Ewald/CutoffPeriodic/CutoffNonPeriodic/NoCutoff.",
    )
    parser.add_argument("--nonbonded-cutoff-nm", type=float, default=1.0)
    parser.add_argument("--platform", default="CPU")
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=250000,
        help="Write a restart point every N production steps (0: only at the end).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch, overwriting existing structures and simulations (default: reuse/continue them).",
    )
    parser.add_argument("--md-seed-base", type=int, default=1300)
    parser.add_argument("--rb-ang", type=float, default=6.0)
    parser.add_argument("--rp-ang", type=float, default=12.0)
    parser.add_argument("--grid-spacing-ang", type=float, default=3.0)
    parser.add_argument("--sigma-ang", type=float, default=3.0)
    parser.add_argument("--max-grid-points", type=int, default=64)
    parser.add_argument("--occupancy-only", action="store_true")
    parser.add_argument(
        "--frame-mode",
        default="shared_site",
        choices=["shared_site", "legacy_self"],
        help="Coordinate frame of the comparison: shared_site (default, WT and mutant superposed on the same site scaffold) or legacy_self (original behaviour, not rotation-invariant).",
    )
    parser.add_argument(
        "--no-null-control",
        action="store_true",
        help="Skip the WT-vs-WT null control that calibrates the panel at the end.",
    )
    parser.add_argument(
        "--max-null-pairs",
        type=int,
        default=30,
        help="Maximum number of WT-vs-WT pairs analysed by the null control (<=0: all).",
    )
    parser.add_argument("--vamp-lag", type=int, default=4)
    parser.add_argument("--latent-b", type=int, default=6)
    parser.add_argument("--latent-p", type=int, default=6)
    parser.add_argument("--latent-l", type=int, default=3)
    parser.add_argument(
        "--vamp-solver",
        default="auto",
        choices=["auto", "full"],
        help="auto (default): exact reduced-dimension solver; full: original full-dimensional computation.",
    )
    parser.add_argument("--var-max-order", type=int, default=3)
    parser.add_argument("--te-surrogates", type=int, default=64)
    parser.add_argument("--te-seed-base", type=int, default=200)
    parser.add_argument(
        "--mutation-set",
        action="append",
        default=None,
        help=(
            "Scenario as NAME:MUT1,MUT2 in reference numbering; repeat for several. "
            "Examples: ex19del:L747_A750del or double:K2R,Q20dup. "
            "A silent substitution (e.g. wt_model:K2K) gives a modelled-WT control."
        ),
    )
    parser.add_argument(
        "--mutation-file",
        action="append",
        default=None,
        help=(
            "JSON or CSV file with scenarios. JSON: {\"name\": [\"K2R\"]}, a list of {name, mutations} objects, "
            "or either under a \"mutation_sets\" key. CSV: columns name,mutations."
        ),
    )
    parser.add_argument(
        "--include-builtins",
        action="store_true",
        help="Also run the built-in 1ZNF toy scenarios next to the custom ones.",
    )
    parser.add_argument(
        "--scenarios",
        action="append",
        default=None,
        help="Run only these scenario names (repeat and/or comma-separated); applied after the scenarios are loaded.",
    )
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(run_md_panel(args), indent=2))


if __name__ == "__main__":
    main()
