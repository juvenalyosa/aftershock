"""``pert fast-panel`` - build and fast-score a panel of mutation sets.
"""

from __future__ import annotations

import argparse
import json

from pert.pipelines.fast_panel import run_fast_panel
from pert.pipelines.scenarios import collect_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast no-MD mutation/allostery screening panel.")
    parser.add_argument("--reference-pdb", default="data/toy/1ZNF.pdb")
    parser.add_argument("--template-pdb", default="data/toy/1ZNF.pdb")
    parser.add_argument("--ligand-xml", default="data/toy/empty_ligand.xml")
    parser.add_argument("--ligand-resname", default="ZN")
    parser.add_argument("--template", default="1ZNFA")
    parser.add_argument("--chain-id", default="A")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--output-dir", default="runs/fast_panel_1znf")
    parser.add_argument("--num-models", type=int, default=4)
    parser.add_argument("--num-processors", type=int, default=1)
    parser.add_argument("--cutoff-ang", type=float, default=6.0)
    parser.add_argument("--contact-threshold-ang", type=float, default=8.0)
    parser.add_argument("--source-hotspots", type=int, default=8)
    parser.add_argument("--top-residues", type=int, default=30)
    parser.add_argument(
        "--mutation-set",
        action="append",
        default=None,
        help=(
            "Custom named variant set as NAME:MUT1,MUT2. Repeat for multiple sets. "
            "Examples: snps:K2R,Q20R or complex:10_11delinsG,Q20dup."
        ),
    )
    parser.add_argument(
        "--mutation-file",
        action="append",
        default=None,
        help=(
            "JSON or CSV mutation-set file. JSON may be {'name':['K2R']} mapping, "
            "a list of {'name','mutations'} objects, or {'mutation_sets': ...}. "
            "CSV requires columns: name,mutations."
        ),
    )
    parser.add_argument(
        "--include-builtins",
        action="store_true",
        help="Run built-in examples in addition to custom --mutation-set/--mutation-file entries.",
    )
    parser.add_argument(
        "--scenarios",
        action="append",
        default=None,
        help="Scenario name filter, repeat/comma separated. Applies after built-in/custom sets are loaded.",
    )
    args = parser.parse_args()
    # Scenario filtering happens here, so unknown names are reported instead of ignored.
    scenario_defs = collect_scenarios(
        mutation_sets=args.mutation_set,
        mutation_files=args.mutation_file,
        include_builtins=bool(args.include_builtins),
        selected=args.scenarios,
    )
    manifest = run_fast_panel(
        reference_pdb=args.reference_pdb,
        template_pdb=args.template_pdb,
        ligand_xml=args.ligand_xml,
        ligand_resname=args.ligand_resname,
        template=args.template,
        chain_id=args.chain_id,
        replicas=args.replicas,
        output_dir=args.output_dir,
        num_models=args.num_models,
        num_processors=args.num_processors,
        selected_scenarios=[],
        cutoff_ang=args.cutoff_ang,
        contact_threshold_ang=args.contact_threshold_ang,
        source_hotspots=args.source_hotspots,
        top_residues=args.top_residues,
        scenario_defs=scenario_defs,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
