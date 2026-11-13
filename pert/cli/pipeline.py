"""``pert pipeline`` - run one WT-vs-MUT comparison (build -> MD -> analysis).
"""

from __future__ import annotations

import argparse
import json

from pert.config import BuildConfig, FieldConfig, LatentConfig, MDConfig, TEConfig
from pert.pipelines.single_run import run_full_pipeline
from pert.utils.io import parse_csv_tokens


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser of ``pert pipeline`` (one flag per configuration field)."""
    parser = argparse.ArgumentParser(
        description="Topology-safe WT vs MUT MD comparator (INS/DEL/INSDEL/DUP/SNP aware)."
    )
    parser.add_argument("--reference-pdb", required=True, help="WT reference PDB containing ligand.")
    parser.add_argument("--ligand-xml", required=True, help="OpenMM XML with ligand parameters.")
    parser.add_argument(
        "--ligand-resname",
        action="append",
        required=True,
        help="Ligand residue name(s). Repeat or use comma-separated values.",
    )
    parser.add_argument(
        "--mutations",
        action="append",
        required=True,
        help="Mutation list (repeat and/or comma-separated). Supports SNP/INS/DEL/INSDEL/DUP.",
    )
    parser.add_argument(
        "--templates",
        action="append",
        required=True,
        help="MODELLER template codes, e.g. 2XYZA (repeat and/or comma-separated).",
    )
    parser.add_argument("--chain-id", default="A", help="Protein chain used for sequence extraction/alignment.")
    parser.add_argument(
        "--original-sequence",
        default=None,
        help="Original protein sequence used by the mutation and modelling stages. If omitted, derived from reference PDB chain.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=None,
        help="Sequence numbering start position for mutation indices; inferred from reference PDB if omitted.",
    )
    parser.add_argument("--num-models", type=int, default=50, help="Number of MODELLER models to build.")
    parser.add_argument("--num-processors", type=int, default=4, help="Parallel workers for MODELLER.")
    parser.add_argument("--modeller-seed", type=int, default=None, help="Deterministic MODELLER seed.")

    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--friction-per-ps", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--barostat-interval", type=int, default=25)
    parser.add_argument(
        "--nonbonded-method",
        default="PME",
        help="Nonbonded method: PME/Ewald/CutoffPeriodic/CutoffNonPeriodic/NoCutoff.",
    )
    parser.add_argument("--nonbonded-cutoff-nm", type=float, default=1.0)
    parser.add_argument("--no-solvent", action="store_true", help="Disable water box + ion addition.")
    parser.add_argument("--water-model", default="tip3p", help="Water model for addSolvent (tip3p/spce/tip4pew/tip5p).")
    parser.add_argument("--solvent-padding-nm", type=float, default=1.0)
    parser.add_argument("--ionic-strength-m", type=float, default=0.15)
    parser.add_argument("--no-neutralize", action="store_true", help="Disable system neutralization when adding solvent.")
    parser.add_argument("--positive-ion", default="Na+")
    parser.add_argument("--negative-ion", default="Cl-")
    parser.add_argument("--equil-steps", type=int, default=5000)
    parser.add_argument("--heating-initial-k", type=float, default=50.0)
    parser.add_argument("--heating-steps", type=int, default=5000)
    parser.add_argument("--heating-stages", type=int, default=10)
    parser.add_argument("--prod-steps", type=int, default=50000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--md-seed", type=int, default=None, help="Random seed for WT MD; MUT uses seed+1.")
    parser.add_argument("--platform", default="CPU", help="OpenMM platform: AUTO/CPU/CUDA/OpenCL/Reference.")
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

    parser.add_argument("--rb-ang", type=float, default=6.0, help="Pocket radius R_B in Angstrom.")
    parser.add_argument("--rp-ang", type=float, default=12.0, help="Context shell radius R_P in Angstrom.")
    parser.add_argument("--grid-spacing-ang", type=float, default=1.2)
    parser.add_argument("--sigma-ang", type=float, default=1.2)
    parser.add_argument("--max-grid-points", type=int, default=512)
    parser.add_argument("--occupancy-only", action="store_true", help="Disable velocity channels.")
    parser.add_argument("--keep-com", action="store_true", help="Disable COM removal.")
    parser.add_argument(
        "--frame-mode",
        default="shared_site",
        choices=["shared_site", "legacy_self"],
        help="Coordinate frame of the comparison: shared_site (default, WT and mutant superposed on the same site scaffold) or legacy_self (original behaviour, not rotation-invariant).",
    )
    parser.add_argument(
        "--no-density-normalize",
        action="store_true",
        help="Disable per-region atom-count normalization in field channels.",
    )
    parser.add_argument(
        "--no-append-type-counts",
        action="store_true",
        help="Disable appended per-type count features.",
    )

    parser.add_argument("--vamp-lag", type=int, default=5)
    parser.add_argument("--latent-b", type=int, default=10)
    parser.add_argument("--latent-p", type=int, default=10)
    parser.add_argument("--latent-l", type=int, default=6)
    parser.add_argument("--vamp-reg", type=float, default=1e-6)
    parser.add_argument(
        "--vamp-solver",
        default="auto",
        choices=["auto", "full"],
        help="auto (default): exact reduced-dimension solver; full: original full-dimensional computation.",
    )

    parser.add_argument("--var-max-order", type=int, default=6)
    parser.add_argument("--te-surrogates", type=int, default=64)
    parser.add_argument("--te-seed", type=int, default=13)

    parser.add_argument("--output-dir", default="runs/topology_safe_run", help="Output directory.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ligand_resnames = parse_csv_tokens(args.ligand_resname)
    mutations = parse_csv_tokens(args.mutations)
    templates = parse_csv_tokens(args.templates)

    build_cfg = BuildConfig(
        reference_pdb=args.reference_pdb,
        ligand_xml=args.ligand_xml,
        ligand_resnames=ligand_resnames,
        mutations=mutations,
        templates=templates,
        chain_id=args.chain_id,
        original_sequence=args.original_sequence,
        start_pos=args.start_pos,
        num_models=args.num_models,
        num_processors=args.num_processors,
        modeller_seed=args.modeller_seed,
    )
    md_cfg = MDConfig(
        temperature_k=args.temperature_k,
        friction_per_ps=args.friction_per_ps,
        timestep_fs=args.timestep_fs,
        pressure_bar=args.pressure_bar,
        barostat_interval=args.barostat_interval,
        nonbonded_method=args.nonbonded_method,
        nonbonded_cutoff_nm=args.nonbonded_cutoff_nm,
        add_solvent=not args.no_solvent,
        water_model=args.water_model,
        solvent_padding_nm=args.solvent_padding_nm,
        ionic_strength_m=args.ionic_strength_m,
        neutralize=not args.no_neutralize,
        positive_ion=args.positive_ion,
        negative_ion=args.negative_ion,
        heating_initial_temperature_k=args.heating_initial_k,
        heating_steps=args.heating_steps,
        heating_stages=args.heating_stages,
        equilibration_steps=args.equil_steps,
        production_steps=args.prod_steps,
        report_interval=args.report_interval,
        platform=args.platform,
        random_seed=args.md_seed,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
    )
    field_cfg = FieldConfig(
        rb_ang=args.rb_ang,
        rp_ang=args.rp_ang,
        grid_spacing_ang=args.grid_spacing_ang,
        sigma_ang=args.sigma_ang,
        max_grid_points=args.max_grid_points,
        include_velocity=not args.occupancy_only,
        remove_com=not args.keep_com,
        density_normalize=not args.no_density_normalize,
        append_type_counts=not args.no_append_type_counts,
        frame_mode=args.frame_mode,
    )
    latent_cfg = LatentConfig(
        lag=args.vamp_lag,
        m_b=args.latent_b,
        m_p=args.latent_p,
        m_l=args.latent_l,
        reg=args.vamp_reg,
        solver=args.vamp_solver,
    )
    te_cfg = TEConfig(
        max_order=args.var_max_order,
        surrogates=args.te_surrogates,
        random_seed=args.te_seed,
    )

    summary = run_full_pipeline(
        build_cfg=build_cfg,
        md_cfg=md_cfg,
        field_cfg=field_cfg,
        latent_cfg=latent_cfg,
        te_cfg=te_cfg,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
