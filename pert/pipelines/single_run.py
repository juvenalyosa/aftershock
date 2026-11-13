"""End-to-end pipeline for one mutation set: build -> MD -> analysis.

``run_full_pipeline`` is the unit of work of the package.  It writes a
self-describing run directory whose ``run_summary.json`` is the entry point of
every post-processing command.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from typing import Dict

from pert.analysis.topology_safe import run_topology_safe_analysis
from pert.config import BuildConfig, FieldConfig, LatentConfig, MDConfig, TEConfig
from pert.simulation.md import run_openmm_md
from pert.structure.prepare import prepare_structures
from pert.utils.provenance import collect_provenance
from pert.visualization.vmd_comparison import write_vmd_comparison_script


def run_full_pipeline(
    build_cfg: BuildConfig,
    md_cfg: MDConfig,
    field_cfg: FieldConfig,
    latent_cfg: LatentConfig,
    te_cfg: TEConfig,
    output_dir: str,
    resume: bool = True,
) -> Dict[str, object]:
    """Run build -> MD (WT and mutant) -> topology-safe analysis for one mutation set.

    Writes ``structures/``, ``md/``, ``analysis/`` and ``run_summary.json`` under
    ``output_dir`` and returns the summary dict.

    The call is restartable (``resume=True``): structures that are already
    built and simulations that are already complete are reused, an interrupted
    simulation continues from its last checkpoint, and the (cheap) analysis is
    always recomputed.  Files produced with different settings are never
    overwritten silently; ``resume=False`` starts everything from scratch.
    """
    os.makedirs(output_dir, exist_ok=True)
    prep = prepare_structures(build_cfg, output_dir=output_dir, resume=resume)
    md_dir = os.path.join(output_dir, "md")
    os.makedirs(md_dir, exist_ok=True)

    wt_md = run_openmm_md(
        input_pdb=prep.wt_pdb,
        ligand_xml=build_cfg.ligand_xml,
        md_config=md_cfg,
        output_prefix=os.path.join(md_dir, "wt"),
        resume=resume,
    )
    md_cfg_mut = md_cfg
    if md_cfg.random_seed is not None:
        md_cfg_mut = replace(md_cfg, random_seed=int(md_cfg.random_seed) + 1)
    mut_md = run_openmm_md(
        input_pdb=prep.mutant_ligand_pdb,
        ligand_xml=build_cfg.ligand_xml,
        md_config=md_cfg_mut,
        output_prefix=os.path.join(md_dir, "mut"),
        resume=resume,
    )

    analysis_dir = os.path.join(output_dir, "analysis")
    analysis = run_topology_safe_analysis(
        wt_pdb=wt_md.prepared_pdb,
        wt_traj_npz=wt_md.trajectory_npz,
        mut_pdb=mut_md.prepared_pdb,
        mut_traj_npz=mut_md.trajectory_npz,
        ligand_resnames=build_cfg.ligand_resnames,
        temperature_k=md_cfg.temperature_k,
        field_cfg=field_cfg,
        latent_cfg=latent_cfg,
        te_cfg=te_cfg,
        output_dir=analysis_dir,
    )
    vmd_script = write_vmd_comparison_script(
        wt_pdb=wt_md.prepared_pdb,
        wt_dcd=wt_md.trajectory_dcd,
        mut_pdb=mut_md.prepared_pdb,
        mut_dcd=mut_md.trajectory_dcd,
        ligand_resnames=build_cfg.ligand_resnames,
        output_tcl=os.path.join(analysis_dir, "view_wt_mut.vmd.tcl"),
        latents_npz=os.path.join(analysis_dir, "latents.npz"),
    )
    summary = {
        "structures": {
            "wt_with_ligand": prep.wt_pdb,
            "mutant_raw": prep.mutant_raw_pdb,
            "mutant_with_ligand": prep.mutant_ligand_pdb,
            "sequence_start_pos": prep.start_pos,
            "alignment_pairs": prep.alignment_pairs,
            "local_alignment_pairs": prep.local_alignment_pairs,
            "local_alignment_rmsd_ang": prep.local_alignment_rmsd_ang,
            "ligand_atoms": prep.ligand_atoms,
        },
        "md": {
            "wt": {
                "prepared_pdb": wt_md.prepared_pdb,
                "traj_npz": wt_md.trajectory_npz,
                "traj_dcd": wt_md.trajectory_dcd,
                "frames": wt_md.n_frames,
                "log": wt_md.log_path,
                "resumes": wt_md.resumes,
                "reused": wt_md.reused,
            },
            "mut": {
                "prepared_pdb": mut_md.prepared_pdb,
                "traj_npz": mut_md.trajectory_npz,
                "traj_dcd": mut_md.trajectory_dcd,
                "frames": mut_md.n_frames,
                "log": mut_md.log_path,
                "resumes": mut_md.resumes,
                "reused": mut_md.reused,
            },
        },
        "analysis_path": os.path.join(analysis_dir, "analysis_results.json"),
        "analysis": analysis,
        "visualization": {
            "vmd_script": vmd_script,
        },
        # Everything needed to know how this run was produced and to repeat it.
        "config": {
            "build": asdict(build_cfg),
            "md_wt": asdict(md_cfg),
            "md_mut": asdict(md_cfg_mut),
            "field": asdict(field_cfg),
            "latent": asdict(latent_cfg),
            "te": {**asdict(te_cfg), "ridge_values": [float(v) for v in te_cfg.ridge_values]},
        },
        "provenance": collect_provenance(),
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
