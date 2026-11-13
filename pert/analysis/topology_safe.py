"""Stage 3: the complete topology-safe WT-vs-MUT comparison.

Chains together field embedding -> shared VAMP latents -> basin ddG,
distribution shift, conditional transfer entropy and susceptibility, and writes
``analysis_results.json`` + ``latents.npz``.

The fields and the estimands use no WT<->MUT atom mapping, which is what makes
the comparison valid for insertions, deletions and duplications.  The only
correspondence used is the handful of conserved site residues that anchor the
shared coordinate frame (``pert.analysis.site_frame``).

``comparison_metrics`` reduces a result to the scalars that are compared
against the WT-vs-WT null distribution (``pert.pipelines.null_control``).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Optional, Sequence

import numpy as np

from pert.analysis.ensemble import compute_basin_ddg, distribution_shift
from pert.analysis.pocket_fields import compute_region_features
from pert.analysis.site_frame import build_shared_site_frames
from pert.analysis.susceptibility import susceptibility_gamma
from pert.analysis.transfer_entropy import conditional_te_linear_gaussian
from pert.analysis.vamp import LinearVAMP
from pert.config import FieldConfig, LatentConfig, TEConfig
from pert.deps import app, ensure_openmm
from pert.simulation.trajectory import load_traj_npz


def run_topology_safe_analysis(
    wt_pdb: str,
    wt_traj_npz: str,
    mut_pdb: str,
    mut_traj_npz: str,
    ligand_resnames: Sequence[str],
    temperature_k: float,
    field_cfg: FieldConfig,
    latent_cfg: LatentConfig,
    te_cfg: TEConfig,
    output_dir: str,
) -> Dict[str, object]:
    """Compute every region-level estimand for a WT/mutant trajectory pair.

    Writes ``latents.npz`` and ``analysis_results.json`` to ``output_dir`` and
    returns the results dict (basin ddG, distribution shift, TE for WT and mutant
    and their difference, susceptibility matrices and their difference).
    """
    ensure_openmm()
    os.makedirs(output_dir, exist_ok=True)

    wt_positions, wt_velocities, wt_dt_ps = load_traj_npz(wt_traj_npz)
    mut_positions, mut_velocities, mut_dt_ps = load_traj_npz(mut_traj_npz)
    if not np.isclose(wt_dt_ps, mut_dt_ps):
        raise ValueError(
            f"WT and MUT dt mismatch: {wt_dt_ps} ps vs {mut_dt_ps} ps. "
            "Use comparable sampling for TE/VAMP."
        )

    wt_top = app.PDBFile(wt_pdb).topology
    mut_top = app.PDBFile(mut_pdb).topology

    # Coordinate frame in which the two systems are compared.
    if field_cfg.frame_mode == "shared_site":
        wt_frame, mut_frame, frame_info = build_shared_site_frames(
            wt_pdb=wt_pdb,
            mut_pdb=mut_pdb,
            ligand_resnames=ligand_resnames,
            site_cutoff_ang=field_cfg.frame_site_cutoff_ang,
            min_anchors=field_cfg.frame_min_anchors,
        )
    elif field_cfg.frame_mode == "legacy_self":
        wt_frame = mut_frame = None
        frame_info = {}
    else:
        raise ValueError(
            f"Unknown frame_mode '{field_cfg.frame_mode}'. Use 'shared_site' or 'legacy_self'."
        )

    wt_feat = compute_region_features(
        topology=wt_top,
        positions=wt_positions,
        velocities=wt_velocities,
        dt_ps=wt_dt_ps,
        ligand_resnames=ligand_resnames,
        field_cfg=field_cfg,
        site_frame=wt_frame,
    )
    mut_feat = compute_region_features(
        topology=mut_top,
        positions=mut_positions,
        velocities=mut_velocities,
        dt_ps=mut_dt_ps,
        ligand_resnames=ligand_resnames,
        field_cfg=field_cfg,
        site_frame=mut_frame,
    )
    frame_report: Dict[str, object] = {"mode": field_cfg.frame_mode, **frame_info}
    if wt_frame is not None:
        # How much the site scaffold itself deviates from the WT reference.
        frame_report["site_fit_rmsd_ang_wt_mean"] = float(np.mean(wt_feat["frame_fit_rmsd_ang"]))
        frame_report["site_fit_rmsd_ang_mut_mean"] = float(np.mean(mut_feat["frame_fit_rmsd_ang"]))

    vamp_b = LinearVAMP(lag=latent_cfg.lag, n_components=latent_cfg.m_b, reg=latent_cfg.reg, solver=latent_cfg.solver)
    vamp_p = LinearVAMP(lag=latent_cfg.lag, n_components=latent_cfg.m_p, reg=latent_cfg.reg, solver=latent_cfg.solver)
    vamp_l = LinearVAMP(lag=latent_cfg.lag, n_components=latent_cfg.m_l, reg=latent_cfg.reg, solver=latent_cfg.solver)
    vamp_b.fit_from_trajectories([wt_feat["B"], mut_feat["B"]])
    vamp_p.fit_from_trajectories([wt_feat["P"], mut_feat["P"]])
    vamp_l.fit_from_trajectories([wt_feat["L"], mut_feat["L"]])

    u_b_wt = vamp_b.transform(wt_feat["B"])
    u_b_mut = vamp_b.transform(mut_feat["B"])
    u_p_wt = vamp_p.transform(wt_feat["P"])
    u_p_mut = vamp_p.transform(mut_feat["P"])
    u_l_wt = vamp_l.transform(wt_feat["L"])
    u_l_mut = vamp_l.transform(mut_feat["L"])

    basin = compute_basin_ddg(
        u_wt=u_b_wt,
        u_mut=u_b_mut,
        temperature_k=temperature_k,
        n_bootstrap=400,
        random_seed=te_cfg.random_seed,
    )

    dist_shift = distribution_shift(u_b_wt, u_b_mut)

    te_wt = conditional_te_linear_gaussian(y=u_b_wt, x=u_p_wt, w=u_l_wt, cfg=te_cfg)
    te_mut = conditional_te_linear_gaussian(y=u_b_mut, x=u_p_mut, w=u_l_mut, cfg=te_cfg)

    gamma_wt = susceptibility_gamma(u_b_wt, u_p_wt, random_seed=te_cfg.random_seed)
    gamma_mut = susceptibility_gamma(u_b_mut, u_p_mut, random_seed=te_cfg.random_seed + 1)

    delta_gamma = np.asarray(gamma_mut["gamma_matrix"]) - np.asarray(gamma_wt["gamma_matrix"])
    delta_gamma_norm = float(np.linalg.norm(delta_gamma, ord="fro"))

    np.savez_compressed(
        os.path.join(output_dir, "latents.npz"),
        u_b_wt=u_b_wt,
        u_b_mut=u_b_mut,
        u_p_wt=u_p_wt,
        u_p_mut=u_p_mut,
        u_l_wt=u_l_wt,
        u_l_mut=u_l_mut,
    )

    results = {
        "topology_constraint": (
            "No WT↔MUT atom mapping is used in the fields or the estimands; conserved site "
            "residues only anchor the shared coordinate frame."
            if wt_frame is not None
            else "No WT↔MUT atom/residue one-to-one mapping is used in embedding/estimands."
        ),
        "temperature_k": temperature_k,
        "ligand_resnames": [str(name) for name in ligand_resnames],
        # Complete configurations, so a result can be reproduced and compared
        # with others computed the same way (see the null control).
        "field_config": asdict(field_cfg),
        "latent_config": asdict(latent_cfg),
        "te_config": {**asdict(te_cfg), "ridge_values": [float(v) for v in te_cfg.ridge_values]},
        "frame": frame_report,
        # Sizes, singular values and over-fitting indicators of the three projections.
        "vamp": {"B": vamp_b.diagnostics(), "P": vamp_p.diagnostics(), "L": vamp_l.diagnostics()},
        "basin_ddg": basin,
        "distribution_shift": dist_shift,
        "te_wt_p_to_b_given_l": te_wt,
        "te_mut_p_to_b_given_l": te_mut,
        "te_delta_mut_minus_wt": float(
            te_mut["te_driver_to_target"] - te_wt["te_driver_to_target"]
        ),
        "susceptibility_wt": gamma_wt,
        "susceptibility_mut": gamma_mut,
        "susceptibility_delta_fro_norm": delta_gamma_norm,
    }
    with open(os.path.join(output_dir, "analysis_results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


# Scalars that summarise "how different is the mutant from the WT".  For all of
# them a larger value means a larger difference, so they can be ranked against
# a WT-vs-WT null distribution.
COMPARISON_METRICS = (
    "js_divergence",
    "classifier_auc",
    "basin_population_tv",
    "abs_te_delta",
    "susceptibility_delta_fro_norm",
)


def _number(value: Optional[object]) -> float:
    return float("nan") if value is None else float(value)


def comparison_metrics(results: Dict[str, object]) -> Dict[str, float]:
    """Headline scalars of one comparison (the ``COMPARISON_METRICS`` plus the signed delta TE).

    Works on results produced before the null control existed: metrics that are
    missing come back as NaN.
    """
    dist = results.get("distribution_shift") or {}
    basin = results.get("basin_ddg") or {}
    te_delta = _number(results.get("te_delta_mut_minus_wt"))
    return {
        "js_divergence": _number(dist.get("js_divergence")),
        "classifier_auc": _number(dist.get("classifier_auc")),
        "basin_population_tv": _number(basin.get("population_tv_distance")),
        "te_delta": te_delta,
        "abs_te_delta": abs(te_delta),
        "susceptibility_delta_fro_norm": _number(results.get("susceptibility_delta_fro_norm")),
    }
