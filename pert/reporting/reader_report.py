"""Plain-language Markdown report for one run (``analysis/reader_report.md``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pert.structure.binding_site import read_binding_residue_keys
from pert.structure.selection import ligand_selection
from pert.utils.io import load_json, md_table
from pert.utils.paths import resolve_run_path


def _optional_float(value: Any) -> Optional[float]:
    """``float(value)`` or None."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_float(value: Optional[float], nd: int = 3) -> str:
    """Fixed-decimal formatting with ``NA`` for None/NaN."""
    if value is None:
        return "NA"
    if value != value:  # NaN
        return "NA"
    return f"{value:.{nd}f}"


@dataclass(frozen=True)
class _RunPaths:
    """Locations of the files of one run that the report reads or links to."""
    run_dir: str
    analysis_dir: str
    run_summary_json: str
    analysis_results_json: str
    flow_summary_json: str
    binding_csv: str
    allostery_manifest_json: str
    allostery_residue_csv: str
    allostery_region_csv: str
    allostery_network_json: str
    vmd_wt_mut: str
    vmd_flow: str
    vmd_allostery: str


def _infer_paths(run_dir: str) -> _RunPaths:
    """Standard file locations inside a run directory."""
    analysis_dir = os.path.join(run_dir, "analysis")
    return _RunPaths(
        run_dir=run_dir,
        analysis_dir=analysis_dir,
        run_summary_json=os.path.join(run_dir, "run_summary.json"),
        analysis_results_json=os.path.join(analysis_dir, "analysis_results.json"),
        flow_summary_json=os.path.join(analysis_dir, "flow_summary.json"),
        binding_csv=os.path.join(analysis_dir, "binding_site_residues.csv"),
        allostery_manifest_json=os.path.join(analysis_dir, "allostery_manifest.json"),
        allostery_residue_csv=os.path.join(analysis_dir, "allostery_residue_scores.csv"),
        allostery_region_csv=os.path.join(analysis_dir, "allostery_region_scores.csv"),
        allostery_network_json=os.path.join(analysis_dir, "allostery_network.json"),
        vmd_wt_mut=os.path.join(analysis_dir, "view_wt_mut.vmd.tcl"),
        vmd_flow=os.path.join(analysis_dir, "view_flow_network.vmd.tcl"),
        vmd_allostery=os.path.join(analysis_dir, "view_allostery_hotspots.vmd.tcl"),
    )


def _extract_te_block(analysis: Dict[str, Any], key: str) -> Dict[str, Any]:
    """TE result block of the analysis dict (empty dict if missing)."""
    block = analysis.get(key)
    return block if isinstance(block, dict) else {}


def _best_candidate(te_block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The VAR candidate the estimator selects: lowest BIC among accepted models (all models if none passed)."""
    candidates = te_block.get("model_candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    # Prefer accepted models; otherwise the lowest BIC.
    accepted = [c for c in candidates if isinstance(c, dict) and bool(c.get("accepted"))]
    pool = accepted if accepted else [c for c in candidates if isinstance(c, dict)]
    if not pool:
        return None
    return sorted(pool, key=lambda c: float(c.get("bic", 1e30)))[0]


def write_reader_report(run_dir: str, out_path: str) -> str:
    """Write the Markdown reader report of a run and return its path."""
    paths = _infer_paths(run_dir)
    summary = load_json(paths.run_summary_json)
    analysis = load_json(paths.analysis_results_json)
    flow = load_json(paths.flow_summary_json) if os.path.exists(paths.flow_summary_json) else {}
    allostery = load_json(paths.allostery_network_json) if os.path.exists(paths.allostery_network_json) else {}

    # Identify “mutation label” from folder name.
    mutation_label = os.path.basename(os.path.dirname(run_dir))  # e.g. snp_k2r
    replicate_label = os.path.basename(run_dir)  # e.g. rep1

    ligand_resname = "ZN"
    try:
        ligand_resname = str(summary.get("analysis", {}).get("ligand_resname", ligand_resname))
    except Exception:
        pass

    temperature_k = _optional_float(analysis.get("temperature_k")) or 300.0
    kbt_kcal_mol = 0.00198720425864083 * temperature_k  # k_B [kcal/mol/K] * T

    field_cfg = analysis.get("field_config") if isinstance(analysis.get("field_config"), dict) else {}
    latent_cfg = analysis.get("latent_config") if isinstance(analysis.get("latent_config"), dict) else {}

    # Basin DDG
    basin_block = analysis.get("basin_ddg") if isinstance(analysis.get("basin_ddg"), dict) else {}
    basins = basin_block.get("basins") if isinstance(basin_block.get("basins"), list) else []
    basin_rows: List[List[str]] = []
    for row in basins:
        if not isinstance(row, dict):
            continue
        basin_rows.append(
            [
                str(row.get("basin", "")),
                _fmt_float(_optional_float(row.get("p_wt")), 6),
                _fmt_float(_optional_float(row.get("p_mut")), 6),
                # Results written before the overlap flag existed have no "overlap" key.
                "NA" if row.get("overlap") is None else ("yes" if row.get("overlap") else "no"),
                _fmt_float(_optional_float(row.get("ddg_kcal_mol")), 3),
                _fmt_float(_optional_float(row.get("ddg_ci95_low")), 3),
                _fmt_float(_optional_float(row.get("ddg_ci95_high")), 3),
            ]
        )

    # Distribution shift
    dist = analysis.get("distribution_shift") if isinstance(analysis.get("distribution_shift"), dict) else {}
    kl = _optional_float(dist.get("kl_wt_to_mut"))
    diag_wt = dist.get("gaussian_diagnostics_wt") if isinstance(dist.get("gaussian_diagnostics_wt"), dict) else {}
    diag_mut = dist.get("gaussian_diagnostics_mut") if isinstance(dist.get("gaussian_diagnostics_mut"), dict) else {}

    # TE blocks
    te_wt = _extract_te_block(analysis, "te_wt_p_to_b_given_l")
    te_mut = _extract_te_block(analysis, "te_mut_p_to_b_given_l")
    te_delta = _optional_float(analysis.get("te_delta_mut_minus_wt"))

    te_wt_best = _best_candidate(te_wt) or {}
    te_mut_best = _best_candidate(te_mut) or {}

    # Susceptibility
    sus_wt = analysis.get("susceptibility_wt") if isinstance(analysis.get("susceptibility_wt"), dict) else {}
    sus_mut = analysis.get("susceptibility_mut") if isinstance(analysis.get("susceptibility_mut"), dict) else {}
    sus_delta_f = _optional_float(analysis.get("susceptibility_delta_fro_norm"))

    # Binding site residues (for “what is being targeted”)
    binding_res_keys = read_binding_residue_keys(paths.binding_csv)

    # Residue flow (intuitive proxy)
    flow_total_wt = _optional_float(flow.get("total_flow_wt"))
    flow_total_mut = _optional_float(flow.get("total_flow_mut"))
    flow_total_delta = _optional_float(flow.get("total_flow_delta_mut_minus_wt"))

    flow_sources = flow.get("source_residues") if isinstance(flow.get("source_residues"), list) else []
    flow_targets = flow.get("target_residues") if isinstance(flow.get("target_residues"), list) else []
    top_paths = flow.get("top_mutation_to_binding_paths") if isinstance(flow.get("top_mutation_to_binding_paths"), list) else []
    top_path_str = "NA"
    if top_paths and isinstance(top_paths[0], dict) and isinstance(top_paths[0].get("path"), list):
        top_path_str = " → ".join(str(x) for x in top_paths[0]["path"])

    top_allostery_residues = []
    if isinstance(allostery.get("top_residues"), list):
        top_allostery_residues = [row for row in allostery["top_residues"] if isinstance(row, dict)]
    top_allostery_regions = []
    if isinstance(allostery.get("top_regions"), list):
        top_allostery_regions = [row for row in allostery["top_regions"] if isinstance(row, dict)]

    # Paths to important files (prefer relative for readability)
    rel = lambda p: os.path.relpath(p, run_dir)
    wt_pdb = resolve_run_path(run_dir, summary["md"]["wt"]["prepared_pdb"])
    mut_pdb = resolve_run_path(run_dir, summary["md"]["mut"]["prepared_pdb"])
    wt_dcd = resolve_run_path(run_dir, summary["md"]["wt"]["traj_dcd"])
    mut_dcd = resolve_run_path(run_dir, summary["md"]["mut"]["traj_dcd"])

    lines: List[str] = []
    lines.append(f"# WT vs MUT Pocket Perturbation Report: `{mutation_label}` ({replicate_label})")
    lines.append("")
    lines.append("This report explains what the pipeline computed, what each output means, and how to interpret it physically.")
    lines.append("")

    lines.append("## 1) What was analyzed")
    lines.append(f"- **Run directory**: `{run_dir}`")
    lines.append(f"- **WT structure**: `{wt_pdb}`")
    lines.append(f"- **MUT structure**: `{mut_pdb}`")
    lines.append(f"- **WT trajectory**: `{wt_dcd}`")
    lines.append(f"- **MUT trajectory**: `{mut_dcd}`")
    lines.append(f"- **Temperature**: `{temperature_k:.1f} K` (so `kBT ≈ {kbt_kcal_mol:.3f} kcal/mol`) ")
    lines.append("")

    lines.append("## 2) Topology-safe core idea (why it works for INS/DEL/DUP/etc.)")
    lines.append("- Regions are defined geometrically (distance to a pocket center), so they exist even if residue correspondence breaks.")
    lines.append("- Local structure/dynamics are embedded into fixed-size, permutation-invariant fields (grid + kernels), so differing atom counts are OK.")
    lines.append("- WT and MUT are projected into a shared low-dimensional latent space (pooled VAMP fit), so comparisons are on the same axes.")
    lines.append("")

    lines.append("## 3) Configuration actually used in this run")
    lines.append(
        f"- **Pocket radii**: `R_B={_fmt_float(_optional_float(field_cfg.get('rb_ang')),1)} Å`, `R_P={_fmt_float(_optional_float(field_cfg.get('rp_ang')),1)} Å`"
    )
    lines.append(
        f"- **Grid**: spacing `{_fmt_float(_optional_float(field_cfg.get('grid_spacing_ang')),1)} Å`, kernel σ `{_fmt_float(_optional_float(field_cfg.get('sigma_ang')),1)} Å`, max points `{field_cfg.get('max_grid_points','NA')}`"
    )
    lines.append(f"- **Velocities in features**: `{field_cfg.get('include_velocity','NA')}`")
    lines.append(f"- **COM translation removed**: `{field_cfg.get('remove_com','NA')}`")
    lines.append(
        f"- **VAMP lag**: `{latent_cfg.get('lag','NA')}` frames; dims `m_B={latent_cfg.get('m_b','NA')}`, `m_P={latent_cfg.get('m_p','NA')}`, `m_L={latent_cfg.get('m_l','NA')}`"
    )
    lines.append("")

    vamp_diag = analysis.get("vamp") if isinstance(analysis.get("vamp"), dict) else {}
    if vamp_diag:
        risky = [region for region, diag in vamp_diag.items() if isinstance(diag, dict) and diag.get("overfit_risk")]
        pocket = vamp_diag.get("B") if isinstance(vamp_diag.get("B"), dict) else {}
        lines.append(
            f"- **Slow-coordinate sampling (pocket)**: `{pocket.get('n_pairs', 'NA')}` lagged pairs for "
            f"`{pocket.get('effective_dimension', 'NA')}` effective dimensions; leading singular values "
            f"`{[round(float(v), 3) for v in (pocket.get('singular_values') or [])[:4]]}`"
        )
        if risky:
            lines.append(
                f"- **Warning**: too few frames to resolve slow coordinates in region(s) {', '.join(risky)} "
                "(fewer than 5 lagged pairs per effective dimension). The latent coordinates, and everything "
                "computed from them, are over-fitted to this sample."
            )
        lines.append("")

    lines.append("## 4) Output A: Pocket macrostate stabilization shift (ΔΔG in latent space)")
    lines.append("The basin score is:")
    lines.append("")
    lines.append("`ΔΔG(A) = -kBT * log(p_MUT(A) / p_WT(A))`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Negative: basin A is stabilized in MUT (MUT spends more time there).")
    lines.append("- Positive: basin A is destabilized in MUT.")
    lines.append("- This is an **end-state population shift in reduced pocket space**, not a formal standard-state binding free energy.")
    lines.append("- A basin visited by only one system (`visited by both = no`) has **no defined ΔΔG**; it means the two trajectories do not overlap there.")
    lines.append("")
    if basin_rows:
        lines.append(
            md_table(
                ["basin", "p_WT", "p_MUT", "visited by both", "ΔΔG (kcal/mol)", "CI95 low", "CI95 high"],
                basin_rows,
            )
        )
        lines.append("")
        tv = _optional_float(basin_block.get("population_tv_distance"))
        if tv is not None:
            lines.append(
                f"- **Basin population shift (total variation)**: `{_fmt_float(tv, 3)}` "
                f"(CI95 `{_fmt_float(_optional_float(basin_block.get('population_tv_ci95_low')), 3)}` to "
                f"`{_fmt_float(_optional_float(basin_block.get('population_tv_ci95_high')), 3)}`; 0 = same populations, 1 = disjoint)"
            )
        lines.append(
            f"- **Effective samples**: WT `{_fmt_float(_optional_float(basin_block.get('effective_samples_wt')), 1)}`, "
            f"MUT `{_fmt_float(_optional_float(basin_block.get('effective_samples_mut')), 1)}`"
        )
        if basin_block.get("sampling_note"):
            lines.append(f"- **Sampling**: {basin_block.get('sampling_note')}")
        lines.append("")
    else:
        lines.append("- No basin results were found in `analysis_results.json`.")
        lines.append("")

    lines.append("## 5) Output B: Pocket distribution shift (Gaussian KL if valid)")
    lines.append("If the pocket latent distribution is approximately Gaussian in each ensemble, the pipeline reports the closed-form Gaussian KL.")
    lines.append("")
    lines.append(f"- **Method**: `{dist.get('method','NA')}`")
    lines.append(f"- **KL(WT || MUT)**: `{_fmt_float(kl, 3)}` (bigger means larger WT→MUT shift in mean/covariance)")
    lines.append(
        f"- **Held-out separability**: classifier AUC `{_fmt_float(_optional_float(dist.get('classifier_auc')), 3)}` "
        f"(0.5 = indistinguishable, 1 = fully separable), JS divergence `{_fmt_float(_optional_float(dist.get('js_divergence')), 3)}` "
        f"(0 to 0.693). Estimator: `{dist.get('js_estimator', 'in-sample (legacy result)')}`."
    )
    lines.append(
        "- Separability is expected even between two WT simulations that have not mixed; "
        "judge it against the WT-vs-WT null control (`python -m pert null-control`)."
    )
    lines.append(
        f"- **Gaussianity diagnostics**: WT passed `{diag_wt.get('passed_for_gaussian_kl','NA')}` (min p-value `{_fmt_float(_optional_float(diag_wt.get('normality_min_pvalue')),3)}`), "
        f"MUT passed `{diag_mut.get('passed_for_gaussian_kl','NA')}` (min p-value `{_fmt_float(_optional_float(diag_mut.get('normality_min_pvalue')),3)}`)"
    )
    lines.append("")

    lines.append("## 6) Output C: Directed predictive dependence (linear-Gaussian TE/Granger)")
    lines.append("This answers: **does past context help predict the pocket beyond the pocket’s own history (and ligand history)?**")
    lines.append("")
    lines.append("For linear-Gaussian VAR models, conditional TE is:")
    lines.append("")
    lines.append("`TE = 0.5 * log(det(Sigma_restricted) / det(Sigma_full))`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Larger TE means the driver carries predictive information about the target beyond what’s already in the target’s own past.")
    lines.append("- This is **predictive dependence**, not proof of physical causation. Diagnostics and reverse-direction controls matter.")
    lines.append("")
    lines.append(
        md_table(
            ["ensemble", "TE(P→B|L)", "TE(B→P|L)", "surrogate p-value", "selected p", "accepted? (best)"],
            [
                [
                    "WT",
                    _fmt_float(_optional_float(te_wt.get("te_driver_to_target")), 3),
                    _fmt_float(_optional_float(te_wt.get("te_target_to_driver")), 3),
                    _fmt_float(_optional_float(te_wt.get("surrogate_pvalue")), 3),
                    str(te_wt.get("selected_order", "NA")),
                    str(bool(te_wt_best.get("accepted"))) if te_wt_best else "NA",
                ],
                [
                    "MUT",
                    _fmt_float(_optional_float(te_mut.get("te_driver_to_target")), 3),
                    _fmt_float(_optional_float(te_mut.get("te_target_to_driver")), 3),
                    _fmt_float(_optional_float(te_mut.get("surrogate_pvalue")), 3),
                    str(te_mut.get("selected_order", "NA")),
                    str(bool(te_mut_best.get("accepted"))) if te_mut_best else "NA",
                ],
            ],
        )
    )
    lines.append("")
    lines.append(f"- **ΔTE (MUT−WT)**: `{_fmt_float(te_delta, 3)}`")
    lines.append("")
    if te_wt_best or te_mut_best:
        lines.append("Best-model diagnostics (lowest BIC among accepted; otherwise lowest BIC overall):")
        lines.append(
            md_table(
                ["ensemble", "whiteness max |ACF|", "spectral radius", "BIC"],
                [
                    [
                        "WT",
                        _fmt_float(_optional_float(te_wt_best.get("whiteness_max_abs_acf")), 3) if te_wt_best else "NA",
                        _fmt_float(_optional_float(te_wt_best.get("spectral_radius")), 3) if te_wt_best else "NA",
                        _fmt_float(_optional_float(te_wt_best.get("bic")), 1) if te_wt_best else "NA",
                    ],
                    [
                        "MUT",
                        _fmt_float(_optional_float(te_mut_best.get("whiteness_max_abs_acf")), 3) if te_mut_best else "NA",
                        _fmt_float(_optional_float(te_mut_best.get("spectral_radius")), 3) if te_mut_best else "NA",
                        _fmt_float(_optional_float(te_mut_best.get("bic")), 1) if te_mut_best else "NA",
                    ],
                ],
            )
        )
        lines.append("")
        lines.append("Rule of thumb:")
        lines.append("- Lower `whiteness max |ACF|` is better (residuals less autocorrelated).")
        lines.append("- `spectral radius < 1` indicates a stable VAR.")
        lines.append("")

    lines.append("## 7) Output D: Susceptibility (fast equilibrium coupling proxy)")
    lines.append("This estimates a linear-response mapping `Γ = Σ_BP Σ_PP^{-1}` between context and pocket latents.")
    lines.append("")
    lines.append(
        md_table(
            ["ensemble", "||Γ||_F", "top singular value", "CI95 ||Γ||_F"],
            [
                [
                    "WT",
                    _fmt_float(_optional_float(sus_wt.get("fro_norm")), 3),
                    _fmt_float(_optional_float(sus_wt.get("top_singular_value")), 3),
                    f"{_fmt_float(_optional_float(sus_wt.get('fro_norm_ci95_low')),3)}–{_fmt_float(_optional_float(sus_wt.get('fro_norm_ci95_high')),3)}",
                ],
                [
                    "MUT",
                    _fmt_float(_optional_float(sus_mut.get("fro_norm")), 3),
                    _fmt_float(_optional_float(sus_mut.get("top_singular_value")), 3),
                    f"{_fmt_float(_optional_float(sus_mut.get('fro_norm_ci95_low')),3)}–{_fmt_float(_optional_float(sus_mut.get('fro_norm_ci95_high')),3)}",
                ],
            ],
        )
    )
    lines.append("")
    lines.append(f"- **Δ||Γ||_F proxy (pipeline)**: `{_fmt_float(sus_delta_f, 3)}`")
    lines.append("")

    lines.append("## 8) Residue-level flow proxy (for intuition + VMD)")
    lines.append("This is a fast residue-motion-based directed predictability proxy (separate from the topology-safe field/VAMP layer).")
    lines.append("")
    lines.append(f"- **Flow total (WT)**: `{_fmt_float(flow_total_wt, 3)}`")
    lines.append(f"- **Flow total (MUT)**: `{_fmt_float(flow_total_mut, 3)}`")
    lines.append(f"- **ΔFlow (MUT−WT)**: `{_fmt_float(flow_total_delta, 3)}`")
    lines.append(f"- **Source residues (mutation hotspots)**: `{', '.join(str(x) for x in flow_sources) if flow_sources else 'NA'}`")
    lines.append(f"- **Target residues (binding-site neighborhood)**: `{', '.join(str(x) for x in flow_targets) if flow_targets else 'NA'}`")
    lines.append(f"- **Top inferred route**: `{top_path_str}`")
    lines.append("")
    lines.append("How to interpret the `view_flow_network.vmd.tcl` animation:")
    lines.append("- The tube and moving ball are an **inferred communication route proxy**, not a physical particle or energy packet.")
    lines.append("- Ball size/color tracks the windowed `Δflow`; bigger means stronger mutation-associated coupling to the site in that window.")
    lines.append("- The 2D panel plots `Δflow` over time windows; the marker follows the current playback frame.")
    lines.append("")

    lines.append("## 9) Allostery residue and region ranking")
    if top_allostery_residues:
        rows = []
        for row in top_allostery_residues[:10]:
            rows.append(
                [
                    str(row.get("rank", "")),
                    str(row.get("node_label", "")),
                    _fmt_float(_optional_float(row.get("total_score")), 3),
                    str(row.get("is_source_hotspot", "NA")),
                    str(row.get("is_target_site", "NA")),
                ]
            )
        lines.append(md_table(["rank", "residue", "score", "source?", "target?"], rows))
        lines.append("")
        if top_allostery_regions:
            region_rows = []
            for row in top_allostery_regions[:5]:
                region_rows.append(
                    [
                        str(row.get("rank", "")),
                        str(row.get("region", "")),
                        _fmt_float(_optional_float(row.get("mean_total_score")), 3),
                        str(row.get("top_residue", "")),
                    ]
                )
            lines.append(md_table(["rank", "region", "mean score", "top residue"], region_rows))
            lines.append("")
        lines.append(f"- Residue scores: `{rel(paths.allostery_residue_csv)}`")
        lines.append(f"- Region scores: `{rel(paths.allostery_region_csv)}`")
        lines.append(f"- Network details: `{rel(paths.allostery_network_json)}`")
        lines.append(f"- Hotspot VMD view: `vmd -e {rel(paths.vmd_allostery)}`")
    else:
        lines.append("- Allostery ranking has not been generated yet. Run `python -m pert rank --run-dir <run-dir>`.")
    lines.append("")

    lines.append("## 10) Visual checks in VMD (recommended)")
    lines.append(f"- WT vs MUT structure/trajectory compare: `vmd -e {rel(paths.vmd_wt_mut)}`")
    lines.append(f"- MUT residue-flow proxy view: `vmd -e {rel(paths.vmd_flow)}`")
    if os.path.exists(paths.vmd_allostery):
        lines.append(f"- Allostery hotspot residue view: `vmd -e {rel(paths.vmd_allostery)}`")
    lines.append("")

    lines.append("## 11) What to trust (and what not to over-interpret)")
    lines.append("- **Trust most**: basin population shifts (ΔΔG in latent space) and large latent distribution shifts (KL), when sampling is adequate.")
    lines.append("- **Treat carefully**: TE/Granger (requires stable, approximately white residuals; check surrogate p-values and reverse direction).")
    lines.append("- **Treat as visualization/intuition**: residue-level flow and the moving ball path (it summarizes a statistical proxy).")
    lines.append("")

    lines.append("## 12) Binding-site definition used here")
    lines.append(f"- Ligand: `{ligand_resname}` (selection is `{ligand_selection(ligand_resname).vmd()}`)")
    lines.append(f"- Binding-site residues file: `{rel(paths.binding_csv)}`")
    lines.append(f"- Binding-site residue keys (count {len(binding_res_keys)}): `{', '.join(binding_res_keys) if binding_res_keys else 'NA'}`")
    lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return out_path
