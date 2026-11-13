"""Machine-readable interpretation summary of one run
(``analysis/interpretation_summary.json``).
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np


def write_interpretation_summary(
    summary: Dict,
    analysis: Dict,
    labels_wt: np.ndarray,
    labels_mut: np.ndarray,
    out_json: str,
) -> None:
    """Write the headline numbers of a run (TE, p-values, divergences, basin populations) as JSON."""
    basin_wt = np.bincount(labels_wt, minlength=analysis["basin_ddg"]["n_basins"]) / len(labels_wt)
    basin_mut = np.bincount(labels_mut, minlength=analysis["basin_ddg"]["n_basins"]) / len(labels_mut)
    payload = {
        "frames_wt": int(summary["md"]["wt"]["frames"]),
        "frames_mut": int(summary["md"]["mut"]["frames"]),
        "te_wt_p_to_b_given_l": analysis["te_wt_p_to_b_given_l"]["te_driver_to_target"],
        "te_mut_p_to_b_given_l": analysis["te_mut_p_to_b_given_l"]["te_driver_to_target"],
        "te_delta_mut_minus_wt": analysis["te_delta_mut_minus_wt"],
        "wt_surrogate_pvalue": analysis["te_wt_p_to_b_given_l"]["surrogate_pvalue"],
        "mut_surrogate_pvalue": analysis["te_mut_p_to_b_given_l"]["surrogate_pvalue"],
        "distribution_method": analysis["distribution_shift"]["method"],
        "js_divergence": analysis["distribution_shift"].get("js_divergence"),
        "classifier_auc": analysis["distribution_shift"].get("classifier_auc"),
        "basin_population_tv_distance": analysis["basin_ddg"].get("population_tv_distance"),
        "sampling_ok": analysis["basin_ddg"].get("sampling_ok"),
        "gmm_kl_wt_to_mut": analysis["distribution_shift"].get("gmm_kl_wt_to_mut"),
        "susceptibility_delta_fro_norm": analysis["susceptibility_delta_fro_norm"],
        "empirical_basin_pop_wt": basin_wt.tolist(),
        "empirical_basin_pop_mut": basin_mut.tolist(),
    }
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
