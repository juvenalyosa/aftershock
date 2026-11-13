"""Summary figures across mutation scenarios (MD panel, fast panel, null control).
"""

from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np

from pert.utils.mpl import plt


def plot_md_panel(agg: Dict[str, Dict[str, float]], out_png: str) -> None:
    """Four-panel comparison of the MD panel (delta TE, JS, susceptibility shift, effect map)."""
    names = list(agg.keys())
    x = np.arange(len(names))

    te_delta_mean = np.array([agg[n]["te_delta_mut_minus_wt_mean"] for n in names], dtype=float)
    te_delta_std = np.array([agg[n]["te_delta_mut_minus_wt_std"] for n in names], dtype=float)
    js_mean = np.array([agg[n]["js_divergence_mean"] for n in names], dtype=float)
    js_std = np.array([agg[n]["js_divergence_std"] for n in names], dtype=float)
    sus_mean = np.array([agg[n]["susceptibility_delta_fro_norm_mean"] for n in names], dtype=float)
    sus_std = np.array([agg[n]["susceptibility_delta_fro_norm_std"] for n in names], dtype=float)
    p_mean = np.array([agg[n]["te_mut_pvalue_mean"] for n in names], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].bar(x, te_delta_mean, yerr=te_delta_std, color="#1f77b4", capsize=4)
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=20, ha="right")
    axes[0, 0].set_title("ΔTE (MUT-WT)")
    axes[0, 0].set_ylabel("nats")
    axes[0, 0].grid(alpha=0.2, axis="y")

    axes[0, 1].bar(x, js_mean, yerr=js_std, color="#d62728", capsize=4)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=20, ha="right")
    axes[0, 1].set_title("Distribution Shift (JS)")
    axes[0, 1].set_ylabel("JS divergence")
    axes[0, 1].grid(alpha=0.2, axis="y")

    axes[1, 0].bar(x, sus_mean, yerr=sus_std, color="#2ca02c", capsize=4)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=20, ha="right")
    axes[1, 0].set_title("Susceptibility Shift")
    axes[1, 0].set_ylabel("||ΔΓ||F")
    axes[1, 0].grid(alpha=0.2, axis="y")

    axes[1, 1].scatter(js_mean, te_delta_mean, s=180, c=sus_mean, cmap="viridis", edgecolor="black")
    for i, name in enumerate(names):
        axes[1, 1].annotate(name, (js_mean[i], te_delta_mean[i]), textcoords="offset points", xytext=(6, 6))
    axes[1, 1].set_xlabel("JS divergence")
    axes[1, 1].set_ylabel("ΔTE (MUT-WT)")
    axes[1, 1].set_title("Mutation Effect Map (color=||ΔΓ||F)")
    axes[1, 1].grid(alpha=0.25)

    text_lines = [f"{n}: p={p_mean[i]:.3f}" for i, n in enumerate(names)]
    axes[1, 1].text(
        0.03,
        0.03,
        "\n".join(text_lines),
        transform=axes[1, 1].transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    fig.suptitle("Mutation Panel Comparison")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_fast_panel_summary(summary: Dict[str, Dict[str, float]], out_png: str) -> None:
    """Fast score, destabilisation proxy and CA RMSD per scenario of the fast panel."""
    if not summary:
        return
    names = list(summary.keys())
    x = np.arange(len(names), dtype=float)
    score = np.asarray([summary[n]["global_fast_allostery_score_mean"] for n in names], dtype=float)
    score_sd = np.asarray([summary[n]["global_fast_allostery_score_std"] for n in names], dtype=float)
    destab = np.asarray([summary[n]["mean_top10_destabilization_proxy_mean"] for n in names], dtype=float)
    rmsd = np.asarray([summary[n]["aligned_ca_rmsd_ang_mean"] for n in names], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].bar(x, score, yerr=score_sd, color="#4C78A8", capsize=4)
    axes[0].set_title("Fast allostery score")
    axes[0].set_ylabel("score")
    axes[1].bar(x, destab, color="#E45756")
    axes[1].set_title("Destabilization proxy")
    axes[2].bar(x, rmsd, color="#54A24B")
    axes[2].set_title("Aligned CA RMSD")
    axes[2].set_ylabel("Angstrom")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("Fast Mutation Screening Panel")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_null_control(
    metrics: Sequence[str],
    null_values: Dict[str, Sequence[float]],
    observed_by_scenario: Dict[str, Dict[str, Sequence[float]]],
    out_png: str,
) -> None:
    """Observed WT-vs-MUT metrics against the WT-vs-WT null, one panel per metric.

    The grey band spans the null pairs: a scenario whose points sit inside it
    is not distinguishable from two simulations of the unmutated protein.
    """
    scenarios = list(observed_by_scenario.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 4.4), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, 10))
    for ax, metric in zip(axes[0], metrics):
        null = np.asarray([v for v in null_values.get(metric, []) if np.isfinite(v)], dtype=float)
        if null.size:
            ax.axhspan(float(np.min(null)), float(np.max(null)), color="#B0B0B0", alpha=0.35, zorder=0)
            jitter = np.linspace(-0.18, 0.18, null.size) if null.size > 1 else np.zeros(1)
            ax.scatter(jitter, null, color="#555555", s=28, zorder=2)
        for index, scenario in enumerate(scenarios, start=1):
            values = np.asarray(observed_by_scenario[scenario].get(metric, []), dtype=float)
            values = values[np.isfinite(values)]
            jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.zeros(values.size)
            ax.scatter(index + jitter, values, color=colors[(index - 1) % 10], s=46, edgecolor="black", zorder=3)
        ax.set_xticks(range(len(scenarios) + 1))
        ax.set_xticklabels(["WT vs WT"] + scenarios, rotation=35, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("Null control: grey band = range of WT-vs-WT pairs")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
