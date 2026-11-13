"""Stacked-bar figures of ranked residues (per run, fast scan, panel consensus).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

import numpy as np

from pert.utils.mpl import plt


def plot_ranked_residues(rows: Sequence[Dict[str, object]], out_png: str, top_n: int) -> None:
    """Stacked score components of the top MD-ranked residues."""
    top = list(rows[: max(1, int(top_n))])
    if not top:
        return
    top = list(reversed(top))
    labels = [str(row["node_label"]) for row in top]
    y = np.arange(len(top), dtype=float)

    structural = np.asarray([float(row["structural_component"]) for row in top], dtype=float)
    dynamics = np.asarray([float(row["dynamics_component"]) for row in top], dtype=float)
    contacts = np.asarray([float(row["contact_component"]) for row in top], dtype=float)
    te = np.asarray([float(row["te_component"]) for row in top], dtype=float)
    relay = np.asarray([float(row["relay_component"]) for row in top], dtype=float)
    target = np.asarray([float(row["target_coupling_component"]) for row in top], dtype=float)
    total = np.asarray([float(row["total_score"]) for row in top], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.32 * len(top) + 1.5)))
    left = np.zeros_like(y)
    parts = [
        ("structure", structural, "#4C78A8"),
        ("dynamics", dynamics, "#F58518"),
        ("contacts", contacts, "#54A24B"),
        ("TE", te, "#B279A2"),
        ("relay", relay, "#E45756"),
        ("site coupling", target, "#72B7B2"),
    ]
    for name, values, color in parts:
        ax.barh(y, values, left=left, color=color, alpha=0.88, label=name)
        left += values
    ax.plot(total * max(float(np.max(left)), 1.0), y, "ko", markersize=3.5, label="total score")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Normalized component contribution")
    ax.set_title("Top Mutation-Associated Allosteric Residues")
    ax.grid(alpha=0.18, axis="x")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_fast_scores(rows: Sequence[Dict[str, Any]], out_png: str, top_n: int) -> None:
    """Stacked score components of the top fast-scan residues."""
    top = list(rows[: max(1, int(top_n))])
    if not top:
        return
    plot_rows = list(reversed(top))
    labels = [str(row["node_label"]) for row in plot_rows]
    y = np.arange(len(plot_rows), dtype=float)
    parts = [
        ("structure", np.asarray([float(row["structural_component"]) for row in plot_rows]), "#4C78A8"),
        ("contacts", np.asarray([float(row["contact_component"]) for row in plot_rows]), "#54A24B"),
        ("strain", np.asarray([float(row["strain_component"]) for row in plot_rows]), "#F58518"),
        ("path", np.asarray([float(row["path_component"]) for row in plot_rows]), "#B279A2"),
        ("pocket", np.asarray([float(row["pocket_component"]) for row in plot_rows]), "#72B7B2"),
        ("source", np.asarray([float(row["source_component"]) for row in plot_rows]), "#E45756"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.34 * len(plot_rows) + 1.5)))
    left = np.zeros(len(plot_rows), dtype=float)
    for name, values, color in parts:
        ax.barh(y, values, left=left, color=color, alpha=0.88, label=name)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Normalized component contribution")
    ax.set_title("Fast Mutation/Allostery Residue Scan")
    ax.grid(alpha=0.18, axis="x")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_consensus_scenario(
    scenario: str,
    rows: Sequence[Dict[str, Any]],
    out_png: str,
    top_n: int,
) -> None:
    """Mean score with 95% CI of the top consensus residues of one scenario, coloured by replicate support."""
    scenario_rows = [row for row in rows if str(row["scenario"]) == scenario]
    scenario_rows = scenario_rows[: max(1, int(top_n))]
    if not scenario_rows:
        return
    plot_rows = list(reversed(scenario_rows))
    labels = [str(row["node_label"]) for row in plot_rows]
    y = np.arange(len(plot_rows), dtype=float)
    means = np.asarray([float(row["total_score_mean"]) for row in plot_rows], dtype=float)
    low = np.asarray([float(row["total_score_ci95_low"]) for row in plot_rows], dtype=float)
    high = np.asarray([float(row["total_score_ci95_high"]) for row in plot_rows], dtype=float)
    support = np.asarray([float(row["support_top_n_fraction"]) for row in plot_rows], dtype=float)
    xerr = np.vstack([np.maximum(0.0, means - low), np.maximum(0.0, high - means)])

    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.34 * len(plot_rows) + 1.5)))
    colors = plt.cm.viridis(np.clip(support, 0.0, 1.0))
    ax.barh(y, means, xerr=xerr, color=colors, alpha=0.9, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean allostery score across replicas")
    ax.set_title(f"Consensus Allosteric Residues: {scenario}")
    ax.grid(alpha=0.2, axis="x")
    for yi, row in zip(y, plot_rows):
        ax.text(
            float(row["total_score_mean"]) + 0.01,
            yi,
            f"support={float(row['support_top_n_fraction']):.2f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
