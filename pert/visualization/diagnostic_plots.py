"""Extended diagnostic figures for a single run (convergence, time courses,
information dashboard, binding-site distance shifts).
"""

from __future__ import annotations

import csv
from typing import Dict, List, Tuple

import numpy as np

from pert.deps import GaussianMixture
from pert.utils.mpl import plt


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average (returns the input when it is shorter than the window)."""
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _pad_dims(x: np.ndarray, n: int) -> np.ndarray:
    """First ``n`` columns of ``x``, zero-padded when ``x`` has fewer."""
    if x.shape[1] >= n:
        return x[:, :n]
    out = np.zeros((x.shape[0], n), dtype=float)
    out[:, : x.shape[1]] = x
    return out


def plot_basin_convergence(
    u_b_wt: np.ndarray,
    u_b_mut: np.ndarray,
    n_basins: int,
    out_png: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative basin occupancies over time for WT and mutant (convergence check).

    Basins come from a GMM with ``n_basins`` components fitted on the pooled
    latents.  Returns the basin labels ``(labels_wt, labels_mut)``.
    """
    pooled = np.vstack([u_b_wt, u_b_mut])
    gmm = GaussianMixture(
        n_components=max(1, n_basins),
        covariance_type="full",
        reg_covar=1e-6,
        random_state=13,
    )
    gmm.fit(pooled)
    labels_wt = gmm.predict(u_b_wt)
    labels_mut = gmm.predict(u_b_mut)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, n_basins))

    for ax, labels, title in [
        (axes[0], labels_wt, "WT cumulative basin occupancy"),
        (axes[1], labels_mut, "MUT cumulative basin occupancy"),
    ]:
        n = len(labels)
        t = np.arange(1, n + 1, dtype=float)
        for basin in range(n_basins):
            indicator = (labels == basin).astype(float)
            cum = np.cumsum(indicator) / t
            ax.plot(t, cum, color=colors[basin], label=f"B{basin}")
        ax.set_xlabel("Frame")
        ax.set_title(title)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Cumulative occupancy")
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("Convergence View: Basin Population Trajectories")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return labels_wt, labels_mut


def plot_latent_timecourse(
    u_b_wt: np.ndarray,
    u_b_mut: np.ndarray,
    out_png: str,
) -> None:
    """Time course of the first three pocket latents of WT and mutant."""
    wt = _pad_dims(u_b_wt, 3)
    mut = _pad_dims(u_b_mut, 3)
    smooth_w = max(3, len(wt) // 25)
    smooth_m = max(3, len(mut) // 25)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex="col")
    for comp in range(3):
        t_w = np.arange(len(wt))
        t_m = np.arange(len(mut))
        axes[comp, 0].plot(t_w, wt[:, comp], color="#1f77b4", alpha=0.25, linewidth=1)
        axes[comp, 0].plot(
            t_w,
            _moving_average(wt[:, comp], smooth_w),
            color="#1f77b4",
            linewidth=2,
        )
        axes[comp, 0].set_ylabel(f"uB[{comp}]")
        axes[comp, 0].grid(alpha=0.25, linewidth=0.5)

        axes[comp, 1].plot(t_m, mut[:, comp], color="#d62728", alpha=0.25, linewidth=1)
        axes[comp, 1].plot(
            t_m,
            _moving_average(mut[:, comp], smooth_m),
            color="#d62728",
            linewidth=2,
        )
        axes[comp, 1].grid(alpha=0.25, linewidth=0.5)

    axes[0, 0].set_title("WT pocket latent components")
    axes[0, 1].set_title("MUT pocket latent components")
    axes[2, 0].set_xlabel("Frame")
    axes[2, 1].set_xlabel("Frame")
    fig.suptitle("Latent Time-Course (raw + smoothed)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_information_dashboard(analysis: Dict, out_png: str) -> None:
    """Conditional TE (both directions) and susceptibility summary for WT and mutant."""
    te_wt = analysis["te_wt_p_to_b_given_l"]
    te_mut = analysis["te_mut_p_to_b_given_l"]
    gamma_wt = analysis["susceptibility_wt"]
    gamma_mut = analysis["susceptibility_mut"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    names = ["P→B|L", "B→P|L"]
    x = np.arange(len(names))
    width = 0.35
    wt_vals = [te_wt["te_driver_to_target"], te_wt["te_target_to_driver"]]
    mut_vals = [te_mut["te_driver_to_target"], te_mut["te_target_to_driver"]]
    axes[0].bar(x - width / 2, wt_vals, width=width, color="#1f77b4", label="WT")
    axes[0].bar(x + width / 2, mut_vals, width=width, color="#d62728", label="MUT")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("TE (nats)")
    axes[0].set_title("Directed predictive dependence")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2, linewidth=0.5, axis="y")

    labels = ["WT p-value", "MUT p-value", "0.05"]
    vals = [te_wt["surrogate_pvalue"], te_mut["surrogate_pvalue"], 0.05]
    colors = ["#1f77b4", "#d62728", "#777777"]
    axes[1].bar(labels, vals, color=colors)
    axes[1].set_ylim(0.0, max(0.2, max(vals) * 1.35))
    axes[1].set_ylabel("Surrogate-test p-value")
    axes[1].set_title("Significance + coupling shifts")
    axes[1].grid(alpha=0.2, linewidth=0.5, axis="y")

    text = (
        f"ΔTE(MUT-WT): {analysis['te_delta_mut_minus_wt']:.3f}\\n"
        f"||Γ||F WT: {gamma_wt['fro_norm']:.3f}\\n"
        f"||Γ||F MUT: {gamma_mut['fro_norm']:.3f}\\n"
        f"Δ||Γ||F: {analysis['susceptibility_delta_fro_norm']:.3f}"
    )
    axes[1].text(
        0.03,
        0.95,
        text,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    fig.suptitle("Information-Flow Dashboard")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _read_binding_csv(path: str) -> List[Dict[str, str]]:
    """Rows of the binding-site table."""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def plot_binding_site_shift(binding_csv: str, out_png: str) -> None:
    """Change of the ligand distance of every binding-site residue (MUT - WT)."""
    rows = _read_binding_csv(binding_csv)
    points = []
    only_wt = []
    only_mut = []
    for row in rows:
        residue = row["residue"]
        w = row["wt_min_dist_ang"].strip()
        m = row["mut_min_dist_ang"].strip()
        d = row["delta_mut_minus_wt"].strip()
        if d:
            points.append((residue, float(d)))
        elif w and not m:
            only_wt.append(residue)
        elif m and not w:
            only_mut.append(residue)

    points.sort(key=lambda x: abs(x[1]), reverse=True)
    if not points:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No shared residues with numeric Δdistance.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        return

    residues = [x[0] for x in points]
    deltas = np.array([x[1] for x in points], dtype=float)
    y = np.arange(len(residues))
    colors = np.where(deltas >= 0.0, "#d62728", "#1f77b4")

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(residues) + 1.5)))
    ax.barh(y, deltas, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(residues)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Δ min distance to ligand (MUT - WT, Å)")
    ax.set_title("Binding-site residue shifts around ligand")
    ax.grid(alpha=0.2, linewidth=0.5, axis="x")

    notes = []
    if only_wt:
        notes.append("WT-only: " + ", ".join(only_wt))
    if only_mut:
        notes.append("MUT-only: " + ", ".join(only_mut))
    if notes:
        ax.text(
            0.02,
            0.02,
            "\n".join(notes),
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
