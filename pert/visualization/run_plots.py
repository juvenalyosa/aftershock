"""Standard figures for a single WT-vs-MUT run.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from pert.utils.mpl import plt


def plot_basin_ddg(analysis: Dict, out_png: str) -> None:
    """Bar plot of basin ddG with 95% intervals.

    Basins populated in only one system have no defined ddG: they are labelled
    "WT only" / "MUT only" instead of being drawn as bars.

    The interval is drawn where the bootstrap puts it, not as an error bar
    hanging from the bar: for a basin that is barely populated in one system
    the resamples in which it survives are not centred on the estimate, and the
    interval may not contain it.
    """
    basins = analysis["basin_ddg"]["basins"]
    labels = [str(row["basin"]) for row in basins]
    positions = np.arange(len(basins))

    def _values(key: str) -> np.ndarray:
        return np.array([np.nan if row.get(key) is None else row[key] for row in basins], dtype=float)

    ddg = _values("ddg_kcal_mol")
    low = _values("ddg_ci95_low")
    high = _values("ddg_ci95_high")
    has_ci = np.isfinite(low) & np.isfinite(high)

    plt.figure(figsize=(7, 4))
    plt.bar(positions, np.nan_to_num(ddg, nan=0.0), color="#4C78A8", alpha=0.9)
    if np.any(has_ci):
        middle = 0.5 * (low[has_ci] + high[has_ci])
        half_width = 0.5 * np.abs(high[has_ci] - low[has_ci])
        plt.errorbar(positions[has_ci], middle, yerr=half_width, fmt="none", ecolor="black", capsize=4)
    plt.xticks(positions, labels)
    for x, row in enumerate(basins):
        if row.get("ddg_kcal_mol") is None:
            only = "WT only" if float(row.get("p_wt", 0.0)) > 0.0 else "MUT only"
            plt.text(x, 0.0, only, ha="center", va="bottom", fontsize=8, color="#B22222", rotation=90)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Basin")
    plt.ylabel("ΔΔG (kcal/mol)")
    plt.title("Pocket Basin Stabilization Shift (MUT vs WT)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_latent_trajectory(latents_npz: str, out_png: str) -> None:
    """WT and mutant paths in the first two pocket latents."""
    data = np.load(latents_npz)
    wt = np.asarray(data["u_b_wt"], dtype=float)
    mut = np.asarray(data["u_b_mut"], dtype=float)

    def _pad2(x: np.ndarray) -> np.ndarray:
        if x.shape[1] >= 2:
            return x[:, :2]
        return np.column_stack([x[:, 0], np.zeros(len(x))])

    wt2 = _pad2(wt)
    mut2 = _pad2(mut)

    plt.figure(figsize=(7, 6))
    plt.plot(wt2[:, 0], wt2[:, 1], color="#1f77b4", alpha=0.6, label="WT path")
    plt.plot(mut2[:, 0], mut2[:, 1], color="#d62728", alpha=0.6, label="MUT path")
    plt.scatter(wt2[:, 0], wt2[:, 1], color="#1f77b4", s=14)
    plt.scatter(mut2[:, 0], mut2[:, 1], color="#d62728", s=14)

    for arr, color in [(wt2, "#1f77b4"), (mut2, "#d62728")]:
        d = arr[1:] - arr[:-1]
        plt.quiver(
            arr[:-1, 0],
            arr[:-1, 1],
            d[:, 0],
            d[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            color=color,
            alpha=0.35,
        )

    plt.xlabel("Pocket latent 1")
    plt.ylabel("Pocket latent 2")
    plt.title("Pocket Latent Trajectories with Local Flow Vectors")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_te(analysis: Dict, out_png: str) -> None:
    """Forward and reverse conditional TE for WT and mutant."""
    wt = analysis["te_wt_p_to_b_given_l"]
    mut = analysis["te_mut_p_to_b_given_l"]
    names = ["P→B|L", "B→P|L"]
    wt_vals = [wt["te_driver_to_target"], wt["te_target_to_driver"]]
    mut_vals = [mut["te_driver_to_target"], mut["te_target_to_driver"]]

    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(6.8, 4.2))
    plt.bar(x - width / 2, wt_vals, width=width, label="WT", color="#1f77b4")
    plt.bar(x + width / 2, mut_vals, width=width, label="MUT", color="#d62728")
    plt.xticks(x, names)
    plt.ylabel("TE (nats)")
    plt.title("Directed Predictive Dependence (Linear-Gaussian TE)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_susceptibility_quiver(analysis: Dict, out_png: str) -> None:
    """Context -> pocket linear response fields of WT and mutant."""
    gamma_wt = np.asarray(analysis["susceptibility_wt"]["gamma_matrix"], dtype=float)
    gamma_mut = np.asarray(analysis["susceptibility_mut"]["gamma_matrix"], dtype=float)
    g_wt = gamma_wt[:2, :2]
    g_mut = gamma_mut[:2, :2]

    grid = np.linspace(-2.0, 2.0, 11)
    xx, yy = np.meshgrid(grid, grid)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    vec_wt = pts @ g_wt.T
    vec_mut = pts @ g_mut.T

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    axes[0].quiver(pts[:, 0], pts[:, 1], vec_wt[:, 0], vec_wt[:, 1], color="#1f77b4")
    axes[0].set_title("WT susceptibility field")
    axes[1].quiver(pts[:, 0], pts[:, 1], vec_mut[:, 0], vec_mut[:, 1], color="#d62728")
    axes[1].set_title("MUT susceptibility field")
    for ax in axes:
        ax.set_xlabel("Context latent 1")
        ax.set_ylabel("Context latent 2")
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.axvline(0.0, color="black", linewidth=0.6)
    fig.suptitle("Estimated Context→Pocket Linear Response (Γ)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_flow_timeseries(
    time_ps: np.ndarray,
    flow_wt: np.ndarray,
    flow_mut: np.ndarray,
    flow_delta: np.ndarray,
    out_png: str,
) -> None:
    """Windowed mutation -> site flow score for WT, mutant and their difference."""
    time_ns = np.asarray(time_ps, dtype=float) * 1e-3
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    axes[0].plot(time_ns, flow_wt, color="#1f77b4", linewidth=2.0, label="WT flow score")
    axes[0].plot(time_ns, flow_mut, color="#d62728", linewidth=2.0, label="MUT flow score")
    axes[0].set_ylabel("Flow score")
    axes[0].set_title("Mutation→Binding-Site Information-Flow Proxy")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(alpha=0.25, linewidth=0.8)

    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].plot(time_ns, flow_delta, color="#8c564b", linewidth=2.0, label="Δ flow (MUT−WT)")
    axes[1].fill_between(time_ns, 0.0, flow_delta, where=(flow_delta >= 0.0), color="#ff9896", alpha=0.35)
    axes[1].fill_between(time_ns, 0.0, flow_delta, where=(flow_delta < 0.0), color="#9ecae1", alpha=0.35)
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Δ flow")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(alpha=0.25, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
