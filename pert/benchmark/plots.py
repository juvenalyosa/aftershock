"""Benchmark summary figure.
"""

from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np

from pert.benchmark.metrics import binary_label_column, numeric_column
from pert.utils.mpl import plt


def plot_benchmark(
    rows: Sequence[Dict[str, object]],
    output_png: str,
    label_column: str,
    effect_column: str,
) -> None:
    """Score distribution and score-vs-label/effect scatter of a benchmark."""
    scores = np.asarray(numeric_column(rows, "global_fast_allostery_score"), dtype=float)
    labels = binary_label_column(rows, label_column) if label_column else None
    effects = np.asarray(numeric_column(rows, effect_column), dtype=float) if effect_column else np.asarray([])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].hist(scores[np.isfinite(scores)], bins=min(12, max(3, len(scores))), color="#4C78A8", alpha=0.85)
    axes[0].set_title("Fast Score Distribution")
    axes[0].set_xlabel("global_fast_allostery_score")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.2, axis="y")

    if labels is not None:
        jitter = np.linspace(-0.04, 0.04, len(scores)) if len(scores) else np.asarray([])
        axes[1].scatter(np.asarray(labels, dtype=float) + jitter, scores, color="#E45756", edgecolor="black", alpha=0.85)
        axes[1].set_xticks([0, 1])
        axes[1].set_xlabel(label_column)
        axes[1].set_ylabel("global_fast_allostery_score")
        axes[1].set_title("Score vs Binary Label")
    elif effects.size:
        mask = np.isfinite(effects) & np.isfinite(scores)
        axes[1].scatter(effects[mask], scores[mask], color="#E45756", edgecolor="black", alpha=0.85)
        axes[1].set_xlabel(effect_column)
        axes[1].set_ylabel("global_fast_allostery_score")
        axes[1].set_title("Score vs Measured Effect")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No label/effect column supplied", ha="center", va="center")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
