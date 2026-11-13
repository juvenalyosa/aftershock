"""Benchmark layer: metrics and sequence-derived mutation specs."""

from __future__ import annotations

import pytest

from pert.benchmark.metrics import average_precision, best_binary_threshold, pearson, roc_auc, spearman
from pert.benchmark.normalize import _derive_mutations_from_sequences
from pert.sequence.mutations import mutate_sequence

ZNF = "YKCGLCERSFVEKSALSRHQRVHKN"


def test_ranking_metrics_on_a_perfect_predictor():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(labels, scores) == pytest.approx(1.0)
    assert average_precision(labels, scores) == pytest.approx(1.0)
    best = best_binary_threshold(labels, scores)
    assert best["balanced_accuracy"] == pytest.approx(1.0)
    assert 0.2 < best["threshold"] <= 0.8


def test_metrics_are_undefined_without_both_classes():
    assert roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None
    assert best_binary_threshold([0, 0], [0.1, 0.2]) is None


def test_correlations():
    x = [1.0, 2.0, 3.0, 4.0]
    assert pearson(x, [2.0, 4.0, 6.0, 8.0]) == pytest.approx(1.0)
    assert spearman(x, [1.0, 8.0, 27.0, 64.0]) == pytest.approx(1.0)  # monotonic, not linear
    assert pearson(x, [5.0, 5.0, 5.0, 5.0]) is None  # constant column


@pytest.mark.parametrize(
    "mutations",
    [
        ["K2R"],
        ["10_12del"],
        ["R8_S9insA"],
        ["Q20dup"],
        ["K2R", "10_11delinsG", "Q20dup"],
    ],
)
def test_derived_mutations_reproduce_the_mutant_sequence(mutations):
    mutant = mutate_sequence(ZNF, mutations, start_pos=1)
    derived = _derive_mutations_from_sequences(ZNF, mutant, start_pos=1)
    assert derived, "a different sequence must yield at least one mutation spec"
    assert mutate_sequence(ZNF, derived, start_pos=1) == mutant
