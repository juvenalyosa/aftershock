"""Sequence layer: mutation application, reference mapping and alignment."""

from __future__ import annotations

import pytest

from pert.sequence.alignment import align_index_pairs, needleman_wunsch_affine
from pert.sequence.mutations import MutationError, mutate_sequence, mutation_reference_map

# Zinc finger 1ZNF, the toy system of the repository.
ZNF = "YKCGLCERSFVEKSALSRHQRVHKN"


@pytest.mark.parametrize(
    "mutations, expected",
    [
        (["K2R"], "YRCGLCERSFVEKSALSRHQRVHKN"),
        (["R8_S9insA"], "YKCGLCERASFVEKSALSRHQRVHKN"),
        (["10_11delinsG"], "YKCGLCERSGEKSALSRHQRVHKN"),
        (["10_11insdelG"], "YKCGLCERSGEKSALSRHQRVHKN"),  # alias spelling
        (["Q20dup"], "YKCGLCERSFVEKSALSRHQQRVHKN"),
        (["F10del"], "YKCGLCERSVEKSALSRHQRVHKN"),
        (["10_12del"], "YKCGLCERSKSALSRHQRVHKN"),
    ],
)
def test_single_mutation_classes(mutations, expected):
    assert mutate_sequence(ZNF, mutations, start_pos=1) == expected


def test_multiple_mutations_use_reference_coordinates():
    # Q20dup must still refer to reference position 20 after the upstream indel.
    mutated = mutate_sequence(ZNF, ["K2R", "10_11delinsG", "Q20dup"], start_pos=1)
    assert mutated == "YRCGLCERSGEKSALSRHQQRVHKN"


def test_start_position_offset():
    assert mutate_sequence("AKST", ["K699R"], start_pos=698) == "ARST"


@pytest.mark.parametrize(
    "mutations",
    [
        ["A2R"],  # wrong reference residue
        ["K99R"],  # position outside the sequence
        ["K2R", "2_3del"],  # overlapping edits
        ["R8_F10insA"],  # insertion between non-adjacent residues
        ["not-a-mutation"],
    ],
)
def test_invalid_mutations_raise(mutations):
    with pytest.raises(MutationError):
        mutate_sequence(ZNF, mutations, start_pos=1)


def test_reference_map_is_consistent_with_mutate_sequence():
    mutations = ["K2R", "10_11delinsG", "Q20dup"]
    ref_map = mutation_reference_map(ZNF, mutations, start_pos=1)
    assert ref_map["mutated_sequence"] == mutate_sequence(ZNF, mutations, start_pos=1)
    assert ref_map["deleted_ref_positions"] == [10, 11]
    assert len(ref_map["residue_map"]) == len(ref_map["mutated_sequence"])
    sources = [row for row in ref_map["residue_map"] if row["is_source"]]
    assert {row["status"] for row in sources} == {"substitution", "delins", "duplication"}
    # unchanged residues keep their own reference position and amino acid
    for row in ref_map["residue_map"]:
        if row["status"] == "unchanged":
            assert ZNF[row["ref_pos"] - 1] == row["mut_aa"]


def test_alignment_places_a_single_gap_for_a_deletion():
    mutated = mutate_sequence(ZNF, ["10_12del"], start_pos=1)
    aligned_ref, aligned_mut = needleman_wunsch_affine(ZNF, mutated)
    assert aligned_ref == ZNF
    assert aligned_mut.replace("-", "") == mutated
    assert aligned_mut.count("-") == 3
    assert "---" in aligned_mut  # affine gaps keep the deletion contiguous


def test_align_index_pairs_skips_gaps():
    mutated = mutate_sequence(ZNF, ["R8_S9insA"], start_pos=1)
    pairs = align_index_pairs(ZNF, mutated)
    assert len(pairs) == len(ZNF)
    assert all(ZNF[i] == mutated[j] for i, j in pairs)
