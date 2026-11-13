"""Protein sequence mutator (the *perturbation* definition).

Applies HGVS-like protein mutations to a sequence.  Supported operations:

==============  =========================  ==================================
kind            example                    meaning
==============  =========================  ==================================
substitution    ``K2R``                    Lys2 -> Arg
deletion        ``A10del``, ``10_12del``   remove one residue / a range
insertion       ``R8_S9insA``              insert between adjacent residues
deletion-ins.   ``10_11delinsG``           replace a range (alias ``insdel``)
duplication     ``Q20dup``, ``10_12dup``   repeat a residue / a range
==============  =========================  ==================================

All positions in a multi-mutation list refer to the *reference* sequence
(they are not shifted by earlier indels), overlapping edits are rejected, and
any mismatch with the reference raises ``MutationError``.

``mutation_reference_map`` additionally returns, for every residue of the
mutant, which reference position it came from.  That map is what lets the
structure-level scanners handle insertions/deletions without guessing.

Original author: El_Dryosa, version 1.0.
"""

from __future__ import annotations

import re


class MutationError(Exception):
    """
    Raised when a requested mutation cannot be parsed or does not
    match the reference sequence (non-existent position, AA mismatch, etc.).

    This is intentionally a normal Python exception rather than an
    immediate program exit. Library / package code should *raise*
    MutationError and let the top-level application decide whether to:

        - catch it and handle/report the problem, or
        - not catch it, in which case the process will stop with a traceback.

    Example (recommended in a CLI script):

        from pert.sequence.mutations import mutate_sequence, MutationError

        try:
            mutated = mutate_sequence(seq, mutations, start_pos)
        except MutationError as exc:
            print(f"ERROR: {exc}")
            raise SystemExit(1)
    """
    pass


# --- sequence <-> {position: residue} dictionary -------------------------------------


def convert_to_dict(sequence, start_pos):
    """
    Converts a sequence string to a dictionary mapping position numbers to amino acids.
    This data structure is ideal for mutations because it handles insertions and
    deletions without needing to manually re-calculate list indices.
    
    Example: "AGK", start_pos=1 -> {1: 'A', 2: 'G', 3: 'K'}
    """
    return {i + start_pos: aa for i, aa in enumerate(sequence)}


def convert_to_sequence(aa_dict):
    """
    Converts an amino acid dictionary back to a continuous sequence string.
    It correctly handles any gaps in position numbers by creating a contiguous
    string from the minimum to the maximum position.
    """
    if not aa_dict:
        return ""
    # Determine the full range of the sequence after mutations.
    min_pos = min(aa_dict.keys())
    max_pos = max(aa_dict.keys())
    # Rebuild the string, getting the amino acid for each position or an empty string if a position was deleted.
    return ''.join(aa_dict.get(i, '') for i in range(min_pos, max_pos + 1))


# --- parsing of mutation specs ---------------------------------------------------------


def _normalize_mutation_token(mutation):
    """Return ``(raw, normalised)`` where the normalised token is upper-case and uses DELINS for INSDEL."""
    raw_mutation = str(mutation).strip()
    if not raw_mutation:
        raise MutationError("Empty mutation token.")
    return raw_mutation, raw_mutation.upper().replace("INSDEL", "DELINS")


def parse_mutation(mutation):
    """Parse one mutation spec into an operation dict.

    Returns a dict with ``kind`` in {snp, del, delins, ins, dup}, the reference
    interval ``start``/``end`` and the kind-specific fields (new residue, inserted
    sequence, flanking residues to validate...).  Raises ``MutationError`` when the
    token matches no supported syntax.
    """
    raw_mutation, normalized = _normalize_mutation_token(mutation)

    snp_match = re.match(r'^([A-Z])(\d+)([A-Z])$', normalized)
    if snp_match:
        old_aa, pos, new_aa = snp_match.groups()
        pos = int(pos)
        return {
            "kind": "snp",
            "raw": raw_mutation,
            "start": pos,
            "end": pos,
            "old_aa": old_aa,
            "new_aa": new_aa,
        }

    range_del_match = re.match(r'^([A-Z]?)(\d+)_([A-Z]?)(\d+)DEL$', normalized)
    if range_del_match:
        start_aa, start_pos, end_aa, end_pos = range_del_match.groups()
        return {
            "kind": "del",
            "raw": raw_mutation,
            "start": int(start_pos),
            "end": int(end_pos),
            "start_aa": start_aa,
            "end_aa": end_aa,
        }
    single_del_match = re.match(r'^([A-Z]?)(\d+)DEL$', normalized)
    if single_del_match:
        start_aa, start_pos = single_del_match.groups()
        start_pos = int(start_pos)
        return {
            "kind": "del",
            "raw": raw_mutation,
            "start": start_pos,
            "end": start_pos,
            "start_aa": start_aa,
            "end_aa": start_aa,
        }

    delins_match = re.match(r'^([A-Z]?)(\d+)(?:_([A-Z]?)(\d+))?DELINS([A-Z]+)$', normalized)
    if delins_match:
        start_aa, start_pos, end_aa, end_pos, insertion = delins_match.groups()
        start_pos = int(start_pos)
        end_pos = int(end_pos) if end_pos else start_pos
        return {
            "kind": "delins",
            "raw": raw_mutation,
            "start": start_pos,
            "end": end_pos,
            "start_aa": start_aa,
            "end_aa": end_aa or start_aa,
            "insertion": insertion,
        }

    ins_match = re.match(r'^([A-Z]?)(\d+)_([A-Z]?)(\d+)INS([A-Z]+)$', normalized)
    if ins_match:
        left_aa, left_pos, right_aa, right_pos, insertion = ins_match.groups()
        return {
            "kind": "ins",
            "raw": raw_mutation,
            "start": int(left_pos),
            "end": int(left_pos),
            "right_pos": int(right_pos),
            "left_aa": left_aa,
            "right_aa": right_aa,
            "insertion": insertion,
        }

    range_dup_match = re.match(r'^([A-Z]?)(\d+)_([A-Z]?)(\d+)DUP$', normalized)
    if range_dup_match:
        start_aa, start_pos, end_aa, end_pos = range_dup_match.groups()
        return {
            "kind": "dup",
            "raw": raw_mutation,
            "start": int(start_pos),
            "end": int(end_pos),
            "start_aa": start_aa,
            "end_aa": end_aa,
        }
    single_dup_match = re.match(r'^([A-Z]?)(\d+)DUP$', normalized)
    if single_dup_match:
        start_aa, start_pos = single_dup_match.groups()
        start_pos = int(start_pos)
        return {
            "kind": "dup",
            "raw": raw_mutation,
            "start": start_pos,
            "end": start_pos,
            "start_aa": start_aa,
            "end_aa": start_aa,
        }

    raise MutationError(f"Could not parse mutation: '{raw_mutation}'.")


# --- validation against the reference sequence -------------------------------------------


def _validate_range(aa_dict, start, end, raw_mutation):
    """Check that ``start..end`` is ordered and that every position exists in the reference."""
    if end < start:
        raise MutationError(f"Mutation {raw_mutation} has an invalid range ({start}>{end}).")
    missing = [str(pos) for pos in range(start, end + 1) if pos not in aa_dict]
    if missing:
        raise MutationError(
            f"Mutation {raw_mutation} references missing position(s): {', '.join(missing)}."
        )


def _validate_residue(aa_dict, pos, expected_aa, raw_mutation, label):
    """Check that the reference residue at ``pos`` is the one named in the mutation (if named)."""
    if expected_aa and aa_dict.get(pos) != expected_aa:
        raise MutationError(
            f"Mutation {raw_mutation} mismatch at {label} position {pos}: "
            f"found '{aa_dict.get(pos)}' but expected '{expected_aa}'."
        )


def _validate_operation(op, aa_dict):
    """Validate one parsed operation against the reference sequence."""
    kind = op["kind"]
    raw = op["raw"]
    if kind == "ins":
        left_pos = op["start"]
        right_pos = op["right_pos"]
        if right_pos != left_pos + 1:
            raise MutationError(f"Insertion {raw} must be between adjacent reference positions.")
        _validate_range(aa_dict, left_pos, left_pos, raw)
        _validate_range(aa_dict, right_pos, right_pos, raw)
        _validate_residue(aa_dict, left_pos, op.get("left_aa", ""), raw, "left")
        _validate_residue(aa_dict, right_pos, op.get("right_aa", ""), raw, "right")
        return

    _validate_range(aa_dict, op["start"], op["end"], raw)
    if kind == "snp":
        _validate_residue(aa_dict, op["start"], op["old_aa"], raw, "substitution")
    else:
        _validate_residue(aa_dict, op["start"], op.get("start_aa", ""), raw, "start")
        _validate_residue(aa_dict, op["end"], op.get("end_aa", ""), raw, "end")


def _prepare_operations(ops):
    """Index operations by reference position and reject conflicting ones.

    Returns three dicts keyed by reference position: replacements (snp/del/delins)
    by their start, insertions by the residue they follow, duplications by the
    last residue of the duplicated range.
    """
    replacement_by_start = {}
    insertion_after = {}
    duplication_after = {}
    occupied_by = {}
    deleted_or_replaced_positions = set()

    for op in ops:
        kind = op["kind"]
        if kind == "ins":
            insertion_after.setdefault(op["start"], []).append(op)
            continue

        interval = set(range(op["start"], op["end"] + 1))
        overlap = sorted(pos for pos in interval if pos in occupied_by)
        if overlap:
            first = overlap[0]
            raise MutationError(
                f"Mutation {op['raw']} overlaps mutation {occupied_by[first]} at reference position {first}."
            )
        for pos in interval:
            occupied_by[pos] = op["raw"]

        if kind == "dup":
            duplication_after.setdefault(op["end"], []).append(op)
        else:
            replacement_by_start[op["start"]] = op
            if kind in {"del", "delins"}:
                deleted_or_replaced_positions.update(interval)

    for pos, insertions in insertion_after.items():
        if pos in deleted_or_replaced_positions:
            names = ", ".join(op["raw"] for op in insertions)
            raise MutationError(f"Insertion after deleted/replaced position {pos} is ambiguous: {names}.")

    return replacement_by_start, insertion_after, duplication_after


# --- applying mutations ----------------------------------------------------------------------


def mutate_sequence(sequence, mutations, start_pos=1):
    """
    Applies mutations to a protein sequence using reference coordinates.

    Args:
        sequence (str): The original amino acid sequence.
        mutations (list): A list of mutations in a simplified HGVS-like format.
                          Supported formats:
                          - Substitution: "A123G"
                          - Deletion: "A10del", "10_12del"
                          - Insertion: "A10_B11insGGG"
                          - Deletion-Insertion: "A10delinsGGG", "10_12delinsGGG"
                            ("insdel" is accepted as an alias)
                          - Duplication: "A10dup", "10_12dup"
                          Operation names are case-insensitive.
                          In a multi-mutation list, positions are interpreted
                          against the original/reference sequence rather than
                          shifted after each earlier indel.
        start_pos (int): The starting position number of the sequence (e.g., 1 for 1-based indexing).

    Returns:
        str: The mutated sequence.
    """
    sequence = str(sequence).strip().upper()
    aa_dict = convert_to_dict(sequence, start_pos)
    if not mutations:
        return sequence

    ops = [parse_mutation(mutation) for mutation in mutations]
    for op in ops:
        _validate_operation(op, aa_dict)
    replacement_by_start, insertion_after, duplication_after = _prepare_operations(ops)

    result = []
    min_pos = start_pos
    max_pos = start_pos + len(sequence) - 1
    pos = min_pos
    while pos <= max_pos:
        op = replacement_by_start.get(pos)
        emitted_reference_pos = None
        if op and op["kind"] == "snp":
            result.append(op["new_aa"])
            emitted_reference_pos = pos
            pos += 1
        elif op and op["kind"] == "del":
            pos = op["end"] + 1
            continue
        elif op and op["kind"] == "delins":
            result.extend(op["insertion"])
            pos = op["end"] + 1
            continue
        else:
            result.append(aa_dict[pos])
            emitted_reference_pos = pos
            pos += 1

        for insertion in insertion_after.get(emitted_reference_pos, []):
            result.extend(insertion["insertion"])
        for duplication in duplication_after.get(emitted_reference_pos, []):
            result.extend(aa_dict[i] for i in range(duplication["start"], duplication["end"] + 1))

    return "".join(result)


def mutation_reference_map(sequence, mutations, start_pos=1):
    """
    Return the mutated sequence plus per-mutant-residue reference mapping.

    This is used by structure-level scanners so compound indels/duplications
    are interpreted by reference coordinates instead of by ambiguous sequence
    realignment alone.
    """
    sequence = str(sequence).strip().upper()
    aa_dict = convert_to_dict(sequence, start_pos)
    ops = [parse_mutation(mutation) for mutation in mutations]
    for op in ops:
        _validate_operation(op, aa_dict)
    replacement_by_start, insertion_after, duplication_after = _prepare_operations(ops)

    rows = []
    deleted_ref_positions = []
    min_pos = start_pos
    max_pos = start_pos + len(sequence) - 1
    pos = min_pos
    while pos <= max_pos:
        op = replacement_by_start.get(pos)
        emitted_reference_pos = None
        if op and op["kind"] == "snp":
            rows.append(
                {
                    "mut_aa": op["new_aa"],
                    "ref_pos": pos,
                    "ref_span": [pos, pos],
                    "status": "substitution",
                    "is_source": True,
                    "mutation": op["raw"],
                }
            )
            emitted_reference_pos = pos
            pos += 1
        elif op and op["kind"] == "del":
            deleted_ref_positions.extend(range(op["start"], op["end"] + 1))
            pos = op["end"] + 1
            continue
        elif op and op["kind"] == "delins":
            deleted_ref_positions.extend(range(op["start"], op["end"] + 1))
            for i, aa in enumerate(op["insertion"]):
                rows.append(
                    {
                        "mut_aa": aa,
                        "ref_pos": op["start"] if i == 0 else None,
                        "ref_span": [op["start"], op["end"]],
                        "status": "delins",
                        "is_source": True,
                        "mutation": op["raw"],
                    }
                )
            pos = op["end"] + 1
            continue
        else:
            rows.append(
                {
                    "mut_aa": aa_dict[pos],
                    "ref_pos": pos,
                    "ref_span": [pos, pos],
                    "status": "unchanged",
                    "is_source": False,
                    "mutation": "",
                }
            )
            emitted_reference_pos = pos
            pos += 1

        for insertion in insertion_after.get(emitted_reference_pos, []):
            for aa in insertion["insertion"]:
                rows.append(
                    {
                        "mut_aa": aa,
                        "ref_pos": None,
                        "ref_span": [insertion["start"], insertion["right_pos"]],
                        "status": "insertion",
                        "is_source": True,
                        "mutation": insertion["raw"],
                    }
                )
        for duplication in duplication_after.get(emitted_reference_pos, []):
            for ref_pos in range(duplication["start"], duplication["end"] + 1):
                rows.append(
                    {
                        "mut_aa": aa_dict[ref_pos],
                        "ref_pos": ref_pos,
                        "ref_span": [duplication["start"], duplication["end"]],
                        "status": "duplication",
                        "is_source": True,
                        "mutation": duplication["raw"],
                    }
                )

    return {
        "mutated_sequence": "".join(row["mut_aa"] for row in rows),
        "residue_map": rows,
        "deleted_ref_positions": deleted_ref_positions,
    }
