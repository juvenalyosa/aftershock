"""Normalisation of external mutation-effect tables into the benchmark format.

Accepts CSV/TSV tables with arbitrary column names (ProteinGym, SKEMPI,
ProThermDB, ...), validates or derives mutation specs (including from full
mutant sequences), optionally transforms the measured effect and derives a
binary label.  Output columns are listed in ``CANONICAL_COLUMNS``.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
from typing import Dict, List, Optional, Sequence

from pert.sequence.mutations import MutationError, parse_mutation


CANONICAL_COLUMNS = [
    "name",
    "mutations",
    "wt_pdb",
    "mut_pdb",
    "run_dir",
    "ligand_resname",
    "label",
    "effect_score",
]


# --- table I/O ---------------------------------------------------------------------------


def _sniff_delimiter(path: str) -> str:
    """Delimiter from the extension (.tsv/.tab) or sniffed from the first 4 kB (comma as fallback)."""
    lower = path.lower()
    if lower.endswith(".tsv") or lower.endswith(".tab"):
        return "\t"
    with open(path, "r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return str(dialect.delimiter)
    except csv.Error:
        return ","


def _read_table(path: str) -> List[Dict[str, str]]:
    """Read a delimited table into dict rows."""
    delimiter = _sniff_delimiter(path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Input table has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Input table contains no rows: {path}")
    return rows


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write rows with the canonical columns first, extra columns after."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys: List[str] = []
    for key in CANONICAL_COLUMNS:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _first_existing(row: Dict[str, str], candidates: Sequence[str]) -> str:
    """First non-empty value among candidate column names."""
    for key in candidates:
        if key in row and str(row.get(key, "")).strip():
            return str(row[key]).strip()
    return ""


def _as_float(value: object) -> Optional[float]:
    """Finite float or None."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


# --- mutation specs --------------------------------------------------------------------------


def _parse_tokens(value: object, separators: str = ",;") -> List[str]:
    """Split a mutation cell on any of the given separator characters."""
    out: List[str] = []
    text = str(value or "")
    for sep in separators:
        text = text.replace(sep, ",")
    for chunk in text.split(","):
        token = chunk.strip()
        if token:
            out.append(token)
    return out


def _validate_mutations(tokens: Sequence[str]) -> List[str]:
    """Syntax-check mutation tokens; returns the error messages (empty list when all are valid)."""
    errors: List[str] = []
    for token in tokens:
        try:
            parse_mutation(token)
        except MutationError as exc:
            errors.append(str(exc))
    return errors


def _aa_at(sequence: str, start_pos: int, pos: int) -> str:
    """Reference residue at a 1-based (``start_pos``-based) position."""
    return sequence[pos - start_pos]


def _range_label(sequence: str, start_pos: int, start: int, end: int) -> str:
    """Label of a reference range, e.g. ``E12`` or ``E12_K14``."""
    if start == end:
        return f"{_aa_at(sequence, start_pos, start)}{start}"
    return f"{_aa_at(sequence, start_pos, start)}{start}_{_aa_at(sequence, start_pos, end)}{end}"


def _find_upstream_dup(sequence: str, start_pos: int, left_pos: int, inserted: str) -> Optional[str]:
    """If ``inserted`` repeats the residues just before the insertion point, return the DUP spec."""
    dup_len = len(inserted)
    dup_start = left_pos - dup_len + 1
    if dup_start < start_pos:
        return None
    segment = sequence[dup_start - start_pos : left_pos - start_pos + 1]
    if segment != inserted:
        return None
    return f"{_range_label(sequence, start_pos, dup_start, left_pos)}dup"


def _derive_mutations_from_sequences(reference_sequence: str, mutated_sequence: str, start_pos: int) -> List[str]:
    """Infer mutation specs (SNP/DEL/INS/DUP/DELINS) from a reference and a mutant sequence."""
    reference_sequence = str(reference_sequence).strip().upper()
    mutated_sequence = str(mutated_sequence).strip().upper()
    if not reference_sequence or not mutated_sequence:
        raise ValueError("Reference and mutant sequences are required for sequence-derived mutation specs.")
    max_ref_pos = start_pos + len(reference_sequence) - 1
    mutations: List[str] = []
    matcher = difflib.SequenceMatcher(a=reference_sequence, b=mutated_sequence, autojunk=False)
    for tag, ref_i1, ref_i2, mut_i1, mut_i2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        ref_text = reference_sequence[ref_i1:ref_i2]
        mut_text = mutated_sequence[mut_i1:mut_i2]

        if tag == "insert":
            left_pos = start_pos + ref_i1 - 1
            right_pos = left_pos + 1
            dup = _find_upstream_dup(reference_sequence, start_pos, left_pos, mut_text)
            if dup:
                mutations.append(dup)
            elif left_pos < start_pos:
                first = start_pos
                aa = _aa_at(reference_sequence, start_pos, first)
                mutations.append(f"{aa}{first}delins{mut_text}{aa}")
            elif right_pos > max_ref_pos:
                last = max_ref_pos
                aa = _aa_at(reference_sequence, start_pos, last)
                mutations.append(f"{aa}{last}delins{aa}{mut_text}")
            else:
                left_aa = _aa_at(reference_sequence, start_pos, left_pos)
                right_aa = _aa_at(reference_sequence, start_pos, right_pos)
                mutations.append(f"{left_aa}{left_pos}_{right_aa}{right_pos}ins{mut_text}")
            continue

        start = start_pos + ref_i1
        end = start_pos + ref_i2 - 1

        if tag == "delete":
            mutations.append(f"{_range_label(reference_sequence, start_pos, start, end)}del")
            continue

        if tag == "replace":
            if len(ref_text) == 1 and len(mut_text) == 1:
                mutations.append(f"{ref_text}{start}{mut_text}")
            else:
                mutations.append(f"{_range_label(reference_sequence, start_pos, start, end)}delins{mut_text}")
            continue

        raise ValueError(f"Unexpected sequence diff opcode: {tag}")

    return mutations


# --- row-level transforms -----------------------------------------------------------------------


def _format_pattern(pattern: str, row: Dict[str, str], name: str) -> str:
    """Fill a ``{column}``-style path pattern with values of the row."""
    values = {str(k): str(v).strip() for k, v in row.items()}
    values.setdefault("name", name)
    try:
        return pattern.format(**values)
    except KeyError as exc:
        raise ValueError(f"Pattern {pattern!r} references missing column {exc.args[0]!r}.") from exc


def _derive_label(effect: Optional[float], threshold: float, direction: str) -> str:
    """Binary label from a continuous effect and a threshold rule."""
    if effect is None:
        return ""
    direction = direction.lower()
    if direction == "ge":
        return "1" if effect >= threshold else "0"
    if direction == "le":
        return "1" if effect <= threshold else "0"
    if direction == "abs-ge":
        return "1" if abs(effect) >= threshold else "0"
    raise ValueError("--label-direction must be ge, le, or abs-ge.")


def _transform_effect(value: Optional[float], mode: str) -> Optional[float]:
    """Apply the requested transform (identity, abs, neg...) to the measured effect."""
    if value is None:
        return None
    mode = mode.lower()
    if mode == "raw":
        return value
    if mode == "abs":
        return abs(value)
    if mode == "neg":
        return -value
    raise ValueError("--effect-mode must be raw, abs, or neg.")


def _parse_renames(values: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse repeated ``old:new`` column renames."""
    renames: Dict[str, str] = {}
    for value in values or []:
        if ":" not in value:
            raise ValueError("--rename-column must use SOURCE:DEST syntax.")
        src, dest = value.split(":", 1)
        src = src.strip()
        dest = dest.strip()
        if not src or not dest:
            raise ValueError(f"Invalid --rename-column value: {value!r}")
        renames[src] = dest
    return renames


def normalize_rows(args: argparse.Namespace) -> Dict[str, object]:
    """Convert the input table to canonical rows, write the CSV and the JSON report, return the report."""
    input_rows = _read_table(args.input)
    out_rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    renames = _parse_renames(args.rename_column)

    for idx, row in enumerate(input_rows, start=1):
        name = (
            str(row.get(args.name_column, "")).strip()
            if args.name_column
            else _first_existing(row, ["name", "variant_id", "id", "mutation_id"])
        )
        if not name:
            name = f"variant_{idx}"

        mutation_text = (
            str(row.get(args.mutations_column, "")).strip()
            if args.mutations_column
            else _first_existing(row, ["mutations", "mutation_set", "mutation", "variant", "mutant"])
        )
        mutations = _parse_tokens(mutation_text, args.mutation_separators)
        if not mutations and args.mutated_sequence_column:
            mutated_sequence = str(row.get(args.mutated_sequence_column, "")).strip()
            reference_sequence = (
                str(row.get(args.reference_sequence_column, "")).strip()
                if args.reference_sequence_column
                else args.reference_sequence
            )
            try:
                mutations = _derive_mutations_from_sequences(
                    reference_sequence=reference_sequence,
                    mutated_sequence=mutated_sequence,
                    start_pos=args.sequence_start,
                )
            except ValueError as exc:
                skipped.append({"row": idx, "name": name, "reason": str(exc)})
                continue
        if not mutations:
            skipped.append({"row": idx, "name": name, "reason": "missing mutations"})
            continue
        mutation_errors = _validate_mutations(mutations)
        if mutation_errors and not args.allow_invalid_mutations:
            skipped.append({"row": idx, "name": name, "reason": "; ".join(mutation_errors)})
            continue

        effect_value = _as_float(row.get(args.effect_source_column, "")) if args.effect_source_column else _as_float(row.get("effect_score", ""))
        effect_value = _transform_effect(effect_value, args.effect_mode) if effect_value is not None else None
        label = str(row.get(args.label_source_column, "")).strip() if args.label_source_column else str(row.get("label", "")).strip()
        if not label and args.derive_label_from_effect:
            label = _derive_label(effect_value, args.label_threshold, args.label_direction)

        wt_pdb = str(row.get(args.wt_pdb_column, "")).strip() if args.wt_pdb_column else str(row.get("wt_pdb", "")).strip()
        mut_pdb = str(row.get(args.mut_pdb_column, "")).strip() if args.mut_pdb_column else str(row.get("mut_pdb", "")).strip()
        run_dir = str(row.get(args.run_dir_column, "")).strip() if args.run_dir_column else str(row.get("run_dir", "")).strip()
        if args.wt_pdb_pattern and not wt_pdb:
            wt_pdb = _format_pattern(args.wt_pdb_pattern, row, name)
        if args.mut_pdb_pattern and not mut_pdb:
            mut_pdb = _format_pattern(args.mut_pdb_pattern, row, name)
        if args.run_dir_pattern and not run_dir:
            run_dir = _format_pattern(args.run_dir_pattern, row, name)

        if args.drop_missing_structure and not run_dir and not (wt_pdb and mut_pdb):
            skipped.append({"row": idx, "name": name, "reason": "missing structure paths"})
            continue

        out: Dict[str, object] = {
            "name": name,
            "mutations": ",".join(mutations),
            "wt_pdb": wt_pdb,
            "mut_pdb": mut_pdb,
            "run_dir": run_dir,
            "ligand_resname": str(row.get(args.ligand_resname_column, "")).strip()
            if args.ligand_resname_column
            else str(row.get("ligand_resname", args.default_ligand_resname)).strip(),
            "label": label,
            "effect_score": "" if effect_value is None else float(effect_value),
        }
        for src, dest in renames.items():
            out[dest] = str(row.get(src, "")).strip()
        for key in args.keep_column or []:
            out[key] = str(row.get(key, "")).strip()
        out_rows.append(out)

        if args.limit and len(out_rows) >= args.limit:
            break

    if not out_rows:
        raise ValueError("No rows survived normalization.")
    _write_csv(args.output, out_rows)
    report = {
        "input": args.input,
        "output": args.output,
        "rows_in": len(input_rows),
        "rows_out": len(out_rows),
        "rows_skipped": len(skipped),
        "skipped": skipped[:50],
        "note": "Mutation validation checks syntax only; sequence/PDB compatibility is checked during benchmark scoring.",
    }
    report_path = args.report_json or os.path.splitext(args.output)[0] + "_normalization_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report["report_json"] = report_path
    return report
