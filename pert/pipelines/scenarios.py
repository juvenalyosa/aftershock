"""Mutation scenarios of a panel: where they come from and how they are checked.

A *scenario* is a named set of mutations, e.g. ``("ex19del", ["L747_A750del"])``.
Both panels (MD and fast) take their scenarios from

* ``NAME:MUT1,MUT2`` strings on the command line,
* JSON files (``{"name": ["K2R"]}``, a list of ``{"name", "mutations"}``
  objects, or either under a ``"mutation_sets"`` key),
* CSV files with ``name`` and ``mutations`` columns,
* and, when nothing is given or when asked for, the built-in toy panel on the
  zinc finger 1ZNF (``MUTATION_SCENARIOS``).

The scenario name becomes a directory name and the handle of the scenario in
every table, so it must be a simple identifier.

``scenario_seed_offsets`` derives the random seeds of a scenario from its
*name*, not from its position in the list: adding, removing or filtering
scenarios never changes the seeds (and therefore the identity) of the others,
which is what lets a panel be extended and resumed.
"""

from __future__ import annotations

import csv
import json
import re
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pert.sequence.mutations import MutationError, mutate_sequence
from pert.utils.io import parse_csv_tokens

ScenarioDef = Tuple[str, List[str]]

# Built-in toy panel on the zinc finger 1ZNF: one example of each mutation class.
MUTATION_SCENARIOS: List[ScenarioDef] = [
    ("snp_k2r", ["K2R"]),
    ("ins_r8_s9insa", ["R8_S9insA"]),
    ("delins_10_11_g", ["10_11delinsG"]),
    ("dup_q20", ["Q20dup"]),
]

_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Seeds are spread over this many slots per scenario name.
_SEED_SLOTS = 100000


# --- reading scenario definitions ---------------------------------------------------------


def _split_mutations(value: str) -> List[str]:
    """Split a mutation list on commas or semicolons."""
    tokens: List[str] = []
    for chunk in str(value).replace(";", ",").split(","):
        token = chunk.strip()
        if token:
            tokens.append(token)
    return tokens


def parse_mutation_set(value: str) -> ScenarioDef:
    """Parse ``NAME:MUT1,MUT2`` into ``(name, [mutations])``."""
    if ":" not in value:
        raise ValueError(
            "--mutation-set must use NAME:MUT1,MUT2 syntax, for example "
            "egfr_ex19del:L747_A750del,K754R"
        )
    name, mutation_text = value.split(":", 1)
    name = name.strip()
    mutations = _split_mutations(mutation_text)
    if not name or not mutations:
        raise ValueError(f"Invalid mutation set: {value!r}")
    return name, mutations


def _load_json_mutation_sets(payload: Any) -> List[ScenarioDef]:
    """Mutation sets from a JSON payload (dict name -> mutations, or list of objects)."""
    if isinstance(payload, dict) and "mutation_sets" in payload:
        payload = payload["mutation_sets"]
    if isinstance(payload, dict):
        out: List[ScenarioDef] = []
        for name, mutations in payload.items():
            if isinstance(mutations, str):
                out.append((str(name), _split_mutations(mutations)))
            elif isinstance(mutations, list):
                out.append((str(name), [str(item).strip() for item in mutations if str(item).strip()]))
            else:
                raise ValueError(f"Mutation set {name!r} must be a string or list.")
        return out
    if isinstance(payload, list):
        out = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("JSON mutation-set lists must contain objects.")
            name = str(item.get("name", "")).strip()
            mutations = item.get("mutations", [])
            if isinstance(mutations, str):
                parsed = _split_mutations(mutations)
            elif isinstance(mutations, list):
                parsed = [str(token).strip() for token in mutations if str(token).strip()]
            else:
                raise ValueError(f"Mutation set {name!r} has invalid mutations.")
            if not name or not parsed:
                raise ValueError(f"Invalid JSON mutation set entry: {item!r}")
            out.append((name, parsed))
        return out
    raise ValueError("Mutation JSON must be a mapping, a list, or contain a 'mutation_sets' key.")


def load_mutation_sets(path: str) -> List[ScenarioDef]:
    """Mutation sets from a JSON file or from a CSV with ``name`` and ``mutations`` columns."""
    lower = path.lower()
    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            return _load_json_mutation_sets(json.load(handle))

    rows: List[ScenarioDef] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Mutation file has no header: {path}")
        required = {"name", "mutations"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Mutation CSV must contain columns {sorted(required)}; missing {sorted(missing)}")
        for row in reader:
            name = str(row.get("name", "")).strip()
            mutations = _split_mutations(str(row.get("mutations", "")))
            if not name or not mutations:
                raise ValueError(f"Invalid mutation CSV row: {row!r}")
            rows.append((name, mutations))
    return rows


def clean_scenarios(scenarios: Sequence[ScenarioDef]) -> List[ScenarioDef]:
    """Strip names/mutations and reject empty, unusable or duplicated scenario names."""
    seen = set()
    out: List[ScenarioDef] = []
    for name, mutations in scenarios:
        clean_name = str(name).strip()
        clean_mutations = [str(token).strip() for token in mutations if str(token).strip()]
        if not clean_name or not clean_mutations:
            raise ValueError(f"Invalid scenario definition: {(name, mutations)!r}")
        if not _VALID_NAME.match(clean_name):
            raise ValueError(
                f"Scenario name '{clean_name}' cannot be used as a directory name; use letters, digits, '.', '_' or '-'."
            )
        if clean_name in seen:
            raise ValueError(f"Duplicate scenario name: {clean_name}")
        seen.add(clean_name)
        out.append((clean_name, clean_mutations))
    return out


def resolve_scenarios(custom_scenarios: Sequence[ScenarioDef], include_builtins: bool) -> List[ScenarioDef]:
    """Final scenario list: built-in toy panel (if requested, or if no custom set is given) + custom sets."""
    scenarios: List[ScenarioDef] = []
    if include_builtins or not custom_scenarios:
        scenarios.extend((name, list(mutations)) for name, mutations in MUTATION_SCENARIOS)
    scenarios.extend((name, list(mutations)) for name, mutations in custom_scenarios)
    return clean_scenarios(scenarios)


# --- assembling the scenario list of a panel ---------------------------------------------------


def collect_scenarios(
    mutation_sets: Optional[Sequence[str]] = None,
    mutation_files: Optional[Sequence[str]] = None,
    include_builtins: bool = False,
    selected: Optional[Sequence[str]] = None,
) -> List[ScenarioDef]:
    """Scenario list of a panel from its command-line style inputs.

    Custom scenarios (files first, then ``NAME:MUT1,MUT2`` strings) replace the
    built-in toy panel unless ``include_builtins`` is set; with no custom
    scenario the built-in panel is used.  ``selected`` (names, repeated and/or
    comma separated) then keeps only those scenarios; unknown names are an
    error rather than being ignored.
    """
    custom: List[ScenarioDef] = []
    for path in mutation_files or []:
        custom.extend(load_mutation_sets(path))
    for text in mutation_sets or []:
        custom.append(parse_mutation_set(text))
    scenarios = resolve_scenarios(custom, include_builtins=include_builtins)

    wanted = parse_csv_tokens(selected)
    if wanted:
        known = {name for name, _ in scenarios}
        unknown = [name for name in wanted if name not in known]
        if unknown:
            raise ValueError(
                f"Unknown scenario(s) {unknown}. Available: {sorted(known)}."
            )
        scenarios = [item for item in scenarios if item[0] in set(wanted)]
    return scenarios


def validate_scenarios(
    scenarios: Sequence[ScenarioDef],
    reference_sequence: str,
    start_pos: int,
) -> Dict[str, str]:
    """Apply every scenario to the reference sequence before anything expensive runs.

    Returns ``{scenario name: mutated sequence}``.  Raises ``ValueError`` naming
    the scenario when a mutation does not fit the reference (wrong residue,
    position outside the sequence, overlapping edits, bad syntax).
    """
    mutated: Dict[str, str] = {}
    for name, mutations in scenarios:
        try:
            mutated[name] = mutate_sequence(reference_sequence, mutations, start_pos)
        except MutationError as exc:
            raise ValueError(
                f"Scenario '{name}' does not fit the reference sequence "
                f"(residues {start_pos}-{start_pos + len(reference_sequence) - 1}): {exc}"
            ) from exc
    return mutated


def scenario_seed_offsets(names: Sequence[str]) -> Dict[str, int]:
    """Stable, distinct seed offset for every scenario name.

    The offset is a hash of the name, so it does not depend on which other
    scenarios are in the panel.  Two names that hash to the same slot (rare)
    are separated by moving the later one to the next free slot.
    """
    offsets: Dict[str, int] = {}
    used = set()
    for name in names:
        slot = zlib.crc32(name.encode("utf-8")) % _SEED_SLOTS
        while slot in used:
            slot = (slot + 1) % _SEED_SLOTS
        used.add(slot)
        offsets[name] = slot
    return offsets
