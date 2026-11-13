"""Small, dependency-free readers/writers and tolerant value parsers.

These helpers used to be copy-pasted into every script; they live here so the
tabular outputs of all layers are written the same way.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence


# --- JSON / CSV --------------------------------------------------------------


def load_json(path: str) -> Dict:
    """Read a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: str) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dict rows (all values are strings)."""
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write dict rows to CSV.

    Columns are the union of the keys of all rows, in order of first appearance.
    An empty ``rows`` produces an empty file so downstream steps can still find it.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# --- tolerant scalar parsing (CSV cells arrive as strings) -------------------


def safe_float(value: Any, default: float = float("nan")) -> float:
    """``float(value)`` that returns ``default`` (NaN) for None/empty/unparseable input."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """``int(float(value))`` that returns ``default`` for None/empty/unparseable input."""
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    """Interpret booleans read back from CSV (``"True"``, ``"1"``, ``"yes"``...)."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


# --- CLI / report formatting -------------------------------------------------


def parse_csv_tokens(values: Optional[Sequence[str]]) -> List[str]:
    """Flatten repeated and/or comma-separated CLI values into one token list.

    ``["A,B", "C"] -> ["A", "B", "C"]``; ``None`` -> ``[]``.
    """
    out: List[str] = []
    for value in values or []:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)
