"""Building blocks shared by all generated VMD (Tcl) scripts.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pert.structure.residues import node_sort_key


def tcl_braced_path(path: Optional[str]) -> str:
    """Absolute path as a brace-quoted Tcl word (``{}`` for empty)."""
    if not path:
        return "{}"
    norm = os.path.abspath(path).replace("\\", "/").replace("}", "\\}")
    return "{" + norm + "}"


def site_selection_from_keys(keys: Sequence[Tuple[str, str]]) -> str:
    """VMD selection text for a list of ``(chain, resid)`` keys."""
    if not keys:
        return "none"
    terms: List[str] = []
    for chain_id, resid in keys:
        if chain_id:
            terms.append(f"(chain {chain_id} and resid {resid})")
        else:
            terms.append(f"(resid {resid})")
    return "protein and (" + " or ".join(terms) + ")"


def selection_from_rows(rows: Sequence[Dict[str, Any]]) -> str:
    """VMD selection text for result rows that carry ``chain`` and ``resid``."""
    terms: List[str] = []
    seen = set()
    for row in rows:
        chain = str(row.get("chain", "")).strip()
        resid = str(row.get("resid", "")).strip()
        if not resid:
            continue
        key = (chain, resid)
        if key in seen:
            continue
        seen.add(key)
        if chain:
            terms.append(f"(chain {chain} and resid {resid})")
        else:
            terms.append(f"(resid {resid})")
    if not terms:
        return "none"
    return "protein and (" + " or ".join(terms) + ")"


def selection_from_indices(pair_records: Sequence[Dict[str, object]], indices: Sequence[int]) -> str:
    """VMD selection text for pair records addressed by index."""
    terms: List[str] = []
    seen: Set[Tuple[str, str]] = set()
    for idx in indices:
        if idx < 0 or idx >= len(pair_records):
            continue
        chain_id = str(pair_records[idx]["chain_id"]).strip()
        resid = str(pair_records[idx]["resid"]).strip()
        key = (chain_id, resid)
        if key in seen:
            continue
        seen.add(key)
        if chain_id:
            terms.append(f"(chain {chain_id} and resid {resid})")
        else:
            terms.append(f"(resid {resid})")
    if not terms:
        return "none"
    return "protein and (" + " or ".join(terms) + ")"


def selection_from_consensus_rows(rows: Sequence[Dict[str, Any]]) -> str:
    """VMD selection text for consensus rows (falls back to the residue number in the node label)."""
    terms: List[str] = []
    seen = set()
    for row in rows:
        chain = str(row.get("chain", "")).strip()
        resid = str(row.get("resid", "")).strip()
        if not resid:
            _, parsed_resid, _ = node_sort_key(str(row.get("node_label", "")))
            resid = str(parsed_resid) if parsed_resid else ""
        if not resid:
            continue
        key = (chain, resid)
        if key in seen:
            continue
        seen.add(key)
        if chain:
            terms.append(f"(chain {chain} and resid {resid})")
        else:
            terms.append(f"(resid {resid})")
    if not terms:
        return "none"
    return "protein and (" + " or ".join(terms) + ")"
