"""Residue-level views of a PDB structure.

Provides residue classification, sequence extraction and the two C-alpha
record formats used in the package:

* ``ca_records_from_topology`` -> ``[(aa1, xyz_nm), ...]`` for one chain of an
  in-memory OpenMM topology (used while building structures);
* ``protein_ca_records`` -> list of dicts with residue keys, chain, residue id
  and atom index for a PDB file (used by every post-processing layer).

Residue keys look like ``LYS2``; node labels add the chain (``A:LYS2``).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from pert.constants import AA3_TO_1, HALOGENS, METALS, STANDARD_AA3_TO_1, TYPE_TO_INDEX
from pert.deps import app, ensure_openmm, unit
from pert.structure.selection import LigandSpec, ligand_selection


# --- residue / atom classification -----------------------------------------


def is_protein_residue(resname: str) -> bool:
    """True for residue names in the permissive amino-acid table."""
    return resname.upper() in AA3_TO_1


def residue_to_aa1(resname: str) -> Optional[str]:
    """Three-letter -> one-letter code (None for non-protein residues)."""
    return AA3_TO_1.get(resname.upper())


def parse_residue_number(residue_id: str) -> Optional[int]:
    """First integer found in a PDB residue id (handles insertion codes); None if absent."""
    match = re.search(r"-?\d+", str(residue_id))
    if match:
        return int(match.group(0))
    return None


def atom_type_bin(symbol: Optional[str]) -> int:
    """Channel index of an element symbol in ``TYPE_BINS`` (C, N, O, S, P, H, halogen, metal, other)."""
    if not symbol:
        return TYPE_TO_INDEX["OTHER"]
    sym = symbol.upper()
    if sym in TYPE_TO_INDEX:
        return TYPE_TO_INDEX[sym]
    if sym in HALOGENS:
        return TYPE_TO_INDEX["HALOGEN"]
    if sym in METALS:
        return TYPE_TO_INDEX["METAL"]
    return TYPE_TO_INDEX["OTHER"]


def aa_code_from_residue_key(key: str) -> str:
    """Residue-name part of a residue key: ``"LYS2" -> "LYS"``."""
    match = re.match(r"([A-Za-z]+)", str(key).strip())
    return match.group(1).upper() if match else ""


def node_sort_key(label: str) -> Tuple[str, int, str]:
    """Sort key ``(chain, residue number, residue)`` for node labels such as ``A:LYS2``."""
    text = str(label)
    chain = ""
    residue = text
    if ":" in text:
        chain, residue = text.split(":", 1)
    match = re.search(r"-?\d+", residue)
    resid = int(match.group(0)) if match else 0
    return chain, resid, residue


# --- sequence and C-alpha records -------------------------------------------


def sequence_and_start_from_pdb(pdb_path: str, chain_id: Optional[str]) -> Tuple[str, int]:
    """One-letter sequence and first residue number of a chain of a PDB file."""
    ensure_openmm()
    pdb = app.PDBFile(pdb_path)
    seq_chars: List[str] = []
    first_pos: Optional[int] = None
    for chain in pdb.topology.chains():
        if chain_id and chain.id != chain_id:
            continue
        for residue in chain.residues():
            if not is_protein_residue(residue.name):
                continue
            aa1 = residue_to_aa1(residue.name)
            if aa1 is None:
                continue
            seq_chars.append(aa1)
            if first_pos is None:
                parsed = parse_residue_number(residue.id)
                if parsed is not None:
                    first_pos = parsed
        if seq_chars:
            break
    if not seq_chars:
        raise ValueError(
            f"No protein residues found in chain '{chain_id}' for {pdb_path}."
        )
    if first_pos is None:
        first_pos = 1
    return "".join(seq_chars), first_pos


def ca_records_from_topology(
    topology: "app.Topology",
    positions: "unit.Quantity",
    chain_id: Optional[str],
) -> List[Tuple[str, np.ndarray]]:
    """``[(aa1, ca_xyz_nm), ...]`` for the first matching protein chain of a topology."""
    ensure_openmm()
    pos_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    records: List[Tuple[str, np.ndarray]] = []
    for chain in topology.chains():
        if chain_id and chain.id != chain_id:
            continue
        for residue in chain.residues():
            if not is_protein_residue(residue.name):
                continue
            aa1 = residue_to_aa1(residue.name)
            if aa1 is None:
                continue
            ca_atom = None
            for atom in residue.atoms():
                if atom.name == "CA":
                    ca_atom = atom
                    break
            if ca_atom is None:
                continue
            records.append((aa1, pos_nm[ca_atom.index].copy()))
        if records:
            break
    return records


def protein_ca_records(pdb_path: str, ligand: LigandSpec = None) -> List[Dict[str, object]]:
    """C-alpha record of every standard residue of a PDB file.

    Each record is a dict with ``aa1``, ``residue_key`` (e.g. ``LYS2``), ``resid``,
    ``chain_id``, ``atom_index`` (index of the CA atom, usable on trajectories)
    and ``coord`` (nm).  Residues of ``ligand`` (a peptide given as
    ``chain:B``) are not part of the protein and are left out.
    """
    selection = ligand_selection(ligand)
    pdb = app.PDBFile(pdb_path)
    pos_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    out: List[Dict[str, object]] = []
    for chain in pdb.topology.chains():
        for residue in chain.residues():
            resname = residue.name.upper()
            if resname not in STANDARD_AA3_TO_1 or selection.matches(residue):
                continue
            ca_atom = None
            for atom in residue.atoms():
                if atom.name == "CA":
                    ca_atom = atom
                    break
            if ca_atom is None:
                continue
            residue_key = f"{residue.name}{residue.id}"
            out.append(
                {
                    "aa1": STANDARD_AA3_TO_1[resname],
                    "residue_key": residue_key,
                    "resid": str(residue.id).strip(),
                    "chain_id": chain.id,
                    "atom_index": int(ca_atom.index),
                    "coord": pos_nm[ca_atom.index].copy(),
                }
            )
    return out
