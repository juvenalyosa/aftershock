"""Which residues are "the ligand".

The ligand defines the site: its position is the origin of the site frame, its
atoms are region ``L``, and the residues around it are the binding site.  It
is given as a list of tokens, wherever the code takes ``ligand_resname(s)``:

* a residue name, for small molecules and ions: ``ZN``, ``HEM``, ``ATP``;
* ``chain:<id>``, for a peptide or protein partner: ``chain:B`` selects every
  residue of chain ``B`` except water and solvent ions.

A residue of a ligand chain is an amino acid, so the residue name alone cannot
tell it from the protein under study.  Every function that enumerates
"protein" residues therefore takes the ligand and leaves its residues out;
otherwise a peptide ligand would be aligned, anchored and ranked as if it were
part of the protein.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Union

from pert.constants import WATER_RESNAMES

CHAIN_PREFIX = "chain:"

# Monatomic solvent ions that share no chain with a ligand in practice, but
# must never be taken for one if they do.
_SOLVENT_ION_RESNAMES = frozenset({"NA", "CL", "K", "NA+", "CL-", "K+", "SOD", "CLA", "POT"})


@dataclass(frozen=True)
class LigandSelection:
    """Ligand residues, by residue name and/or by chain."""

    resnames: FrozenSet[str] = frozenset()
    chains: FrozenSet[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.resnames or self.chains)

    def matches(self, residue) -> bool:
        """Is this (OpenMM topology) residue part of the ligand?"""
        name = residue.name.strip().upper()
        if name in self.resnames:
            return True
        if self.chains and (residue.chain.id or "").strip() in self.chains:
            return name not in WATER_RESNAMES and name not in _SOLVENT_ION_RESNAMES
        return False

    @property
    def tokens(self) -> List[str]:
        """The selection as the tokens it was written with."""
        return sorted(self.resnames) + [f"{CHAIN_PREFIX}{chain}" for chain in sorted(self.chains)]

    def describe(self) -> str:
        """Human-readable form, e.g. ``resname ZN`` or ``chain B``."""
        parts = [f"resname {name}" for name in sorted(self.resnames)] + [f"chain {chain}" for chain in sorted(self.chains)]
        return " or ".join(parts) if parts else "none"

    def vmd(self) -> str:
        """VMD atom selection of the ligand."""
        parts = [f"resname {name}" for name in sorted(self.resnames)]
        parts += [f"(chain {chain} and not water and not ion)" for chain in sorted(self.chains)]
        return " or ".join(parts) if parts else "none"


LigandSpec = Union[None, str, Sequence[str], LigandSelection]


def ligand_selection(spec: LigandSpec) -> LigandSelection:
    """Build a selection from a token, a list of tokens (or of comma-separated tokens), or pass one through."""
    if isinstance(spec, LigandSelection):
        return spec
    if spec is None:
        return LigandSelection()
    items = [spec] if isinstance(spec, str) else list(spec)
    resnames = set()
    chains = set()
    for item in items:
        for token in str(item).split(","):
            token = token.strip()
            if not token:
                continue
            if token.lower().startswith(CHAIN_PREFIX):
                chain = token[len(CHAIN_PREFIX):].strip()
                if not chain:
                    raise ValueError(f"Ligand token '{token}' names no chain (expected e.g. 'chain:B').")
                chains.add(chain)  # chain identifiers are case-sensitive
            else:
                resnames.add(token.upper())
    return LigandSelection(frozenset(resnames), frozenset(chains))


def is_ligand_residue(residue, spec: Optional[LigandSpec]) -> bool:
    """Convenience form of ``ligand_selection(spec).matches(residue)``."""
    return ligand_selection(spec).matches(residue)
