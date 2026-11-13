"""Chemical vocabularies shared by the whole package.

Two residue tables coexist on purpose:

* ``AA3_TO_1`` is the permissive table used when *building and simulating*
  structures (it accepts protonation variants HID/HIE/HIP and the
  non-canonical SEC/PYL/MSE).
* ``STANDARD_AA3`` / ``STANDARD_AA3_TO_1`` are the strict tables used by the
  *post-processing* layers (C-alpha records, binding-site shells): the 20
  standard residues plus the cysteine states CYM and CYX.

Keep them separate: merging them would silently change which residues the
analysis layers see.
"""

from __future__ import annotations


# --- permissive residue table (structure building / MD) ------------------


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "CYM": "C",
    "CYX": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
}


# --- strict 20-residue tables (post-processing) ---------------------------
# Plus the two other names a cysteine can carry: CYM (thiolate, written by the
# MD stage for metal-bound cysteines) and CYX (disulfide).  Without them the
# residues that hold a metal site would disappear from every analysis.


STANDARD_AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "CYM",
    "CYX",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


STANDARD_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "CYM": "C",
    "CYX": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


# --- solvent / element classes --------------------------------------------


WATER_RESNAMES = {"HOH", "WAT", "TIP3", "TIP3P", "SOL", "H2O"}


HALOGENS = {"F", "CL", "BR", "I"}


METALS = {
    "LI",
    "NA",
    "K",
    "RB",
    "CS",
    "MG",
    "CA",
    "SR",
    "BA",
    "ZN",
    "CU",
    "MN",
    "FE",
    "CO",
    "NI",
    "CD",
    "HG",
    "AL",
}


# --- atom-type channels of the pocket-frame density fields ----------------


TYPE_BINS = ["C", "N", "O", "S", "P", "H", "HALOGEN", "METAL", "OTHER"]


TYPE_TO_INDEX = {name: i for i, name in enumerate(TYPE_BINS)}
