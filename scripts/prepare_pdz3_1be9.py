"""
Build ``data/pdz3/1BE9_pdz3_ref.pdb``, the reference structure of the PDZ3 benchmark.

PDB entry 1BE9 is the third PDZ domain of PSD-95 (chain A) bound to a
C-terminal peptide of CRIPT (chain B, KQTSV).  The reference keeps

* chain A, residues 302-402: the native sequence up to the end of the third
  helix (alpha3, 394-399).  Residues 297-301 and 403-415 of the crystallised
  construct come from the expression vector, are partly disordered and are
  dropped; 402 is also where the constructs of Petit et al. (PNAS 2009) end,
  whose C-terminal truncation (396-402) lowers the affinity for CRIPT 21-fold;
* chain B, the peptide, which is the ligand (``chain:B``);
* no crystallographic water.

Side-chain atoms missing from the crystal structure (Asp332, Lys5 of the
peptide) are rebuilt with MODELLER's ``complete_pdb``; atoms that were observed
keep their coordinates.  Run it from the repository root::

    python scripts/prepare_pdz3_1be9.py

``data/pdz3/1BE9.pdb`` is downloaded from the RCSB if it is not there.
"""

import os
import tempfile
import urllib.request

SOURCE = "data/pdz3/1BE9.pdb"
OUTPUT = "data/pdz3/1BE9_pdz3_ref.pdb"
KEEP = {"A": (302, 402), "B": (5, 9)}


def main() -> None:
    from modeller import Environ, log
    from modeller.scripts import complete_pdb

    if not os.path.exists(SOURCE):
        os.makedirs(os.path.dirname(SOURCE), exist_ok=True)
        urllib.request.urlretrieve("https://files.rcsb.org/download/1BE9.pdb", SOURCE)

    kept = []
    with open(SOURCE, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain, number = line[21], int(line[22:26])
            if chain in KEEP and KEEP[chain][0] <= number <= KEEP[chain][1] and line[16] in (" ", "A"):
                kept.append(line)
    selected = []
    for chain in KEEP:
        selected.extend(line for line in kept if line[21] == chain)
        selected.append("TER\n")

    log.none()
    env = Environ()
    env.libs.topology.read(file="$(LIB)/top_heav.lib")
    env.libs.parameters.read(file="$(LIB)/par.lib")
    with tempfile.TemporaryDirectory() as tmp:
        trimmed = os.path.join(tmp, "trimmed.pdb")
        with open(trimmed, "w", encoding="utf-8") as handle:
            handle.writelines(selected + ["END\n"])
        # heavy atoms only; the residue numbers and chains of the input are kept
        model = complete_pdb(env, trimmed, transfer_res_num=True)
        completed = os.path.join(tmp, "completed.pdb")
        model.write(file=completed)
        with open(completed, "r", encoding="utf-8") as handle:
            atoms = [line for line in handle if line.startswith(("ATOM", "TER"))]

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write("REMARK   PDZ3 of PSD-95 (1BE9 chain A, 302-402) with the CRIPT peptide (chain B, 5-9).\n")
        handle.write("REMARK   Built by scripts/prepare_pdz3_1be9.py; missing side chains rebuilt with MODELLER.\n")
        handle.writelines(atoms)
        handle.write("END\n")
    n_atoms = sum(1 for line in atoms if line.startswith("ATOM"))
    print(f"Wrote {OUTPUT}: {n_atoms} atoms ({len(selected) - len(KEEP)} taken from the crystal structure).")


if __name__ == "__main__":
    main()
