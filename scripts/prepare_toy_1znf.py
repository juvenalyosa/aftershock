"""
One-off helper: build ``data/toy/1ZNF_clean_ref.pdb`` from ``data/toy/1ZNF.pdb``.

Keeps the first protein chain (as chain A) and the zinc ion (as chain Z) of the
NMR entry 1ZNF and drops everything else, producing the clean reference used
by the toy runs.  Run it from the repository root::

    python scripts/prepare_toy_1znf.py
"""

from openmm import app, unit, Vec3

AA3 = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
}

pdb = app.PDBFile('data/toy/1ZNF.pdb')
new_top = app.Topology()
prot_chain = new_top.addChain('A')
lig_chain = new_top.addChain('Z')
old_to_new = {}
selected_xyz_nm = []

protein_chain = None
for chain in pdb.topology.chains():
    if any(res.name in AA3 for res in chain.residues()):
        protein_chain = chain
        break
if protein_chain is None:
    raise RuntimeError('No protein chain found')

for residue in protein_chain.residues():
    if residue.name not in AA3:
        continue
    new_res = new_top.addResidue(residue.name, prot_chain, id=residue.id)
    for atom in residue.atoms():
        new_atom = new_top.addAtom(atom.name, atom.element, new_res, id=atom.id)
        old_to_new[atom] = new_atom
        vec = pdb.positions[atom.index].value_in_unit(unit.nanometer)
        selected_xyz_nm.append((vec.x, vec.y, vec.z))

zn_atom = None
for atom in pdb.topology.atoms():
    if atom.residue.name == 'ZN':
        zn_atom = atom
        break
if zn_atom is None:
    raise RuntimeError('No ZN ligand found')

zn_res = new_top.addResidue('ZN', lig_chain, id='1')
new_top.addAtom('ZN', zn_atom.element, zn_res, id=zn_atom.id)
vec = pdb.positions[zn_atom.index].value_in_unit(unit.nanometer)
selected_xyz_nm.append((vec.x, vec.y, vec.z))

for a1, a2 in pdb.topology.bonds():
    if a1 in old_to_new and a2 in old_to_new:
        try:
            new_top.addBond(old_to_new[a1], old_to_new[a2], type=None, order=None)
        except TypeError:
            new_top.addBond(old_to_new[a1], old_to_new[a2])

positions = [Vec3(x, y, z) for x, y, z in selected_xyz_nm] * unit.nanometer
with open('data/toy/1ZNF_clean_ref.pdb', 'w') as f:
    app.PDBFile.writeFile(new_top, positions, f, keepIds=True)
print('Wrote data/toy/1ZNF_clean_ref.pdb')
