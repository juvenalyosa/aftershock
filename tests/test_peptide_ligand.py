"""A ligand given as a chain (``chain:B``): the peptide of the PDZ3 benchmark."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("openmm")

from pert.analysis.metal_site import metal_site_series  # noqa: E402
from pert.analysis.pocket_fields import _classify_atoms  # noqa: E402
from pert.analysis.site_frame import build_shared_site_frames  # noqa: E402
from pert.deps import app  # noqa: E402
from pert.structure.binding_site import binding_site_keys_from_pdb, binding_site_residues  # noqa: E402
from pert.structure.comparison import aligned_ca_pair_records  # noqa: E402
from pert.structure.residues import protein_ca_records, sequence_and_start_from_pdb  # noqa: E402
from pert.structure.selection import ligand_selection  # noqa: E402

PEPTIDE = "chain:B"


@pytest.fixture(scope="module")
def pdz3(repo_root) -> str:
    path = os.path.join(repo_root, "data", "pdz3", "1BE9_pdz3_ref.pdb")
    if not os.path.exists(path):
        pytest.skip("PDZ3 reference structure not available")
    return path


def test_selection_tokens():
    both = ligand_selection(["zn", "chain:B,HEM"])
    assert both.resnames == {"ZN", "HEM"} and both.chains == {"B"}
    assert both.tokens == ["HEM", "ZN", "chain:B"]
    assert ligand_selection("CHAIN:b").chains == {"b"}  # the prefix is case-insensitive, the chain id is not
    assert ligand_selection(both) is both
    assert not ligand_selection(None) and not ligand_selection([])
    assert ligand_selection("chain:B").vmd() == "(chain B and not water and not ion)"
    assert ligand_selection("ZN").describe() == "resname ZN"
    with pytest.raises(ValueError, match="names no chain"):
        ligand_selection("chain:")


def test_a_chain_selects_the_peptide_and_never_water(pdz3):
    selection = ligand_selection(PEPTIDE)
    topology = app.PDBFile(pdz3).topology
    assert [f"{r.name}{r.id}" for r in topology.residues() if selection.matches(r)] == ["LYS5", "GLN6", "THR7", "SER8", "VAL9"]
    # Solvent that happens to carry the same chain id is not ligand.
    water_chain = topology.addChain("B")
    assert not selection.matches(topology.addResidue("HOH", water_chain))
    assert not selection.matches(topology.addResidue("NA", water_chain))


def test_the_peptide_is_not_counted_as_protein(pdz3):
    assert len(protein_ca_records(pdz3)) == 106                       # every amino acid in the file
    records = protein_ca_records(pdz3, PEPTIDE)
    assert len(records) == 101 and {rec["chain_id"] for rec in records} == {"A"}
    sequence, start = sequence_and_start_from_pdb(pdz3, "A")
    assert start == 302 and "".join(str(rec["aa1"]) for rec in records) == sequence
    assert len(aligned_ca_pair_records(pdz3, pdz3, PEPTIDE)) == 101


def test_binding_site_and_regions_of_a_peptide_ligand(pdz3):
    site = [key for _, key, _ in binding_site_residues(pdz3, PEPTIDE, 6.0)]
    # the carboxylate-binding loop (Gly322-Phe325) and His372 line the groove; the peptide itself is not "site"
    assert {"GLY322", "LEU323", "GLY324", "PHE325", "HIS372"} <= set(site)
    assert not {"LYS5", "GLN6", "THR7", "SER8", "VAL9"} & set(site)
    assert {chain for chain, _ in binding_site_keys_from_pdb(pdz3, [PEPTIDE], 6.0)} == {"A"}

    topology = app.PDBFile(pdz3).topology
    masks = _classify_atoms(topology, [PEPTIDE])
    in_peptide = np.asarray([atom.residue.chain.id == "B" for atom in topology.atoms()])
    assert np.array_equal(masks["is_ligand"], in_peptide)
    assert np.array_equal(masks["is_protein"], ~in_peptide)


def test_site_frame_is_anchored_on_the_protein_not_on_the_peptide(pdz3):
    wt_frame, mut_frame, diagnostics = build_shared_site_frames(pdz3, pdz3, [PEPTIDE])
    atoms = list(app.PDBFile(pdz3).topology.atoms())
    assert {atoms[int(i)].residue.chain.id for i in wt_frame.anchor_indices} == {"A"}
    assert diagnostics["anchor_count"] >= 6 and "HIS372" in diagnostics["anchor_residues_wt"]
    with pytest.raises(ValueError, match="No ligand atoms"):
        build_shared_site_frames(pdz3, pdz3, ["chain:Z"])


def test_a_peptide_ligand_has_no_metal_site(pdz3):
    # Returns before touching the trajectory: there is nothing to follow.
    assert metal_site_series(pdz3, "does-not-exist.npz", [PEPTIDE]) == []


def test_a_model_numbered_from_one_gets_the_numbering_of_the_reference(tmp_path, pdz3):
    """MODELLER numbers models from 1; the WT of a domain cut from a longer protein does not start there."""
    from pert.structure.prepare import _number_like_the_reference

    # A stand-in for a MODELLER model: the reference with chain A renumbered from 1.
    pdb = app.PDBFile(pdz3)
    for chain in pdb.topology.chains():
        if chain.id == "A":
            for index, residue in enumerate(chain.residues(), start=1):
                residue.id = str(index)
    model = str(tmp_path / "model.pdb")
    with open(model, "w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(pdb.topology, pdb.positions, handle, keepIds=True)
    assert protein_ca_records(model, PEPTIDE)[70]["residue_key"] == "HIS71"

    assert _number_like_the_reference(model, "A", 302) is True
    records = protein_ca_records(model, PEPTIDE)
    assert records[0]["residue_key"] == "LEU302" and records[70]["residue_key"] == "HIS372"
    assert [rec["residue_key"] for rec in records] == [rec["residue_key"] for rec in protein_ca_records(pdz3, PEPTIDE)]
    # the peptide keeps its own numbering, and a second call has nothing to do
    assert [f"{r.name}{r.id}" for r in app.PDBFile(model).topology.residues() if r.chain.id == "B"][0] == "LYS5"
    before = open(model).read()
    assert _number_like_the_reference(model, "A", 302) is False and open(model).read() == before


def test_site_check_reports_that_a_peptide_panel_has_no_metal(tmp_path, pdz3):
    from pert.pipelines.site_check import NoMetalSite, run_site_check
    from synthetic import make_trajectory, reference_positions, write_run

    reference = reference_positions(pdz3)
    npz = str(tmp_path / "p" / "s" / "rep1" / "md" / "wt_traj.npz")
    make_trajectory(reference, seed=1, n_frames=5, out_path=npz)
    write_run(str(tmp_path / "p" / "s" / "rep1"), pdz3, npz, pdz3, npz)
    with pytest.raises(NoMetalSite, match="not a metal ion"):
        run_site_check(str(tmp_path / "p"), ligand_resnames=[PEPTIDE])
