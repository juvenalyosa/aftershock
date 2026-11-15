"""Stage 1 of the pipeline: build the WT/mutant structure pair.

``prepare_structures`` applies the mutations to the reference sequence, builds
the mutant with MODELLER, and transfers the ligand, producing::

    <output_dir>/structures/wt_with_ligand.pdb
    <output_dir>/structures/mutant_raw.pdb
    <output_dir>/structures/mutant_with_ligand.pdb
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from typing import Optional

from pert.config import BuildConfig, PreparedStructures
from pert.deps import app, ensure_openmm
from pert.sequence.mutations import mutate_sequence, MutationError
from pert.structure.ligand import transfer_ligand_by_local_alignment
from pert.structure.residues import is_protein_residue, parse_residue_number, sequence_and_start_from_pdb
from pert.utils.paths import pushd


def _build_signature(config: BuildConfig, reference_pdb: str, original_sequence: str, start_pos: int) -> str:
    """Fingerprint of everything that determines the built structures."""
    with open(reference_pdb, "rb") as handle:
        reference_sha1 = hashlib.sha1(handle.read()).hexdigest()
    payload = {
        "reference_pdb_sha1": reference_sha1,
        "ligand_resnames": [name.upper() for name in config.ligand_resnames],
        "mutations": list(config.mutations),
        "templates": list(config.templates),
        "chain_id": config.chain_id,
        "original_sequence": original_sequence,
        "start_pos": start_pos,
        "num_models": config.num_models,
        "modeller_seed": config.modeller_seed,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _reusable_build(record_path: str, signature: str, files) -> Optional[PreparedStructures]:
    """The structures of a previous call, if they were built with the same settings.

    Raises when structures exist but cannot be shown to match: rebuilding would
    silently change the starting point of any simulation already run on them.
    """
    existing = [path for path in files if os.path.exists(path)]
    if not existing:
        return None
    record = None
    if os.path.exists(record_path):
        with open(record_path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    if record is None or record.get("signature") != signature or len(existing) != len(files):
        raise RuntimeError(
            f"Structures already exist in '{os.path.dirname(record_path)}' but were built with different "
            "settings (or by a version that did not record them). Choose another output directory or pass "
            "resume=False (--no-resume) to rebuild."
        )
    return PreparedStructures(**record["structures"])


def _number_like_the_reference(pdb_path: str, chain_id: Optional[str], start_pos: int) -> bool:
    """Number the protein chain of a model consecutively from ``start_pos``.

    MODELLER numbers its models from 1.  When the reference starts elsewhere
    (a domain cut from a longer protein), residue 372 of the WT would be called
    71 in the mutant, and every table that joins the two by residue name would
    silently fail to match them.  After a substitution the mutant now has the
    numbering of the reference; after an insertion or deletion the residues
    that follow are shifted, as their sequence positions are.

    The file is left untouched when it is already numbered that way.  Returns
    whether it was rewritten.
    """
    pdb = app.PDBFile(pdb_path)
    residues = [
        residue
        for chain in pdb.topology.chains()
        if not chain_id or chain.id == chain_id
        for residue in chain.residues()
        if is_protein_residue(residue.name)
    ]
    wanted = [str(start_pos + offset) for offset in range(len(residues))]
    if [str(parse_residue_number(residue.id)) for residue in residues] == wanted:
        return False
    for residue, number in zip(residues, wanted):
        residue.id = number
        residue.insertionCode = ""
    with open(pdb_path, "w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(pdb.topology, pdb.positions, handle, keepIds=True)
    return True


def prepare_structures(config: BuildConfig, output_dir: str, resume: bool = False) -> PreparedStructures:
    """Build the WT/mutant structure pair for one mutation set (see module docstring).

    With ``resume=True`` structures already built with the same settings are
    returned without calling MODELLER again; this keeps the mutant model (and
    therefore every simulation started from it) unchanged across relaunches.
    The settings are recorded in ``structures/build.json``.
    """
    ensure_openmm()
    try:
        from pert.structure.homology import buildm
    except Exception as exc:
        raise ImportError(
            "Could not import `buildm` from pert.structure.homology. Ensure MODELLER and project dependencies are installed."
        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    structure_dir = os.path.join(output_dir, "structures")
    model_dir = os.path.join(output_dir, "model_build")
    os.makedirs(structure_dir, exist_ok=True)

    reference_pdb = os.path.abspath(config.reference_pdb)
    wt_out = os.path.join(structure_dir, "wt_with_ligand.pdb")
    mutant_raw_out = os.path.join(structure_dir, "mutant_raw.pdb")
    mutant_ligand_out = os.path.join(structure_dir, "mutant_with_ligand.pdb")

    if config.original_sequence:
        original_sequence = config.original_sequence.strip().upper()
    else:
        seq_from_pdb, inferred_start = sequence_and_start_from_pdb(reference_pdb, config.chain_id)
        original_sequence = seq_from_pdb
        if config.start_pos is None:
            config.start_pos = inferred_start

    if config.start_pos is None:
        _, inferred_start = sequence_and_start_from_pdb(reference_pdb, config.chain_id)
        config.start_pos = inferred_start

    start_pos = int(config.start_pos)
    try:
        mutated_sequence = mutate_sequence(original_sequence, config.mutations, start_pos)
    except MutationError as exc:
        raise ValueError(f"Mutation specification is invalid: {exc}") from exc

    record_path = os.path.join(structure_dir, "build.json")
    signature = _build_signature(config, reference_pdb, original_sequence, start_pos)
    if resume:
        previous = _reusable_build(record_path, signature, (wt_out, mutant_raw_out, mutant_ligand_out))
        if previous is not None:
            print(f"[build] {structure_dir}: structures already built, reusing.")
            return previous

    with pushd(model_dir):
        best_rel = buildm(
            sequence=original_sequence,
            templates=config.templates,
            mutation=config.mutations,
            num_models=config.num_models,
            start_pos=start_pos,
            num_processors=config.num_processors,
            modeller_seed=config.modeller_seed,
            modeller_deterministic=config.modeller_seed is not None,
        )
        if not best_rel:
            raise RuntimeError("MODELLER did not return a best model path.")
        best_abs = os.path.abspath(best_rel)

    shutil.copy2(reference_pdb, wt_out)
    shutil.copy2(best_abs, mutant_raw_out)
    _number_like_the_reference(mutant_raw_out, config.chain_id, start_pos)

    transfer_info = transfer_ligand_by_local_alignment(
        reference_pdb=reference_pdb,
        mutant_pdb=mutant_raw_out,
        output_pdb=mutant_ligand_out,
        ligand_resnames=config.ligand_resnames,
        chain_id=config.chain_id,
    )
    prepared = PreparedStructures(
        wt_pdb=wt_out,
        mutant_raw_pdb=mutant_raw_out,
        mutant_ligand_pdb=mutant_ligand_out,
        original_sequence=original_sequence,
        mutated_sequence=mutated_sequence,
        start_pos=start_pos,
        alignment_pairs=int(transfer_info["alignment_pairs"]),
        local_alignment_pairs=int(transfer_info["local_alignment_pairs"]),
        local_alignment_rmsd_ang=float(transfer_info["local_alignment_rmsd_ang"]),
        ligand_atoms=int(transfer_info["ligand_atoms"]),
    )
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump({"signature": signature, "structures": asdict(prepared)}, handle, indent=2)
    return prepared
