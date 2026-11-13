"""Stage 2 of the pipeline: molecular dynamics with OpenMM.

Protocol (identical for WT and mutant): Amber14 + TIP3P-FB, hydrogens added,
optional solvation/neutralisation, energy minimisation, staged heating,
equilibration, then NPT/NVT production with a Langevin-middle integrator.
Positions *and velocities* of every atom are stored each ``report_interval``
steps because the downstream field embedding uses both.

Reproducibility
    OpenMM places hydrogens and ions with Python's global random generator.
    It is seeded from ``MDConfig.random_seed`` so that the simulated system is
    the same on every launch, and the simulated system (``*_system.xml``) and
    the state at the start of production (``*_start.xml``) are written to disk.
    The dynamics themselves are only bit-reproducible on deterministic
    platforms (Reference); on GPUs two launches still diverge.

Restart
    Production is written in parts: every ``checkpoint_interval_steps`` the
    frames collected so far, a checkpoint and a progress file are saved.
    Calling ``run_openmm_md`` again with the same inputs

    * returns immediately if the simulation is complete,
    * continues from the last checkpoint if it was interrupted,
    * extends it if ``production_steps`` was increased.

    Existing files produced with *different* settings are never overwritten
    silently: that is an error unless ``resume=False``.

Files written for a prefix ``md/wt``::

    wt_prepared.pdb    topology + coordinates of the simulated system
                       (+ .metal_site.json: protonation chosen for metal-bound residues)
    wt_system.xml      serialized OpenMM System
    wt_start.xml       state at the start of production
    wt_traj.npz        positions, velocities, time_ps, dt_ps (+ box_nm)
    wt_traj.dcd        the same frames, for VMD
    wt_md.json         protocol log
    wt_progress.json   restart bookkeeping
    wt_restart/        latest checkpoint (and trajectory parts while running)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from pert.config import MDConfig, MDRunArtifacts
from pert.constants import METALS, WATER_RESNAMES
from pert.deps import (
    app,
    ensure_openmm,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    unit,
    Vec3,
    XmlSerializer,
)
from pert.structure.residues import is_protein_residue


# --- force-field / platform plumbing ----------------------------------------


def _resolve_nonbonded_method(name: str):
    """Map a method name (PME, Ewald, CutoffPeriodic, ...) to the OpenMM constant."""
    key = (name or "PME").strip().upper()
    mapping = {
        "PME": app.PME,
        "EWALD": app.Ewald,
        "CUTOFFPERIODIC": app.CutoffPeriodic,
        "CUTOFFNONPERIODIC": app.CutoffNonPeriodic,
        "NOCUTOFF": app.NoCutoff,
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported nonbonded_method '{name}'. "
            "Use one of PME, Ewald, CutoffPeriodic, CutoffNonPeriodic, NoCutoff."
        )
    return mapping[key]


def _is_periodic_nonbonded_method(method) -> bool:
    """True when the method needs a periodic box."""
    return method in (app.PME, app.Ewald, app.CutoffPeriodic)


def _available_openmm_platform_names() -> List[str]:
    """Names of the OpenMM platforms compiled into this installation."""
    ensure_openmm()
    names: List[str] = []
    count = int(Platform.getNumPlatforms())
    for index in range(count):
        names.append(str(Platform.getPlatform(index).getName()))
    return names


def _select_openmm_platform(requested: str) -> Tuple["Platform", str, List[str]]:
    """Pick the requested platform, or the fastest available (CUDA > OpenCL > CPU > Reference)."""
    ensure_openmm()
    available = _available_openmm_platform_names()
    if not available:
        raise RuntimeError("No OpenMM platforms are available in this environment.")

    by_upper = {name.upper(): name for name in available}
    requested_clean = (requested or "AUTO").strip()
    requested_upper = requested_clean.upper()

    preference = ["CUDA", "OPENCL", "CPU", "REFERENCE"]

    selected_name: Optional[str] = None
    if requested_upper == "AUTO":
        for pref in preference:
            if pref in by_upper:
                selected_name = by_upper[pref]
                break
        if selected_name is None:
            selected_name = available[0]
    elif requested_upper in by_upper:
        selected_name = by_upper[requested_upper]
    else:
        for pref in preference:
            if pref in by_upper:
                selected_name = by_upper[pref]
                break
        if selected_name is None:
            selected_name = available[0]
        print(
            f"[OpenMM] Requested platform '{requested_clean}' not available. "
            f"Falling back to '{selected_name}'. Available: {', '.join(available)}"
        )

    platform = Platform.getPlatformByName(selected_name)
    return platform, selected_name, available


def _sanitize_topology_for_forcefield(
    topology: "app.Topology",
    positions: "unit.Quantity",
) -> Tuple["app.Topology", "unit.Quantity"]:
    """
    Rebuild topology while removing cross-bonds between protein and non-protein
    residues, which frequently break template matching for metalloproteins.
    """
    ensure_openmm()
    pos_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    new_top = app.Topology()
    atom_map = {}
    new_positions = []

    for chain in topology.chains():
        new_chain = new_top.addChain(chain.id)
        for residue in chain.residues():
            new_res = new_top.addResidue(
                residue.name,
                new_chain,
                id=residue.id,
                insertionCode=residue.insertionCode,
            )
            for atom in residue.atoms():
                new_atom = new_top.addAtom(
                    atom.name,
                    atom.element,
                    new_res,
                    id=atom.id,
                    formalCharge=atom.formalCharge,
                )
                atom_map[atom] = new_atom
                xyz = pos_nm[atom.index]
                new_positions.append(Vec3(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    def _is_protein_res(resname: str) -> bool:
        return is_protein_residue(resname)

    # Rebuild canonical protein bonding from residue templates to avoid
    # nonstandard topology artifacts from modeled PDB files.
    new_top.createStandardBonds()

    # Preserve ligand/cofactor intra-residue bonds from the source topology.
    for atom1, atom2 in topology.bonds():
        res1 = atom1.residue
        res2 = atom2.residue
        same_residue = res1 is res2
        p1 = _is_protein_res(res1.name.upper())
        p2 = _is_protein_res(res2.name.upper())
        if not same_residue:
            continue
        if p1 and p2:
            continue
        try:
            new_top.addBond(atom_map[atom1], atom_map[atom2], type=None, order=None)
        except TypeError:
            new_top.addBond(atom_map[atom1], atom_map[atom2])

    pos_qty = new_positions * unit.nanometer
    try:
        new_top.createDisulfideBonds(pos_qty)
    except Exception:
        pass
    return new_top, pos_qty


# --- reproducibility and restart bookkeeping ------------------------------------

FORCEFIELD_FILES = ("amber14-all.xml", "amber14/tip3pfb.xml")

# Settings that do not change what is simulated: the platform, how long
# production runs and how often it is checkpointed.  Everything else is part of
# the signature that guards against mixing incompatible files.
_NOT_IN_SIGNATURE = ("platform", "production_steps", "checkpoint_interval_steps")


@contextmanager
def _seeded_python_random(seed: Optional[int]) -> Iterator[None]:
    """Seed Python's global RNG (used by OpenMM's Modeller) and restore it afterwards."""
    if seed is None:
        yield
        return
    state = random.getstate()
    random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(state)


def _file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md_signature(input_pdb: str, ligand_xml: str, md_config: MDConfig) -> str:
    """Fingerprint of everything that defines the simulated system and protocol.

    Two calls with the same signature simulate the same thing (up to the
    production length), so their files can be continued or reused.
    """
    payload = {
        "input_pdb_sha1": _file_sha1(input_pdb),
        "ligand_xml_sha1": _file_sha1(ligand_xml),
        "forcefield": list(FORCEFIELD_FILES),
        "config": {k: v for k, v in asdict(md_config).items() if k not in _NOT_IN_SIGNATURE},
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _md_paths(output_prefix: str) -> Dict[str, str]:
    return {
        "prepared": f"{output_prefix}_prepared.pdb",
        "metal_site": f"{output_prefix}_prepared.pdb.metal_site.json",
        "system": f"{output_prefix}_system.xml",
        "start": f"{output_prefix}_start.xml",
        "npz": f"{output_prefix}_traj.npz",
        "dcd": f"{output_prefix}_traj.dcd",
        "log": f"{output_prefix}_md.json",
        "progress": f"{output_prefix}_progress.json",
        "restart": f"{output_prefix}_restart",
    }


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    """Write through a temporary file so a crash never leaves a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def _remove_md_outputs(paths: Dict[str, str]) -> None:
    for key, path in paths.items():
        if key == "restart":
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)


def _artifacts_from_log(paths: Dict[str, str], log: Dict[str, Any], reused: bool) -> MDRunArtifacts:
    return MDRunArtifacts(
        prepared_pdb=paths["prepared"],
        trajectory_npz=paths["npz"],
        trajectory_dcd=paths["dcd"] if os.path.exists(paths["dcd"]) else None,
        log_path=paths["log"],
        dt_ps=float(log["report_interval"]) * float(log["timestep_fs"]) / 1000.0,
        n_frames=int(log["frames"]),
        resumes=int(log.get("resumes", 0)),
        reused=reused,
    )


def _new_integrator(md_config: MDConfig, seed_offset: int = 0):
    integrator = LangevinMiddleIntegrator(
        md_config.temperature_k * unit.kelvin,
        md_config.friction_per_ps / unit.picosecond,
        md_config.timestep_fs * unit.femtoseconds,
    )
    if md_config.random_seed is not None:
        integrator.setRandomNumberSeed(int(md_config.random_seed) + seed_offset)
    return integrator


# --- simulation ---------------------------------------------------------------


def _metal_site_variants(topology, positions, cutoff_ang: float) -> Dict[int, str]:
    """Protonation variants required by metal coordination: ``{residue index: variant}``.

    * cysteine with SG within the cutoff of a metal ion -> ``"CYX"`` (no thiol
      hydrogen; the residue is renamed to the thiolate CYM afterwards);
    * histidine coordinating through NE2 -> ``"HID"`` (proton on ND1), through
      ND1 -> ``"HIE"`` (proton on NE2).
    """
    pos_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    metals = [
        atom.index
        for atom in topology.atoms()
        if atom.element is not None
        and atom.element.symbol.upper() in METALS
        and not is_protein_residue(atom.residue.name)
    ]
    if not metals:
        return {}
    metal_xyz = pos_nm[metals]
    cutoff_nm = float(cutoff_ang) * 0.1

    def near_metal(atom) -> bool:
        return bool(np.min(np.linalg.norm(metal_xyz - pos_nm[atom.index], axis=1)) <= cutoff_nm)

    variants: Dict[int, str] = {}
    for residue in topology.residues():
        name = residue.name.upper()
        atoms = {atom.name: atom for atom in residue.atoms()}
        if name in ("CYS", "CYM") and "SG" in atoms and near_metal(atoms["SG"]):
            variants[residue.index] = "CYX"
        elif name in ("HIS", "HID", "HIE", "HIP"):
            if "NE2" in atoms and near_metal(atoms["NE2"]):
                variants[residue.index] = "HID"
            elif "ND1" in atoms and near_metal(atoms["ND1"]):
                variants[residue.index] = "HIE"
    return variants


def _build_and_equilibrate(
    input_pdb: str,
    ligand_xml: str,
    md_config: MDConfig,
    platform: "Platform",
    paths: Dict[str, str],
):
    """Build the system from the input structure and bring it to the start of production.

    Returns the ``Simulation``.  Writes the prepared PDB, the serialized system
    and the start state.
    """
    forcefield = app.ForceField(*FORCEFIELD_FILES, ligand_xml)
    pdb = app.PDBFile(input_pdb)
    clean_top, clean_pos = _sanitize_topology_for_forcefield(pdb.topology, pdb.positions)
    modeller = app.Modeller(clean_top, clean_pos)
    nonbonded_method = _resolve_nonbonded_method(md_config.nonbonded_method)

    # Residues that hold a metal ion need the protonation state that lets them
    # coordinate it (see MDConfig.metal_site_protonation).
    site_variants: Dict[int, str] = {}
    if md_config.metal_site_protonation:
        site_variants = _metal_site_variants(modeller.topology, modeller.positions, md_config.metal_site_cutoff_ang)
        # Hydrogens that contradict the chosen state must go before new ones are added.
        stale = {"CYX": ("HG",), "HID": ("HE2",), "HIE": ("HD1",)}
        modeller.delete(
            [
                atom
                for residue in modeller.topology.residues()
                if residue.index in site_variants
                for atom in residue.atoms()
                if atom.name in stale[site_variants[residue.index]]
            ]
        )
    variants = (
        [site_variants.get(residue.index) for residue in modeller.topology.residues()] if site_variants else None
    )
    metal_site: Dict[str, str] = {}

    # Hydrogen and ion placement draw from Python's global RNG: seed it so the
    # same inputs always give the same system.  Hydrogens are relaxed on the
    # platform of the run, which makes the whole preparation deterministic
    # whenever that platform is.
    with _seeded_python_random(md_config.random_seed):
        modeller.addHydrogens(forcefield, variants=variants, platform=platform)
        for residue in modeller.topology.residues():
            variant = site_variants.get(residue.index)
            if variant is None:
                continue
            key = f"{residue.name}{residue.id}"
            if variant == "CYX":
                residue.name = "CYM"  # thiolate: no bond partner, charge -1
            metal_site[key] = "CYM" if variant == "CYX" else variant
        if metal_site:
            print(f"[MD] metal-site protonation: {metal_site}")
        if md_config.add_solvent:
            modeller.addSolvent(
                forcefield,
                model=md_config.water_model,
                padding=md_config.solvent_padding_nm * unit.nanometer,
                neutralize=md_config.neutralize,
                ionicStrength=md_config.ionic_strength_m * unit.molar,
                positiveIon=md_config.positive_ion,
                negativeIon=md_config.negative_ion,
            )

    topology = modeller.topology
    create_kwargs = {
        "nonbondedMethod": nonbonded_method,
        "constraints": app.HBonds,
    }
    if nonbonded_method != app.NoCutoff:
        create_kwargs["nonbondedCutoff"] = md_config.nonbonded_cutoff_nm * unit.nanometer
    system = forcefield.createSystem(topology, **create_kwargs)

    barostat = None
    if _is_periodic_nonbonded_method(nonbonded_method) and md_config.pressure_bar > 0 and md_config.barostat_interval > 0:
        barostat = MonteCarloBarostat(
            md_config.pressure_bar * unit.bar,
            md_config.temperature_k * unit.kelvin,
            int(md_config.barostat_interval),
        )
        if md_config.random_seed is not None and hasattr(barostat, "setRandomNumberSeed"):
            barostat.setRandomNumberSeed(int(md_config.random_seed) + 7919)
        system.addForce(barostat)

    integrator = _new_integrator(md_config)
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=md_config.minimization_max_iterations)

    start_temp = max(1.0, float(md_config.heating_initial_temperature_k))
    target_temp = float(md_config.temperature_k)
    if md_config.random_seed is not None:
        simulation.context.setVelocitiesToTemperature(start_temp * unit.kelvin, int(md_config.random_seed))
    else:
        simulation.context.setVelocitiesToTemperature(start_temp * unit.kelvin)

    if md_config.heating_steps > 0 and start_temp < target_temp:
        stages = max(1, int(md_config.heating_stages))
        steps_per_stage = int(md_config.heating_steps) // stages
        remainder = int(md_config.heating_steps) - (steps_per_stage * stages)
        for stage in range(stages):
            frac = float(stage + 1) / float(stages)
            stage_temp = start_temp + frac * (target_temp - start_temp)
            integrator.setTemperature(stage_temp * unit.kelvin)
            if barostat is not None:
                barostat.setDefaultTemperature(stage_temp * unit.kelvin)
            if steps_per_stage > 0:
                simulation.step(steps_per_stage)
        if remainder > 0:
            simulation.step(remainder)
    else:
        integrator.setTemperature(target_temp * unit.kelvin)
        if barostat is not None:
            barostat.setDefaultTemperature(target_temp * unit.kelvin)

    integrator.setTemperature(target_temp * unit.kelvin)
    if barostat is not None:
        barostat.setDefaultTemperature(target_temp * unit.kelvin)
    if md_config.equilibration_steps > 0:
        simulation.step(md_config.equilibration_steps)

    state_eq = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(paths["prepared"], "w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(topology, state_eq.getPositions(), handle, keepIds=True)
    # The exact simulated system and its state at the start of production: with
    # these two files the run can be restarted or replicated without rebuilding.
    with open(paths["system"], "w", encoding="utf-8") as handle:
        handle.write(XmlSerializer.serialize(simulation.context.getSystem()))
    simulation.saveState(paths["start"])
    _write_json_atomic(paths["metal_site"], metal_site)
    return simulation


def _restore_simulation(md_config: MDConfig, platform: "Platform", paths: Dict[str, str], progress: Dict[str, Any]):
    """Recreate the ``Simulation`` of an interrupted or finished run at its last checkpoint.

    The system is read back from disk rather than rebuilt, so the topology is
    guaranteed to match the saved coordinates.  The binary checkpoint restores
    the exact dynamical state (including random-number generators) but only on
    the hardware that wrote it; otherwise the portable XML state is used.
    """
    topology = app.PDBFile(paths["prepared"]).topology
    with open(paths["system"], "r", encoding="utf-8") as handle:
        system = XmlSerializer.deserialize(handle.read())
    resumes = int(progress.get("resumes", 0)) + 1
    integrator = _new_integrator(md_config, seed_offset=7907 * resumes)
    simulation = app.Simulation(topology, system, integrator, platform)
    checkpoint = os.path.join(paths["restart"], progress["checkpoint"])
    state_xml = os.path.join(paths["restart"], progress["state_xml"])
    try:
        simulation.loadCheckpoint(checkpoint)
        restored_from = "checkpoint"
    except Exception:
        simulation.loadState(state_xml)
        restored_from = "state_xml"
    return simulation, resumes, restored_from


def run_openmm_md(
    input_pdb: str,
    ligand_xml: str,
    md_config: MDConfig,
    output_prefix: str,
    resume: bool = True,
) -> MDRunArtifacts:
    """Simulate one system and store its trajectory (restartable).

    Steps: add hydrogens -> (solvate) -> minimise -> staged heating ->
    equilibration -> production.  See the module docstring for the files
    written and for what happens when files from a previous call exist.
    ``resume=False`` discards them and starts from scratch.
    """
    ensure_openmm()
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    paths = _md_paths(output_prefix)
    signature = md_signature(input_pdb, ligand_xml, md_config)
    total_steps = int(md_config.production_steps)

    # --- what is already on disk? ---------------------------------------------------------
    if not resume:
        _remove_md_outputs(paths)
    progress: Optional[Dict[str, Any]] = None
    leftovers = [p for key, p in paths.items() if key != "restart" and os.path.exists(p)]
    if leftovers:
        if os.path.exists(paths["progress"]):
            with open(paths["progress"], "r", encoding="utf-8") as handle:
                progress = json.load(handle)
        elif not os.path.exists(paths["npz"]) and not os.path.exists(paths["log"]):
            # Interrupted while the system was being written, before production
            # started: nothing worth keeping.
            _remove_md_outputs(paths)
            leftovers = []
    if leftovers:
        if progress is None or progress.get("signature") != signature:
            raise RuntimeError(
                f"Files of a previous simulation exist at '{output_prefix}_*' but were produced with different "
                "inputs or settings (or by a version without restart support). Delete them, choose another "
                "output directory, or pass resume=False (--no-resume) to overwrite."
            )
        if int(progress["steps_done"]) > total_steps:
            raise RuntimeError(
                f"The trajectory at '{output_prefix}_*' already has {progress['steps_done']} production steps, "
                f"more than the {total_steps} requested. Refusing to truncate it; pass resume=False to redo it."
            )
        if progress["stage"] == "completed" and int(progress["steps_done"]) == total_steps and os.path.exists(paths["npz"]):
            with open(paths["log"], "r", encoding="utf-8") as handle:
                log = json.load(handle)
            print(f"[MD] {output_prefix}: already complete ({log['frames']} frames), reusing.")
            return _artifacts_from_log(paths, log, reused=True)

    requested_platform_name = str(md_config.platform or "AUTO")
    platform, selected_platform_name, available_platforms = _select_openmm_platform(requested_platform_name)
    if requested_platform_name.strip().upper() == "AUTO":
        print(
            f"[OpenMM] AUTO platform selected '{selected_platform_name}'. "
            f"Available: {', '.join(available_platforms)}"
        )
    os.makedirs(paths["restart"], exist_ok=True)

    report_interval = int(md_config.report_interval)
    periodic = _is_periodic_nonbonded_method(_resolve_nonbonded_method(md_config.nonbonded_method))
    pos_frames: List[np.ndarray] = []
    vel_frames: List[np.ndarray] = []
    box_frames: List[np.ndarray] = []
    time_ps: List[float] = []

    def capture_frame(steps: int) -> None:
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        pos_frames.append(
            np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=np.float64)
        )
        vel_frames.append(
            np.asarray(
                state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
                dtype=np.float64,
            )
        )
        if periodic:
            box_frames.append(
                np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer), dtype=np.float64)
            )
        time_ps.append(steps * md_config.timestep_fs / 1000.0)

    # --- start, continue or extend ----------------------------------------------------------
    if progress is None:
        simulation = _build_and_equilibrate(input_pdb, ligand_xml, md_config, platform, paths)
        steps_done = 0
        parts: List[str] = []
        resumes = 0
        restored_from = None
        capture_frame(0)
    else:
        simulation, resumes, restored_from = _restore_simulation(md_config, platform, paths, progress)
        steps_done = int(progress["steps_done"])
        parts = list(progress["parts"])
        if progress["stage"] == "completed":
            # Extending a finished run: its frames become the first part again.
            part_name = "part_00000.npz"
            shutil.copyfile(paths["npz"], os.path.join(paths["restart"], part_name))
            parts = [part_name]
        print(
            f"[MD] {output_prefix}: continuing from step {steps_done} of {total_steps} "
            f"(restored from {restored_from})."
        )

    def flush(stage: str) -> None:
        """Persist the frames collected since the last flush together with a checkpoint."""
        nonlocal parts
        if pos_frames:
            part_name = f"part_{len(parts):05d}.npz"
            tmp = os.path.join(paths["restart"], f"writing_{part_name}")
            arrays = {
                "positions": np.stack(pos_frames, axis=0),
                "velocities": np.stack(vel_frames, axis=0),
                "time_ps": np.asarray(time_ps, dtype=np.float64),
            }
            if box_frames:
                arrays["box_nm"] = np.stack(box_frames, axis=0)
            np.savez(tmp, **arrays)
            os.replace(tmp, os.path.join(paths["restart"], part_name))
            parts = parts + [part_name]
            pos_frames.clear()
            vel_frames.clear()
            box_frames.clear()
            time_ps.clear()
        checkpoint = f"step_{steps_done:012d}.chk"
        state_xml = f"step_{steps_done:012d}.xml"
        simulation.saveCheckpoint(os.path.join(paths["restart"], f"writing_{checkpoint}"))
        os.replace(os.path.join(paths["restart"], f"writing_{checkpoint}"), os.path.join(paths["restart"], checkpoint))
        simulation.saveState(os.path.join(paths["restart"], state_xml))
        # The progress file is written last: it only ever points at complete files.
        _write_json_atomic(
            paths["progress"],
            {
                "signature": signature,
                "stage": stage,
                "steps_done": steps_done,
                "production_steps_requested": total_steps,
                "parts": parts,
                "checkpoint": checkpoint,
                "state_xml": state_xml,
                "resumes": resumes,
            },
        )
        for name in os.listdir(paths["restart"]):
            if name.startswith("step_") and name not in (checkpoint, state_xml):
                os.remove(os.path.join(paths["restart"], name))

    if progress is None:
        flush("production")  # equilibration is never repeated after this point

    # --- production -----------------------------------------------------------------------------
    checkpoint_interval = int(md_config.checkpoint_interval_steps)
    last_flush = steps_done
    while steps_done < total_steps:
        step = total_steps - steps_done if report_interval <= 0 else min(report_interval, total_steps - steps_done)
        simulation.step(step)
        steps_done += step
        capture_frame(steps_done)
        if checkpoint_interval > 0 and steps_done - last_flush >= checkpoint_interval and steps_done < total_steps:
            flush("production")
            last_flush = steps_done
    flush("production")

    # --- assemble the trajectory ------------------------------------------------------------------
    loaded: List[Dict[str, np.ndarray]] = []
    for name in parts:
        with np.load(os.path.join(paths["restart"], name)) as part:
            loaded.append({key: part[key] for key in part.files})
    positions = np.concatenate([part["positions"] for part in loaded], axis=0)
    velocities = np.concatenate([part["velocities"] for part in loaded], axis=0)
    time_ps_arr = np.concatenate([part["time_ps"] for part in loaded], axis=0)
    boxes = (
        np.concatenate([part["box_nm"] for part in loaded], axis=0)
        if loaded and all("box_nm" in part for part in loaded)
        else None
    )
    del loaded
    dt_ps = md_config.report_interval * md_config.timestep_fs / 1000.0
    arrays = {
        "positions": positions,
        "velocities": velocities,
        "time_ps": time_ps_arr,
        "dt_ps": np.asarray([dt_ps], dtype=np.float64),
    }
    if boxes is not None:
        arrays["box_nm"] = boxes
    tmp_npz = f"{output_prefix}_traj.writing.npz"
    np.savez_compressed(tmp_npz, **arrays)
    os.replace(tmp_npz, paths["npz"])

    topology = simulation.topology
    dcd_path: Optional[str] = None
    if total_steps > 0 and report_interval > 0:
        # Same frames as the npz (unwrapped coordinates, so protein and ligand
        # stay in the same periodic image), frame i of the DCD = frame i of the latents.
        dcd_path = paths["dcd"]
        with open(dcd_path, "wb") as handle:
            dcd = app.DCDFile(handle, topology, md_config.timestep_fs * unit.femtoseconds, 0, report_interval)
            for index in range(positions.shape[0]):
                box = None if boxes is None else [Vec3(*row) for row in boxes[index]] * unit.nanometer
                dcd.writeModel(positions[index], periodicBoxVectors=box)

    metal_site: Dict[str, str] = {}
    if os.path.exists(paths["metal_site"]):
        with open(paths["metal_site"], "r", encoding="utf-8") as handle:
            metal_site = json.load(handle)

    water_residues = 0
    ion_residues = 0
    for residue in topology.residues():
        resname = residue.name.upper()
        if resname in WATER_RESNAMES:
            water_residues += 1
            continue
        atoms = list(residue.atoms())
        if len(atoms) == 1 and not is_protein_residue(resname):
            ion_residues += 1

    state_box = simulation.context.getState()
    box_vectors = state_box.getPeriodicBoxVectors()
    box_nm = None
    if box_vectors is not None:
        try:
            box_nm_arr = np.asarray(box_vectors.value_in_unit(unit.nanometer), dtype=float)
        except Exception:
            box_nm_arr = np.asarray(
                [[float(vec.x), float(vec.y), float(vec.z)] for vec in box_vectors],
                dtype=float,
            )
        box_nm = box_nm_arr.tolist()

    log_payload = {
        "input_pdb": input_pdb,
        "prepared_pdb": paths["prepared"],
        "trajectory_dcd": dcd_path,
        "ligand_xml": ligand_xml,
        "temperature_k": md_config.temperature_k,
        "heating_initial_temperature_k": md_config.heating_initial_temperature_k,
        "heating_steps": md_config.heating_steps,
        "heating_stages": md_config.heating_stages,
        "timestep_fs": md_config.timestep_fs,
        "pressure_bar": md_config.pressure_bar,
        "barostat_interval": md_config.barostat_interval,
        "platform_requested": requested_platform_name,
        "platform_selected": selected_platform_name,
        "platforms_available": available_platforms,
        "nonbonded_method": md_config.nonbonded_method,
        "nonbonded_cutoff_nm": md_config.nonbonded_cutoff_nm,
        "add_solvent": md_config.add_solvent,
        "water_model": md_config.water_model,
        "solvent_padding_nm": md_config.solvent_padding_nm,
        "ionic_strength_m": md_config.ionic_strength_m,
        "neutralize": md_config.neutralize,
        "positive_ion": md_config.positive_ion,
        "negative_ion": md_config.negative_ion,
        "report_interval": md_config.report_interval,
        "production_steps": md_config.production_steps,
        "equilibration_steps": md_config.equilibration_steps,
        "frames": int(positions.shape[0]),
        "num_atoms": int(topology.getNumAtoms()),
        "num_residues": int(sum(1 for _ in topology.residues())),
        "water_residues": int(water_residues),
        "ion_like_residues": int(ion_residues),
        "box_vectors_nm": box_nm,
        # --- reproducibility / restart -----------------------------------------
        "random_seed": md_config.random_seed,
        "metal_site_protonation": metal_site,
        "signature": signature,
        "system_xml": paths["system"],
        "start_state_xml": paths["start"],
        "checkpoint_interval_steps": checkpoint_interval,
        "resumes": resumes,
        "last_restored_from": restored_from,
        "completed": True,
    }
    _write_json_atomic(paths["log"], log_payload)

    # Mark as complete, then drop the parts: the final file now holds every frame.
    # The last checkpoint stays so the run can be extended later.
    finished_parts = parts
    parts = []
    flush("completed")
    for name in finished_parts:
        os.remove(os.path.join(paths["restart"], name))

    return MDRunArtifacts(
        prepared_pdb=paths["prepared"],
        trajectory_npz=paths["npz"],
        trajectory_dcd=dcd_path,
        log_path=paths["log"],
        dt_ps=dt_ps,
        n_frames=int(positions.shape[0]),
        resumes=resumes,
        reused=False,
    )
