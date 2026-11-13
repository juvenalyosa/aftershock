"""Typed configuration and result records for the WT-vs-MUT pipeline.

Every stage of the pipeline receives one of these dataclasses, so a run is
fully described by ``BuildConfig`` + ``MDConfig`` + ``FieldConfig`` +
``LatentConfig`` + ``TEConfig``.  The JSON files in ``configs/`` map one-to-one
onto these classes (see ``pert.cli.pipeline_from_config``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


# --- stage configurations --------------------------------------------------


@dataclass
class BuildConfig:
    """Inputs of the structure-building stage.

    Attributes:
        reference_pdb: WT structure that contains the ligand.
        ligand_xml: OpenMM force-field XML with the ligand parameters.
        ligand_resnames: residue names that define the ligand / site centre.
        mutations: mutation specs in reference numbering (see ``pert.sequence.mutations``).
        templates: MODELLER template codes, PDB id + chain (e.g. ``1ZNFA``).
        chain_id: protein chain used for sequence extraction and alignment.
        original_sequence: WT sequence; read from ``reference_pdb`` when omitted.
        start_pos: residue number of the first residue; inferred when omitted.
        num_models: number of MODELLER models to build (best DOPE is kept).
        num_processors: MODELLER parallel workers.
        modeller_seed: seed passed to MODELLER's random number generator.
    """
    reference_pdb: str
    ligand_xml: str
    ligand_resnames: List[str]
    mutations: List[str]
    templates: List[str]
    chain_id: Optional[str] = "A"
    original_sequence: Optional[str] = None
    start_pos: Optional[int] = None
    num_models: int = 50
    num_processors: int = 4
    modeller_seed: Optional[int] = None


@dataclass
class MDConfig:
    """Molecular-dynamics protocol (OpenMM).

    Lengths are given in integration steps of ``timestep_fs``; a frame
    (positions + velocities) is stored every ``report_interval`` steps.
    ``platform`` may be AUTO, CUDA, OpenCL, CPU or Reference.  WT uses
    ``random_seed`` and the mutant ``random_seed + 1``; the seed also fixes the
    placement of hydrogens and ions.  During production a restart point is
    written every ``checkpoint_interval_steps`` (0 = only at the end).

    ``metal_site_protonation`` (default on) gives residues that coordinate a
    metal ion the protonation state that makes coordination possible: a
    cysteine whose sulfur is within ``metal_site_cutoff_ang`` of a metal becomes
    a thiolate (CYM) and a histidine is protonated on the nitrogen that does
    *not* face the metal.  Without it a zinc site falls apart within
    picoseconds, because a neutral thiol does not bind the ion.
    """
    temperature_k: float = 300.0
    friction_per_ps: float = 1.0
    timestep_fs: float = 2.0
    pressure_bar: float = 1.0
    barostat_interval: int = 25
    nonbonded_method: str = "PME"
    nonbonded_cutoff_nm: float = 1.0
    add_solvent: bool = True
    water_model: str = "tip3p"
    solvent_padding_nm: float = 1.0
    ionic_strength_m: float = 0.15
    neutralize: bool = True
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"
    minimization_max_iterations: int = 500
    heating_initial_temperature_k: float = 50.0
    heating_steps: int = 5000
    heating_stages: int = 10
    equilibration_steps: int = 5000
    production_steps: int = 50000
    report_interval: int = 100
    platform: str = "CPU"
    random_seed: Optional[int] = None
    checkpoint_interval_steps: int = 250000
    metal_site_protonation: bool = True
    metal_site_cutoff_ang: float = 3.0


@dataclass
class FieldConfig:
    """Pocket-frame density-field embedding.

    Attributes:
        rb_ang: radius of the binding-pocket region B around the site centre (A).
        rp_ang: outer radius of the protein-context shell P (A).
        grid_spacing_ang: spacing of the cubic grid clipped to the sphere ``rp_ang``.
        sigma_ang: width of the Gaussian kernel that smears atoms onto the grid.
        max_grid_points: cap on grid size (points are sub-sampled evenly above it).
        include_velocity: add velocity-weighted channels to the occupancy channels.
        remove_com: remove centre-of-mass translation and drift first.
        density_normalize: divide fields by the number of atoms in the region.
        append_type_counts: append per-element atom counts to the feature vector.
        frame_mode: ``"shared_site"`` (default) superposes WT and mutant on the
            same site scaffold so their fields share one coordinate frame;
            ``"legacy_self"`` reproduces the original behaviour (each system
            referenced to its own first frame, not rotation-invariant).
        frame_site_cutoff_ang: residues with a heavy atom this close to the
            ligand in the WT structure anchor the shared frame.
        frame_min_anchors: minimum number of anchor residues (the nearest
            residues are added when the site shell has fewer).
    """
    rb_ang: float = 6.0
    rp_ang: float = 12.0
    grid_spacing_ang: float = 1.2
    sigma_ang: float = 1.2
    max_grid_points: int = 512
    include_velocity: bool = True
    remove_com: bool = True
    density_normalize: bool = True
    append_type_counts: bool = True
    frame_mode: str = "shared_site"
    frame_site_cutoff_ang: float = 6.0
    frame_min_anchors: int = 6


@dataclass
class LatentConfig:
    """Dimensionality reduction (linear VAMP).

    Attributes:
        lag: lag time in frames used to define "slow" motions.
        m_b, m_p, m_l: number of latent coordinates kept for pocket (B),
            protein context (P) and ligand (L).
        reg: covariance regularisation.
        solver: ``"auto"`` (default) solves the same problem in the smallest
            exact subspace; ``"full"`` is the original full-dimensional
            computation (identical numbers to old results, much slower).
    """
    lag: int = 5
    m_b: int = 10
    m_p: int = 10
    m_l: int = 6
    reg: float = 1e-6
    solver: str = "auto"


@dataclass
class TEConfig:
    """Conditional transfer-entropy estimation.

    Attributes:
        max_order: largest VAR order (number of lags) tried.
        ridge_values: ridge penalties tried by blocked cross-validation.
        surrogates: number of phase-randomised surrogates for the p-value.
        whiteness_lag, whiteness_threshold: residual autocorrelation check.
        random_seed: seed of surrogates and bootstraps.
    """
    max_order: int = 6
    ridge_values: Tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2)
    surrogates: int = 64
    whiteness_lag: int = 10
    whiteness_threshold: float = 0.2
    random_seed: int = 13


# --- stage outputs ----------------------------------------------------------


@dataclass
class PreparedStructures:
    """Output of ``prepare_structures``: file paths plus alignment/ligand-transfer diagnostics."""
    wt_pdb: str
    mutant_raw_pdb: str
    mutant_ligand_pdb: str
    original_sequence: str
    mutated_sequence: str
    start_pos: int
    alignment_pairs: int
    local_alignment_pairs: int
    local_alignment_rmsd_ang: float
    ligand_atoms: int


@dataclass
class MDRunArtifacts:
    """Files written by one MD simulation and its sampling interval."""
    prepared_pdb: str
    trajectory_npz: str
    trajectory_dcd: Optional[str]
    log_path: str
    dt_ps: float
    n_frames: int
    resumes: int = 0  # how many times the simulation was continued from a checkpoint
    reused: bool = False  # True when a complete simulation was found and nothing was run
