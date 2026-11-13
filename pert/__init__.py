"""
PERT - how does a perturbation (mutation) reach a protein's active site?

The package compares a wild-type (WT) protein with a mutant (MUT) and
quantifies how the mutation changes the binding/active site, including
mutations that change the number of residues (insertions, deletions,
duplications).

Layers, from low to high level (each one only imports the layers above it in
this list):

==================  =========================================================
``pert.utils``      file I/O, run-directory paths, numerics
``pert.sequence``   mutation specs, sequence alignment
``pert.structure``  residues, binding site, modelling, ligand transfer
``pert.simulation`` OpenMM molecular dynamics
``pert.analysis``   field embedding, VAMP latents, ddG, TE, susceptibility
``pert.visualization`` figures and VMD scenes
``pert.allostery``  fast scan, MD ranking, consensus, evidence
``pert.benchmark``  dataset normalisation and scanner benchmarks
``pert.reporting``  run reports
``pert.pipelines``  end-to-end orchestration
``pert.cli``        command-line entry points (``python -m pert <command>``)
==================  =========================================================
"""

__version__ = "0.2.0"
