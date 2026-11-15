# Aftershock — how does a perturbation reach the active site?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/aftershock/blob/main/notebooks/aftershock_colab.ipynb)

*Repository `aftershock`; the Python package and command are `pert`.*

PERT compares a wild-type protein (WT) with a mutant (MUT) and quantifies how
the mutation changes the **binding/active site**: its conformations, its
coupling to the rest of the protein, and which residues carry the effect.

It works for substitutions **and** for mutations that change the number of
residues (insertions, deletions, delins, duplications), because the comparison
never needs a one-to-one atom mapping between WT and MUT: only a few conserved
residues of the site are matched, to put both proteins in the same frame.

There are two tiers:

| tier | cost | question it answers | entry point |
|---|---|---|---|
| **Fast scan** | seconds, no MD | Is this variant worth simulating? Which residues look affected? | `pert fast-scan`, `pert fast-panel` |
| **MD pipeline** | hours (GPU) | How do the site's ensemble and its coupling to the protein change? | `pert pipeline`, `pert md-panel` |

Fast-scan results are hypotheses; claims should rest on replicated MD
(see [docs/BENCHMARK_DATASETS.md](docs/BENCHMARK_DATASETS.md)).

## Repository layout

```
pert/               Python package (see "Architecture")
configs/            JSON configurations for runs and panels
data/toy/           toy system: zinc finger 1ZNF + ligand force-field XML
data/benchmarks/    benchmark tables (smoke examples, template)
scripts/            shell/python helpers (full 20 ns toy workflow, data prep)
notebooks/          Google Colab notebook that runs a panel on a cloud GPU
tests/              pytest suite (pure-Python kernels + structure smoke test)
docs/               benchmarking guides, result notes
runs/               ALL generated output (one directory per run or panel)
cpp/                incremental C++ port of the mutation and TE kernels
archive/            the original flat scripts, for reference
```

## Installation

Dependencies: Python >= 3.10, NumPy, SciPy, scikit-learn, matplotlib,
OpenMM 8 and MODELLER (licence key required; only needed to *build* mutants).

```bash
conda env create -f environment.yml     # creates the "pert" environment
conda activate pert
pip install -e .                        # optional: gives the `pert` command
```

Without `pip install`, run everything from the repository root with
`python -m pert <command>`.

## Quick start

```bash
python -m pert                          # list commands
python -m pert <command> --help         # options of one command

# Fast tier: score existing WT/MUT structures of a finished run
python -m pert fast-scan --run-dir runs/mutation_panel_1znf_smoke1ns_dry/snp_k2r/rep1

# Fast tier: build + score a custom panel (needs MODELLER)
python -m pert fast-panel --mutation-set 'combo:K2R,10_11delinsG,Q20dup' --output-dir runs/my_fast_panel

# MD tier: one run from a JSON config
python -m pert pipeline-config --config configs/pipeline_config.example.json

# MD tier: a panel of your own scenarios (any protein), see "Running your own protein"
python -m pert md-panel --mutation-set k2r:K2R --mutation-set double:K2R,Q20dup --replicas 2 --output-dir runs/my_panel

# MD tier: the whole workflow from a config (panel -> plots -> ranking -> consensus -> reports)
bash scripts/run_panel_workflow.sh configs/custom_panel.example.json

pytest                                  # run the test suite
```

Mutation syntax (reference numbering, case-insensitive): `K2R`, `A10del`,
`10_12del`, `R8_S9insA`, `10_11delinsG` (alias `insdel`), `Q20dup`, `10_12dup`.
Several mutations can be combined; positions always refer to the WT sequence.

## Commands

| command | purpose |
|---|---|
| `pipeline`, `pipeline-config` | one WT-vs-MUT comparison: build, MD, analysis |
| `md-panel`, `md-panel-config` | scenarios x replicas through the pipeline |
| `analyze-run` | standard plots, binding-site table, residue-flow network, VMD scenes |
| `diagnostics` | convergence/time-course plots, interpretation summary |
| `rank` | MD-based allosteric residue/region ranking of one run |
| `null-control` | calibrate a panel against pairs of its own WT simulations (no mutation) |
| `site-check` | does the metal site stay coordinated in every simulation of a panel? |
| `consensus` | replicate consensus (+ optional control scenario) of a panel |
| `evidence` | conservative evidence report of a panel |
| `report` | plain-language Markdown report of one run |
| `fast-scan`, `fast-panel` | MD-free structural/allosteric screening |
| `normalize-benchmark`, `benchmark` | prepare external datasets and benchmark the fast scanner |

## The algorithm

### Stage 1 — build the structure pair (`pert.structure`)

1. The mutations are applied to the WT sequence (`sequence.mutations`).
2. MODELLER builds models of the mutant on the template; the best DOPE model is
   kept (`structure.homology`).
3. The ligand is copied from the WT into the mutant after superposing only the
   C-alpha atoms around the site, matched through affine-gap sequence alignment
   (`structure.ligand`).

### Stage 2 — molecular dynamics (`pert.simulation`)

WT and mutant are simulated with the same OpenMM protocol (Amber14, optional
explicit solvent, minimisation, staged heating, equilibration, production).
Positions **and velocities** are stored. The stage is restartable and seeded
(see "Restarting and reproducing runs").

### Stage 3 — topology-safe comparison (`pert.analysis`)

The problem: WT and MUT may have different atoms, so coordinates cannot be
subtracted. The solution is to describe both in a common space.

1. **Shared site frame** (`site_frame`). WT and MUT are both superposed, frame
   by frame, on the same scaffold: the C-alpha atoms of the residues that line
   the ligand in the WT structure and exist in both proteins (paired by
   sequence alignment). The origin is the ligand position in the WT reference.
   Rotating or translating a simulation therefore changes nothing. This is the
   only place where a WT↔MUT correspondence is used.
2. **Density fields** (`pocket_fields`). Around the origin, three regions are
   defined: the pocket `B` (protein atoms within `rb`), the protein context `P`
   (between `rb` and `rp`) and the ligand `L`. In every MD frame each region is
   converted into Gaussian density (and velocity) fields on a fixed grid, one
   channel per element class. The feature vector has the same meaning for WT
   and MUT whatever the mutation.
3. **Slow coordinates** (`vamp`). A linear VAMP projection, fitted on WT and MUT
   together, reduces each region to a few slow latent coordinates
   `u_B`, `u_P`, `u_L`. The solver drops features that never change and works
   in the span of the data, which gives the same result as the textbook
   computation at a fraction of the cost. Each result records the singular
   values and an `overfit_risk` flag (fewer than 5 lagged pairs per effective
   dimension): with too few frames almost any direction looks slow.
4. **What changed at the site?** Estimands computed from the latents:

   | estimand | module | meaning |
   |---|---|---|
   | basin population shift | `ensemble` | total-variation distance between the WT and MUT populations of the site's conformational basins (0 = same, 1 = disjoint), block-bootstrap CI. For basins visited by both, `ΔΔG = -kT ln(p_mut/p_wt)`; a basin seen in only one system has no ΔΔG (`overlap = false`) |
   | separability | `ensemble` | can a classifier tell WT frames from MUT frames it has not seen? AUC (0.5 = no, 1 = fully) and Jensen–Shannon divergence, cross-validated on contiguous time blocks; plus KL |
   | conditional transfer entropy | `transfer_entropy` | `TE(P → B | L)`: does the past of the surroundings predict the site beyond the site's and ligand's own past? Linear-Gaussian (Granger) estimator with model selection, surrogate p-value and reverse-direction control; `ΔTE = TE_mut − TE_wt` |
   | susceptibility | `susceptibility` | `Γ = Cov(u_B,u_P) Cov(u_P)⁻¹`, the static response of the site to its context; `‖ΔΓ‖_F` measures rewiring |

5. **Is it more than noise?** (`pipelines.null_control`). Two finite simulations
   of the *same* protein also differ. Every run of a panel contains its own WT
   simulation, so the panel compares its WT trajectories with each other, with
   the same analysis, to obtain the null distribution of each metric; every
   WT-vs-MUT result is then reported with a z-score and an empirical p-value
   against it, and each scenario gets an evidence level (`none`, `suggestive`,
   `supported`). `pert md-panel` does this automatically; `pert null-control`
   does it for any finished panel. A panel needs at least 7 WT trajectories
   (e.g. 4 scenarios x 2 replicas) for a single run to reach p <= 0.05.

   Exceeding every WT-vs-WT pair is necessary, not sufficient. A metric
   *counts* only if it is not saturated in the null (two WT simulations that
   already reach a total-variation distance of 0.999 leave nothing to exceed)
   and if every replica also exceeds the modelled-WT control, when the panel
   has one: a scenario whose "mutant" has the WT sequence (`wt_model:K2K`) is
   recognised as that control. `supported` further requires converged
   simulations. `pert null-control --recalibrate` re-derives the verdicts of a
   finished panel from its stored pairs, without the trajectories.

6. **Did the site survive?** (`analysis.metal_site`, `pert site-check`). Metal
   ions are non-bonded particles, so a site can lose a ligand during a
   simulation. The check follows the distance of every initial ligating atom
   to the ion, and the water that gets in, through each trajectory and labels
   the site `intact`, `distorted` or `lost`. If sites break in some
   simulations, the comparison measures that and not the mutation.

Sampling is reported, not assumed: a trajectory that never changes basin counts
as one effective sample, and `sampling_ok` is false below 10.

### Stage 4 — which residues carry the effect? (`pert.analysis.residue_flow`, `pert.allostery`)

1. WT and MUT residues are paired by sequence alignment; each paired residue
   gets a motion signal (displacement + speed) per frame.
2. A pairwise transfer-entropy proxy gives a directed residue network for WT
   and MUT; its difference is searched for routes from the **sources**
   (residues displaced by the mutation) to the **targets** (binding-site
   residues).
3. `allostery.ranking` combines structural shift, dynamics change, contact
   change, TE change, relay role and site coupling into a per-residue score.
4. `allostery.consensus` aggregates replicas (support, bootstrap CI, control
   baseline) and `allostery.evidence` grades the result conservatively.

### Fast tier (`pert.allostery.fast_scan`)

Uses only the two structures: structural shift after superposition, gained and
lost C-alpha contacts, local strain, graph proximity to both the mutation and
the site, pocket proximity. The weighted sum is the `fast_allostery_score`.

## Architecture

Layers, low to high; a module only imports from layers above it in this list:

```
pert/
  constants.py  config.py  deps.py     vocabularies, typed configs, optional imports
  utils/         io, run-directory paths, numerics, matplotlib setup
  sequence/      mutations, alignment
  structure/     residues, binding_site, comparison, geometry, ligand, homology, prepare
  simulation/    md (OpenMM), trajectory
  analysis/      site_frame, pocket_fields, vamp, ensemble, transfer_entropy, susceptibility,
                 statistics, residue_flow, topology_safe
  visualization/ figures and generated VMD scenes
  allostery/     fast_scan, ranking, consensus, evidence
  benchmark/     normalize, metrics, fast_benchmark, plots
  reporting/     reader_report, interpretation
  pipelines/     scenarios, single_run, md_panel, null_control, fast_panel, post_analysis
  cli/           one thin module per command
```

### Extending

* **New estimand**: add a module in `pert/analysis/`, call it from
  `analysis/topology_safe.py::run_topology_safe_analysis` and add its key to
  the results dict.
* **New residue-level evidence**: add the per-residue array in
  `allostery/ranking.py::run_allostery_ranking` and give it a weight.
* **New command**: add `pert/cli/<name>.py` with a `main()` and one line in
  `pert/cli/__init__.py::COMMANDS`.
* **New system**: see "Running your own protein". Nothing in the package is specific to 1ZNF
  except the built-in toy scenarios and the default paths.

## Run directory

```
runs/<panel>/<scenario>/<replicate>/
  run_summary.json        index of the run (entry point of all post-processing)
  structures/             wt_with_ligand.pdb, mutant_raw.pdb, mutant_with_ligand.pdb
  model_build/            MODELLER working files
  structures/build.json   settings the structures were built with
  md/                     *_prepared.pdb, *_traj.npz (pos+vel), *_traj.dcd, *_md.json
                          *_system.xml, *_start.xml (simulated system and its starting state)
                          *_progress.json, *_restart/ (checkpoint; trajectory parts while running)
  analysis/               analysis_results.json, latents.npz, plots, rankings, VMD scenes
```

`run_summary.json` also records the complete configuration of the run, the
software versions, the command line and the date.

Run directories can be moved: recorded paths are re-resolved relative to the
run directory (`pert.utils.paths.resolve_run_path`).

## Running your own protein

A panel needs five things besides the mutations:

| input | option | notes |
|---|---|---|
| WT structure with the ligand bound | `--reference-pdb` | residue numbering here is the numbering of the mutations |
| template structure for MODELLER | `--template-pdb`, `--template` | `--template` is the code + chain, e.g. `2ITYA`; usually the WT structure itself |
| ligand force field | `--ligand-xml` | OpenMM XML with the ligand parameters |
| ligand residue name | `--ligand-resname` | defines the site |
| protein chain | `--chain-id` | default `A` |

Scenarios are named mutation sets, given on the command line
(`--mutation-set ex19del:L747_A750del`, repeatable), in a JSON/CSV file
(`--mutation-file`), or in the panel config (`"mutation_sets": {"name": [...]}`,
see [configs/custom_panel.example.json](configs/custom_panel.example.json)).
`--scenarios a,b` runs only some of them.

Before anything is built the panel checks that the files exist, that the
ligand is in the reference structure, that the chain is numbered consecutively
and that every mutation fits the reference sequence, and stops with the name of
the offending scenario.

The reference chain must have no missing residues or insertion codes: mutation
positions are mapped by counting from the first residue, so a gap would shift
every later position. Model the gaps first.

Metal sites: residues that coordinate a metal ion are given the protonation
state that makes coordination possible (cysteine thiolate, histidine protonated
on the far nitrogen; `metal_site_protonation`, on by default). The ion itself is
a plain charged sphere without bonds to the protein, which holds the zinc site
of 1ZNF together but with Zn-S distances about 0.4 A shorter than in the
experimental structure (1.93 vs 2.30 A). Organic ligands need their own
force-field XML.

Design advice:

* **Replicas and scenarios.** The null control needs at least 7 WT
  trajectories (scenarios x replicas) for a run to reach p <= 0.05.
* **A modelled-WT control.** A silent substitution (`wt_model:K2K`) makes
  MODELLER rebuild the unmutated protein. Every mutant is a model while the WT
  is the experimental structure, so this scenario shows how much of an effect
  comes from model building alone. The null control uses it: a metric counts
  for a mutant only above what this scenario reaches.
* **Growing a panel.** Seeds depend on the scenario name, not on its position:
  adding or removing scenarios and re-running reuses what is already simulated.

## Running on Google Colab

[`notebooks/aftershock_colab.ipynb`](notebooks/aftershock_colab.ipynb) runs a
panel on a Colab GPU and keeps results and checkpoints on Google Drive:

<https://colab.research.google.com/github/juvenalyosa/aftershock/blob/main/notebooks/aftershock_colab.ipynb>

It installs OpenMM (CUDA wheels) and MODELLER (Debian package) without conda
(`scripts/colab_install.sh`), runs a short test of the whole chain, then
starts `scripts/run_panel_workflow.sh` in the background and follows its log
(`scripts/colab_tools.py`). A session that disconnects loses nothing: running
the notebook again resumes every simulation from its last checkpoint. It needs
a MODELLER licence key.

The same scripts work on any Linux machine without conda: set
`PERT_PYTHON=/path/to/python` and call `scripts/run_panel_workflow.sh`.

## Restarting and reproducing runs

Launching the same command again is always safe and is how a run is resumed:

| what is on disk | what happens |
|---|---|
| structures built with the same settings | reused (MODELLER is not called again) |
| a complete simulation | reused |
| an interrupted simulation | continues from its last checkpoint (written every `checkpoint_interval_steps`, default 250 000) |
| a complete simulation, and `production_steps` was increased | extended |
| the analysis | always recomputed (it is cheap) |
| files produced with *different* settings | error; nothing is overwritten unless `--no-resume` is given |

So a panel killed at hour 10 loses at most one checkpoint interval of one
simulation, and changing only analysis settings re-analyses existing
trajectories without re-simulating.

Reproducibility: hydrogens and ions are placed from `random_seed`, so the same
inputs give the same system atom for atom. On the Reference platform two
launches give bit-identical trajectories, also across an interruption. On
CPU/OpenCL the last digits of the starting coordinates differ and the dynamics
diverge (CUDA was not tested); what is guaranteed there is the same system and
the saved starting state (`*_system.xml`, `*_start.xml`).

## Known limitations

* **Results from before the shared site frame are not comparable with new
  ones.** The original frame was referenced to each simulation's own first
  frame, so the orientation of a simulation leaked into every estimand, and the
  in-sample JS divergence and the ±16.7 kcal/mol ΔΔG of non-overlapping basins
  were saturation artefacts. `pert null-control` re-analyses such panels with
  the current method (without touching the runs); `frame_mode: "legacy_self"`
  together with the latent `solver: "full"` reproduces the old numbers exactly
  if needed.
* **Short simulations are not converged.** On the toy zinc finger, 16
  simulations of 20 ns (`configs/znf_panel_20ns.json`) gave 3 to 7 effective
  samples per trajectory, two WT simulations were told apart with a median AUC
  of 0.92, and no scenario was distinguishable from the null and the
  modelled-WT control. Check `sampling_ok` and the null before reading any
  difference.
* The null control calibrates the region-level estimands only; the per-residue
  ranking and the fast scan are not yet calibrated against a null.
* Trajectories are bit-reproducible only on deterministic platforms (see
  "Restarting and reproducing runs"). On GPUs two launches of the same run are
  two samples of the same ensemble: use replicas, the null control and the
  consensus layer rather than expecting identical numbers.
* Trajectories are still assembled in memory and stored for all atoms
  (solvent included) in double precision.
* The toy runs shipped in `runs/` are short and were simulated before
  metal-site protonation existed: their zinc site is half dissociated (both
  cysteines leave the ion within picoseconds). They exercise the pipeline and
  say nothing about the real site. `configs/znf_panel_20ns.json` is the panel
  to run instead.
