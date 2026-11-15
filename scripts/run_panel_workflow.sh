#!/usr/bin/env bash
# Full MD workflow for a panel described by a JSON config (any protein):
#
#   1. md-panel-config  build + MD + topology-safe analysis for every scenario/replica
#   2. site-check       does the metal site stay coordinated in every simulation?
#   3. analyze-run      standard plots, binding-site table, residue-flow network
#   4. rank             per-run allosteric residue/region ranking
#   5. consensus        replicate consensus across the panel
#   6. evidence         conservative evidence report
#   7. report           plain-language report per run
#
# Usage:  bash scripts/run_panel_workflow.sh path/to/panel_config.json
#
# The output directory and the ligand residue name are read from the config.
# Re-running the same command resumes an interrupted panel.
#
# Environment overrides: CONDA_ENV_NAME, BINDING_CUTOFF_ANG, CONTROL_SCENARIO
# (name of a scenario to use as baseline in the consensus, e.g. a silent
# "wt_model" scenario), and PERT_PYTHON (path of a Python interpreter to use
# directly instead of a conda environment, e.g. on Google Colab).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-bgt-openmm82}"
BINDING_CUTOFF_ANG="${BINDING_CUTOFF_ANG:-6.0}"
CONTROL_SCENARIO="${CONTROL_SCENARIO:-}"
PERT_PYTHON="${PERT_PYTHON:-}"

if [[ -z "${CONFIG_PATH}" || ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: '${CONFIG_PATH}'" >&2
  echo "Usage: bash scripts/run_panel_workflow.sh path/to/panel_config.json" >&2
  exit 1
fi
CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"

# All relative paths (configs, data/, runs/) are relative to the repository root.
cd "${REPO_ROOT}"

run_python() {
  if [[ -n "${PERT_PYTHON}" ]]; then
    "${PERT_PYTHON}" "$@"
  else
    conda run --no-capture-output -n "${CONDA_ENV_NAME}" python "$@"
  fi
}

pert() {
  run_python -m pert "$@"
}

# One source of truth: where the panel writes and which ligand defines the site.
config_value() {
  run_python -c '
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg = cfg.get("panel", cfg)
print(cfg.get(sys.argv[2], sys.argv[3]))' "${CONFIG_PATH}" "$1" "$2"
}
OUTPUT_DIR="$(config_value output_dir runs/mutation_panel_1znf)"
LIGAND_RESNAME="$(config_value ligand_resname ZN)"
echo "Panel output: ${OUTPUT_DIR}   ligand: ${LIGAND_RESNAME}"

CONTROL_ARGS=()
if [[ -n "${CONTROL_SCENARIO}" ]]; then
  CONTROL_ARGS=(--control-scenario "${CONTROL_SCENARIO}")
fi

echo "[1/7] Running the MD panel (build, MD, analysis, null control)..."
pert md-panel-config --config "${CONFIG_PATH}"

echo "[2/7] Checking that the metal site holds together in every simulation..."
# A panel whose ligand is not a metal ion has nothing to check, and says so.
pert site-check --panel-dir "${OUTPUT_DIR}" --ligand-resname "${LIGAND_RESNAME}"

RUN_DIRS=()
while IFS= read -r run_dir; do
  RUN_DIRS+=("${run_dir}")
done < <(find "${OUTPUT_DIR}" -mindepth 2 -maxdepth 2 -type d -name "rep*" | sort)
if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "No run directories found under ${OUTPUT_DIR}" >&2
  exit 2
fi

echo "[3/7] Building per-run flow artifacts and plots..."
for RUN_DIR in "${RUN_DIRS[@]}"; do
  echo "  - ${RUN_DIR}"
  pert analyze-run \
    --run-dir "${RUN_DIR}" \
    --ligand-resname "${LIGAND_RESNAME}" \
    --cutoff-ang "${BINDING_CUTOFF_ANG}"
done

echo "[4/7] Ranking allosteric residue and region effects..."
for RUN_DIR in "${RUN_DIRS[@]}"; do
  pert rank \
    --run-dir "${RUN_DIR}" \
    --ligand-resname "${LIGAND_RESNAME}" \
    --cutoff-ang "${BINDING_CUTOFF_ANG}" \
    --source-hotspots 10 \
    --top-edges 80 \
    --top-paths 30 \
    --top-residues 30
done

echo "[5/7] Building panel-level allostery consensus..."
pert consensus \
  --panel-dir "${OUTPUT_DIR}" \
  --ligand-resname "${LIGAND_RESNAME}" \
  --cutoff-ang "${BINDING_CUTOFF_ANG}" \
  --top-residues 30 \
  --top-regions 10 \
  ${CONTROL_ARGS[@]+"${CONTROL_ARGS[@]}"}

echo "[6/7] Writing panel-level allostery evidence report..."
pert evidence \
  --panel-dir "${OUTPUT_DIR}" \
  --top-residues 10 \
  --top-regions 5 \
  ${CONTROL_ARGS[@]+"${CONTROL_ARGS[@]}"}

echo "[7/7] Writing reader reports..."
for RUN_DIR in "${RUN_DIRS[@]}"; do
  pert report --run-dir "${RUN_DIR}"
done

echo "Done. Panel outputs are in: ${OUTPUT_DIR}"
echo "Null control (is it more than noise?): ${OUTPUT_DIR}/null_control/null_control_summary.json"
if [[ -f "${OUTPUT_DIR}/site_check/site_check_simulations.csv" ]]; then
  echo "Metal-site check (did the site survive?): ${OUTPUT_DIR}/site_check/site_check_simulations.csv"
fi
echo "Consensus ranking: ${OUTPUT_DIR}/allostery_consensus_residues.csv"
echo "Evidence report: ${OUTPUT_DIR}/allostery_evidence_report.md"
echo "Per-run reports: ${OUTPUT_DIR}/<scenario>/<replicate>/analysis/reader_report.md"
