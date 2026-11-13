#!/usr/bin/env bash
# Toy zinc-finger panel (1ZNF), 20 ns per run: the generic panel workflow with the toy config.
#
# Usage:  bash scripts/run_toy_panel_20ns.sh [path/to/panel_config.json]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_panel_workflow.sh" "${1:-${SCRIPT_DIR}/../configs/mutation_panel_20ns_config.json}"
