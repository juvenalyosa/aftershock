"""``pert md-panel-config`` - launch the MD mutation panel from a JSON file.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Tuple


OPTION_MAP = {
    "reference_pdb": "--reference-pdb",
    "template_pdb": "--template-pdb",
    "ligand_xml": "--ligand-xml",
    "ligand_resname": "--ligand-resname",
    "template": "--template",
    "chain_id": "--chain-id",
    "replicas": "--replicas",
    "output_dir": "--output-dir",
    "num_models": "--num-models",
    "num_processors": "--num-processors",
    "temperature_k": "--temperature-k",
    "equil_steps": "--equil-steps",
    "prod_steps": "--prod-steps",
    "report_interval": "--report-interval",
    "nonbonded_method": "--nonbonded-method",
    "nonbonded_cutoff_nm": "--nonbonded-cutoff-nm",
    "platform": "--platform",
    "md_seed_base": "--md-seed-base",
    "rb_ang": "--rb-ang",
    "rp_ang": "--rp-ang",
    "grid_spacing_ang": "--grid-spacing-ang",
    "sigma_ang": "--sigma-ang",
    "max_grid_points": "--max-grid-points",
    "vamp_lag": "--vamp-lag",
    "latent_b": "--latent-b",
    "latent_p": "--latent-p",
    "latent_l": "--latent-l",
    "var_max_order": "--var-max-order",
    "te_surrogates": "--te-surrogates",
    "te_seed_base": "--te-seed-base",
    "scenarios": "--scenarios",
    "mutation_file": "--mutation-file",
    "frame_mode": "--frame-mode",
    "vamp_solver": "--vamp-solver",
    "checkpoint_interval_steps": "--checkpoint-interval-steps",
    "max_null_pairs": "--max-null-pairs",
}


BOOLEAN_FLAGS = {
    "occupancy_only": "--occupancy-only",
    "no_null_control": "--no-null-control",
    "no_resume": "--no-resume",
    "include_builtins": "--include-builtins",
    "no_solvent": "--no-solvent",
}


def _load_config(path: str) -> Dict[str, Any]:
    """Read the panel configuration (the ``panel`` section when present)."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "panel" in payload and isinstance(payload["panel"], dict):
        return payload["panel"]
    return payload


def _build_command(panel_cfg: Dict[str, Any]) -> List[str]:
    """Translate the configuration into a ``pert md-panel`` command line."""
    cmd = [sys.executable, "-m", "pert", "md-panel"]
    for key, opt in OPTION_MAP.items():
        if key in panel_cfg and panel_cfg[key] is not None:
            value = panel_cfg[key]
            if isinstance(value, (list, tuple)):
                for token in value:
                    cmd.extend([opt, str(token)])
            else:
                cmd.extend([opt, str(value)])
    for key, flag in BOOLEAN_FLAGS.items():
        if bool(panel_cfg.get(key, False)):
            cmd.append(flag)
    for name, mutations in _mutation_sets(panel_cfg.get("mutation_sets")):
        cmd.extend(["--mutation-set", f"{name}:{','.join(mutations)}"])
    return cmd


def _mutation_sets(value: Any) -> List[Tuple[str, List[str]]]:
    """Scenarios written in the config: ``{"name": ["K2R", ...]}`` or ``[{"name", "mutations"}]``."""
    if not value:
        return []
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = [(item["name"], item["mutations"]) for item in value]
    else:
        raise ValueError("'mutation_sets' must be a mapping or a list of {name, mutations} objects.")
    out: List[Tuple[str, List[str]]] = []
    for name, mutations in items:
        tokens = [mutations] if isinstance(mutations, str) else list(mutations)
        out.append((str(name), [str(token) for token in tokens]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mutation panel from JSON config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and exit without running.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cmd = _build_command(cfg)
    print(" ".join(shlex.quote(token) for token in cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
