"""``pert pipeline-config`` - run one WT-vs-MUT comparison from a JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from pert.config import BuildConfig, FieldConfig, LatentConfig, MDConfig, TEConfig
from pert.pipelines.single_run import run_full_pipeline


DEFAULT_TEMPLATE: Dict[str, Any] = {
    "build": {
        "reference_pdb": "data/toy/1ZNF.pdb",
        "ligand_xml": "data/toy/empty_ligand.xml",
        "ligand_resnames": ["ZN"],
        "mutations": ["K2R"],
        "templates": ["1ZNFA"],
        "chain_id": "A",
        "original_sequence": None,
        "start_pos": None,
        "num_models": 8,
        "num_processors": 1,
        "modeller_seed": 17,
    },
    "md": {
        "temperature_k": 300.0,
        "friction_per_ps": 1.0,
        "timestep_fs": 2.0,
        "pressure_bar": 1.0,
        "barostat_interval": 25,
        "nonbonded_method": "PME",
        "nonbonded_cutoff_nm": 1.0,
        "add_solvent": True,
        "water_model": "tip3p",
        "solvent_padding_nm": 1.0,
        "ionic_strength_m": 0.15,
        "neutralize": True,
        "positive_ion": "Na+",
        "negative_ion": "Cl-",
        "minimization_max_iterations": 500,
        "heating_initial_temperature_k": 50.0,
        "heating_steps": 5000,
        "heating_stages": 10,
        "equilibration_steps": 10000,
        "production_steps": 50000,
        "report_interval": 100,
        "platform": "CPU",
        "random_seed": 1234,
        "checkpoint_interval_steps": 250000,
    },
    "field": {
        "rb_ang": 6.0,
        "rp_ang": 12.0,
        "grid_spacing_ang": 3.0,
        "sigma_ang": 3.0,
        "max_grid_points": 128,
        "include_velocity": True,
        "remove_com": True,
        "density_normalize": True,
        "append_type_counts": True,
        "frame_mode": "shared_site",
        "frame_site_cutoff_ang": 6.0,
        "frame_min_anchors": 6,
    },
    "latent": {
        "lag": 4,
        "m_b": 6,
        "m_p": 6,
        "m_l": 3,
        "reg": 1e-6,
        "solver": "auto",
    },
    "te": {
        "max_order": 3,
        "ridge_values": [1e-8, 1e-6, 1e-4, 1e-2],
        "surrogates": 64,
        "whiteness_lag": 10,
        "whiteness_threshold": 0.2,
        "random_seed": 13,
    },
    "output_dir": "runs/run_from_config",
}


def _write_template(path: str) -> None:
    """Write an example configuration file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(DEFAULT_TEMPLATE, handle, indent=2)
    print(path)


def _load_config(path: str) -> Dict[str, Any]:
    """Read and minimally validate a pipeline configuration."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "build" not in payload:
        raise ValueError("Config must contain 'build' section.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run topology-safe pipeline from JSON config.")
    parser.add_argument("--config", default=None, help="Path to JSON config file.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch, overwriting existing structures and simulations (default: reuse/continue them).",
    )
    parser.add_argument(
        "--write-template",
        default=None,
        help="Write a template JSON config to this path and exit.",
    )
    args = parser.parse_args()

    if args.write_template:
        _write_template(args.write_template)
        return
    if not args.config:
        raise SystemExit("Provide --config <path> or --write-template <path>.")

    cfg = _load_config(args.config)
    build_cfg = BuildConfig(**cfg["build"])
    md_cfg = MDConfig(**cfg.get("md", {}))
    field_cfg = FieldConfig(**cfg.get("field", {}))
    latent_cfg = LatentConfig(**cfg.get("latent", {}))

    te_cfg_data = dict(cfg.get("te", {}))
    if "ridge_values" in te_cfg_data:
        te_cfg_data["ridge_values"] = tuple(te_cfg_data["ridge_values"])
    te_cfg = TEConfig(**te_cfg_data)
    output_dir = cfg.get("output_dir", "runs/topology_safe_run")

    summary = run_full_pipeline(
        build_cfg=build_cfg,
        md_cfg=md_cfg,
        field_cfg=field_cfg,
        latent_cfg=latent_cfg,
        te_cfg=te_cfg,
        output_dir=output_dir,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
