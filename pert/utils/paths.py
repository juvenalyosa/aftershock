"""Filesystem helpers and the on-disk layout of runs and panels.

Layout produced by the pipelines::

    <panel_dir>/<scenario>/<replicate>/      one WT-vs-MUT run ("run_dir")
        run_summary.json                     index of everything below
        structures/                          WT and mutant PDBs (with ligand)
        model_build/                         MODELLER working files
        md/                                  prepared PDBs + trajectories
        analysis/                            estimands, plots, VMD scripts
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable, List, Optional


@contextmanager
def pushd(path: str) -> Iterable[None]:
    """Context manager: create ``path`` if needed, ``chdir`` into it, and always come back."""
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)



def resolve_run_path(run_dir: str, value: str) -> str:
    """Turn a path recorded in ``run_summary.json`` into a usable file path.

    Summaries store paths exactly as they were given when the run was launched
    (absolute, or relative to the launch directory).  Run directories are
    routinely moved afterwards (cluster -> laptop, ``runs/`` reorganisation),
    so the lookup tries, in order:

    1. the recorded path itself (absolute, or relative to the current directory);
    2. the recorded path relative to ``run_dir``;
    3. the longest trailing part of the recorded path that exists inside
       ``run_dir`` (e.g. ``old/panel/snp/rep1/md/wt_traj.npz`` -> ``<run_dir>/md/wt_traj.npz``).

    If nothing exists the historical answer (1 or 2) is returned so callers
    report the same missing path as before.
    """
    if os.path.isabs(value):
        if os.path.exists(value):
            return value
        legacy = value
    else:
        if os.path.exists(value):
            return os.path.abspath(value)
        legacy = os.path.abspath(os.path.join(run_dir, value))
        if os.path.exists(legacy):
            return legacy

    parts = [part for part in os.path.normpath(value).split(os.sep) if part]
    for start in range(1, len(parts)):
        candidate = os.path.abspath(os.path.join(run_dir, *parts[start:]))
        if os.path.exists(candidate):
            return candidate
    return legacy


def optional_run_path(run_dir: str, value: Optional[str]) -> Optional[str]:
    """``resolve_run_path`` that passes ``None``/empty values through."""
    if not value:
        return None
    return resolve_run_path(run_dir, value)


# --- panel layout: <panel_dir>/<scenario>/<replicate> -----------------------


def scenario_from_run_dir(run_dir: str) -> str:
    """Scenario name = parent directory of the run (``<panel>/<scenario>/<replicate>``)."""
    return os.path.basename(os.path.dirname(os.path.abspath(run_dir)))


def replicate_from_run_dir(run_dir: str) -> str:
    """Replicate name = last component of the run directory."""
    return os.path.basename(os.path.abspath(run_dir))


def find_run_dirs(panel_dir: str) -> List[str]:
    """All directories below ``panel_dir`` that contain a ``run_summary.json``."""
    out: List[str] = []
    for root, _, files in os.walk(panel_dir):
        if "run_summary.json" in files:
            out.append(root)
    return sorted(out)
