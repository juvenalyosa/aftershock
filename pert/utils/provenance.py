"""Record how a result was produced: software versions, machine, command, time."""

from __future__ import annotations

import datetime
import importlib
import platform
import sys
from typing import Dict, Optional


def _version(module_name: str) -> Optional[str]:
    """Version of an already-installed package, without failing when it is absent."""
    try:
        from importlib import metadata

        return metadata.version(module_name)
    except Exception:
        pass
    module = sys.modules.get(module_name)
    if module is None and module_name != "modeller":  # importing MODELLER needs its licence
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return None
    return getattr(module, "__version__", None) if module is not None else None


def collect_provenance() -> Dict[str, object]:
    """Versions of pert and its dependencies, the machine, the command line and a UTC timestamp."""
    import pert

    versions = {"pert": pert.__version__, "python": platform.python_version()}
    for name, package in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("openmm", "openmm"),
        ("modeller", "modeller"),
    ):
        versions[name] = _version(package) or _version(name)
    return {
        "versions": versions,
        "machine": platform.platform(),
        "command": list(sys.argv),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
