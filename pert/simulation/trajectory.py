"""Trajectory container used between the MD and analysis stages.

Trajectories are stored as compressed ``.npz`` files with ``positions`` (nm),
``velocities`` (nm/ps), ``time_ps`` and ``dt_ps``; shapes are
``(frames, atoms, 3)``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def load_traj_npz(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load ``(positions_nm, velocities_nm_per_ps, dt_ps)`` from a trajectory file."""
    data = np.load(path)
    positions = np.asarray(data["positions"], dtype=np.float64)
    velocities = np.asarray(data["velocities"], dtype=np.float64)
    if "dt_ps" in data:
        dt_ps = float(np.asarray(data["dt_ps"]).reshape(-1)[0])
    elif "time_ps" in data and len(data["time_ps"]) > 1:
        t = np.asarray(data["time_ps"], dtype=np.float64)
        dt_ps = float(np.median(np.diff(t)))
    else:
        dt_ps = 1.0
    return positions, velocities, dt_ps
