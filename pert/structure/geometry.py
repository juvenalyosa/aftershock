"""Rigid-body superposition (Kabsch algorithm).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def kabsch_fit(src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Optimal rigid superposition of ``src_points`` onto ``dst_points``.

    Returns ``(rotation, translation, rmsd)`` such that
    ``dst ~= (rotation @ src.T).T + translation``.  Reflections are excluded.
    """
    if src_points.shape != dst_points.shape:
        raise ValueError("Kabsch source and destination arrays must have equal shape.")
    if src_points.shape[0] < 3:
        raise ValueError("At least 3 points are required for Kabsch alignment.")

    src_centroid = src_points.mean(axis=0)
    dst_centroid = dst_points.mean(axis=0)
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = dst_centroid - (rot @ src_centroid)
    transformed = (rot @ src_points.T).T + trans
    rmsd = float(np.sqrt(np.mean(np.sum((transformed - dst_points) ** 2, axis=1))))
    return rot, trans, rmsd


def kabsch_align(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Same superposition as ``kabsch_fit`` without input checks or RMSD: returns ``(rotation, translation)``."""
    src_cent = np.mean(src, axis=0)
    dst_cent = np.mean(dst, axis=0)
    src0 = src - src_cent
    dst0 = dst - dst_cent
    h = src0.T @ dst0
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = dst_cent - rot @ src_cent
    return rot, trans
