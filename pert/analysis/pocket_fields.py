"""Stage 3a: topology-safe description of the site and its surroundings.

WT and mutant can have different numbers of atoms and residues, so their
coordinates cannot be compared atom by atom.  Instead each frame is described
by *density fields* sampled on a fixed grid that moves with the binding site:

1. a local reference frame is attached to the site.  By default it is the
   *shared site frame* of ``pert.analysis.site_frame``: WT and mutant are both
   superposed on the same site scaffold, so their fields are directly
   comparable.  Every frame is translated/rotated into it, and velocities are
   corrected for the frame's own translation and rotation;
2. atoms are split into three regions by distance to the frame origin:

   * ``B`` - binding pocket: protein/cofactor atoms within ``rb_ang``;
   * ``P`` - protein context: protein atoms between ``rb_ang`` and ``rp_ang``;
   * ``L`` - the ligand itself;

3. for each region, atoms are smeared with Gaussian kernels onto the grid, one
   channel per element class (``TYPE_BINS``), giving an occupancy field and,
   optionally, a momentum-like velocity field.

Because the grid and the channels are the same for both systems, the resulting
feature vectors live in a common space regardless of mutations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pert.analysis.site_frame import SiteFrame, compute_shared_frame_series
from pert.config import FieldConfig
from pert.constants import TYPE_BINS, WATER_RESNAMES
from pert.deps import app, unit
from pert.structure.residues import atom_type_bin, is_protein_residue


# --- grid and moving pocket frame ---------------------------------------------


def _build_grid_points(rp_nm: float, spacing_nm: float, max_points: int) -> np.ndarray:
    """Cubic grid points inside a sphere of radius ``rp_nm`` (evenly sub-sampled to ``max_points``)."""
    axis = np.arange(-rp_nm, rp_nm + 1e-12, spacing_nm, dtype=np.float64)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    mask = np.linalg.norm(points, axis=1) <= rp_nm + 1e-12
    points = points[mask]
    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[keep]
    return points


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Time derivative by central differences (one-sided at both ends)."""
    out = np.zeros_like(values)
    if len(values) == 1:
        return out
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    if len(values) > 2:
        out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return out


def _rotation_to_omega(rotations: np.ndarray, dt_ps: float) -> np.ndarray:
    """Angular velocity vectors of a series of rotation matrices (``omega_hat = R_dot R^T``)."""
    r_dot = _finite_difference(rotations, dt_ps)
    omega = np.zeros((rotations.shape[0], 3), dtype=np.float64)
    for t in range(rotations.shape[0]):
        omega_hat = r_dot[t] @ rotations[t].T
        omega[t, 0] = 0.5 * (omega_hat[2, 1] - omega_hat[1, 2])
        omega[t, 1] = 0.5 * (omega_hat[0, 2] - omega_hat[2, 0])
        omega[t, 2] = 0.5 * (omega_hat[1, 0] - omega_hat[0, 1])
    return omega


def _select_anchor_indices(
    topology: "app.Topology",
    ligand_resnames: Sequence[str],
) -> np.ndarray:
    """Atoms that define the pocket frame: ligand heavy atoms, else all ligand atoms, else the first 20 C-alphas."""
    ligand_names = {name.upper() for name in ligand_resnames}
    heavy: List[int] = []
    all_ligand: List[int] = []
    ca_fallback: List[int] = []

    for atom in topology.atoms():
        if atom.name == "CA":
            ca_fallback.append(atom.index)
        if atom.residue.name.upper() in ligand_names:
            all_ligand.append(atom.index)
            if atom.element is None or atom.element.symbol.upper() != "H":
                heavy.append(atom.index)

    if len(heavy) >= 3:
        return np.asarray(heavy, dtype=np.int32)
    if len(all_ligand) >= 3:
        return np.asarray(all_ligand, dtype=np.int32)
    if len(ca_fallback) >= 3:
        return np.asarray(ca_fallback[:20], dtype=np.int32)
    raise ValueError("Could not select at least 3 anchor atoms for pocket frame.")


def _compute_anchor_frame_series(
    positions: np.ndarray,
    anchor_indices: np.ndarray,
    reference_anchor_coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Origin and orientation of the pocket frame in every MD frame.

    The origin is the anchor centroid; the orientation is the rotation that best
    maps the anchors onto their first-frame geometry (with a guard against sudden
    flips for near-symmetric anchors, e.g. a single ion).
    """
    num_frames = positions.shape[0]
    centers = np.zeros((num_frames, 3), dtype=np.float64)
    rotations = np.zeros((num_frames, 3, 3), dtype=np.float64)
    prev_rot = np.eye(3, dtype=np.float64)

    if reference_anchor_coords is None:
        reference_anchor_coords = positions[0, anchor_indices, :].copy()
    ref_center = reference_anchor_coords.mean(axis=0)
    ref_centered = reference_anchor_coords - ref_center

    for t in range(num_frames):
        anchors = positions[t, anchor_indices, :]
        center = anchors.mean(axis=0)
        centers[t] = center
        cur_centered = anchors - center
        if np.linalg.matrix_rank(cur_centered) < 2:
            rotations[t] = prev_rot
            continue

        h = cur_centered.T @ ref_centered
        u, _, vt = np.linalg.svd(h)
        q = vt.T @ u.T  # current -> reference
        if np.linalg.det(q) < 0:
            vt[-1, :] *= -1.0
            q = vt.T @ u.T
        r_world = q.T
        if np.linalg.norm(r_world - prev_rot, ord="fro") > 2.5:
            # Guard against abrupt frame flips in near-symmetric anchors.
            r_world = prev_rot
        rotations[t] = r_world
        prev_rot = r_world

    return centers, rotations


# --- kernel density fields -----------------------------------------------------


def _classify_atoms(
    topology: "app.Topology",
    ligand_resnames: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Boolean masks (ligand / protein / cofactor / water), element channel and mass of every atom."""
    ligand_names = {name.upper() for name in ligand_resnames}
    n_atoms = topology.getNumAtoms()
    is_ligand = np.zeros(n_atoms, dtype=bool)
    is_protein = np.zeros(n_atoms, dtype=bool)
    is_cofactor = np.zeros(n_atoms, dtype=bool)
    is_water = np.zeros(n_atoms, dtype=bool)
    type_ids = np.zeros(n_atoms, dtype=np.int32)
    masses = np.ones(n_atoms, dtype=np.float64)

    for atom in topology.atoms():
        idx = atom.index
        resname = atom.residue.name.upper()
        if resname in ligand_names:
            is_ligand[idx] = True
        elif resname in WATER_RESNAMES:
            is_water[idx] = True
        elif is_protein_residue(resname):
            is_protein[idx] = True
        else:
            is_cofactor[idx] = True

        symbol = atom.element.symbol if atom.element else None
        type_ids[idx] = atom_type_bin(symbol)
        if atom.element is not None:
            masses[idx] = atom.element.mass.value_in_unit(unit.dalton)

    return {
        "is_ligand": is_ligand,
        "is_protein": is_protein,
        "is_cofactor": is_cofactor,
        "is_water": is_water,
        "type_ids": type_ids,
        "masses": masses,
    }


def _lagged_kernel_fields(
    coords: np.ndarray,
    vels: np.ndarray,
    type_ids: np.ndarray,
    atom_indices: np.ndarray,
    grid: np.ndarray,
    sigma_nm: float,
    include_velocity: bool,
    density_normalize: bool,
    append_type_counts: bool,
) -> np.ndarray:
    """Feature vector of one region in one frame.

    Selected atoms are smeared with Gaussian kernels (cut at 3 sigma) onto the
    grid, per element channel: ``phi`` (occupancy) and, optionally, ``psi``
    (kernel-weighted velocity, 3 components).  Concatenates ``phi``, ``psi`` and
    per-element atom counts.
    """
    n_grid = grid.shape[0]
    n_types = len(TYPE_BINS)
    phi = np.zeros((n_grid, n_types), dtype=np.float64)
    if include_velocity:
        psi = np.zeros((n_grid, n_types, 3), dtype=np.float64)
    else:
        psi = None
    type_counts = np.zeros(n_types, dtype=np.float64)

    if atom_indices.size == 0:
        parts: List[np.ndarray] = [phi.ravel()]
        if include_velocity and psi is not None:
            parts.append(psi.ravel())
        if append_type_counts:
            parts.append(type_counts)
        return np.concatenate(parts)

    sel_coords = coords[atom_indices]
    sel_vels = vels[atom_indices]
    sel_types = type_ids[atom_indices]
    type_counts = np.bincount(sel_types, minlength=n_types).astype(np.float64)
    diff = sel_coords[:, None, :] - grid[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    cutoff2 = (3.0 * sigma_nm) ** 2
    weights = np.exp(-dist2 / (2.0 * sigma_nm * sigma_nm))
    weights[dist2 > cutoff2] = 0.0

    for a in range(n_types):
        mask = sel_types == a
        if not np.any(mask):
            continue
        w = weights[mask]
        phi[:, a] = np.sum(w, axis=0)
        if include_velocity and psi is not None:
            vel_a = sel_vels[mask]
            for axis in range(3):
                psi[:, a, axis] = w.T @ vel_a[:, axis]

    if density_normalize:
        denom = max(1.0, float(atom_indices.size))
        phi /= denom
        if include_velocity and psi is not None:
            psi /= denom
        count_features = type_counts / denom
    else:
        count_features = np.log1p(type_counts)

    parts = [phi.ravel()]
    if include_velocity and psi is not None:
        parts.append(psi.ravel())
    if append_type_counts:
        parts.append(count_features)
    return np.concatenate(parts)


def compute_region_features(
    topology: "app.Topology",
    positions: np.ndarray,
    velocities: np.ndarray,
    dt_ps: float,
    ligand_resnames: Sequence[str],
    field_cfg: FieldConfig,
    site_frame: Optional[SiteFrame] = None,
) -> Dict[str, np.ndarray]:
    """Field features of pocket (B), context (P) and ligand (L) for a whole trajectory.

    Returns ``{"B": (frames, n_feat), "P": ..., "L": ..., "grid": points}``.
    Region membership is re-evaluated every frame from distances in the pocket
    frame, so atoms may enter or leave a region.

    With ``site_frame`` (see ``pert.analysis.site_frame``) the fields are
    expressed in the shared site frame of the WT/mutant pair and the result
    also carries ``"frame_fit_rmsd_ang"``.  Without it the legacy frame is
    used: anchored on the ligand (or the first C-alphas) and referenced to this
    trajectory's own first frame.
    """
    atom_info = _classify_atoms(topology, ligand_resnames)
    masses = atom_info["masses"]
    total_mass = float(np.sum(masses))
    pos = positions.copy()
    vel = velocities.copy()

    if field_cfg.remove_com and total_mass > 0:
        com = np.sum(pos * masses[None, :, None], axis=1) / total_mass
        com_v = np.sum(vel * masses[None, :, None], axis=1) / total_mass
        pos -= com[:, None, :]
        vel -= com_v[:, None, :]

    fit_rmsd_nm: Optional[np.ndarray] = None
    if site_frame is not None:
        centers, rotations, fit_rmsd_nm = compute_shared_frame_series(pos, site_frame)
    else:
        anchor_indices = _select_anchor_indices(topology, ligand_resnames)
        centers, rotations = _compute_anchor_frame_series(pos, anchor_indices)
    c_dot = _finite_difference(centers, dt_ps)
    omega = _rotation_to_omega(rotations, dt_ps)

    rb_nm = field_cfg.rb_ang * 0.1
    rp_nm = field_cfg.rp_ang * 0.1
    sigma_nm = field_cfg.sigma_ang * 0.1
    grid = _build_grid_points(
        rp_nm=rp_nm,
        spacing_nm=field_cfg.grid_spacing_ang * 0.1,
        max_points=field_cfg.max_grid_points,
    )

    protein_or_cofactor = atom_info["is_protein"] | atom_info["is_cofactor"]
    protein_only = atom_info["is_protein"]
    ligand_only = atom_info["is_ligand"]
    type_ids = atom_info["type_ids"]
    n_frames = pos.shape[0]

    feat_b: List[np.ndarray] = []
    feat_p: List[np.ndarray] = []
    feat_l: List[np.ndarray] = []

    for t in range(n_frames):
        r_rel = pos[t] - centers[t][None, :]
        r_prime = (rotations[t].T @ r_rel.T).T
        v_rel = vel[t] - c_dot[t][None, :]
        if site_frame is not None:
            # omega is a lab-frame vector: remove the frame's rotation in the lab
            # frame, then rotate into the site frame.
            v_prime = (rotations[t].T @ (v_rel - np.cross(omega[t][None, :], r_rel)).T).T
        else:
            # Legacy expression, kept verbatim so old results can be reproduced
            # (it crosses the lab-frame omega with site-frame coordinates).
            v_prime = (rotations[t].T @ v_rel.T).T - np.cross(omega[t][None, :], r_prime)
        radial = np.linalg.norm(r_prime, axis=1)

        idx_b = np.where(
            protein_or_cofactor & (~ligand_only) & (radial <= rb_nm)
        )[0].astype(np.int32)
        idx_p = np.where(
            protein_only & (radial > rb_nm) & (radial <= rp_nm)
        )[0].astype(np.int32)
        idx_l = np.where(ligand_only)[0].astype(np.int32)

        feat_b.append(
            _lagged_kernel_fields(
                coords=r_prime,
                vels=v_prime,
                type_ids=type_ids,
                atom_indices=idx_b,
                grid=grid,
                sigma_nm=sigma_nm,
                include_velocity=field_cfg.include_velocity,
                density_normalize=field_cfg.density_normalize,
                append_type_counts=field_cfg.append_type_counts,
            )
        )
        feat_p.append(
            _lagged_kernel_fields(
                coords=r_prime,
                vels=v_prime,
                type_ids=type_ids,
                atom_indices=idx_p,
                grid=grid,
                sigma_nm=sigma_nm,
                include_velocity=field_cfg.include_velocity,
                density_normalize=field_cfg.density_normalize,
                append_type_counts=field_cfg.append_type_counts,
            )
        )
        feat_l.append(
            _lagged_kernel_fields(
                coords=r_prime,
                vels=v_prime,
                type_ids=type_ids,
                atom_indices=idx_l,
                grid=grid,
                sigma_nm=sigma_nm,
                include_velocity=field_cfg.include_velocity,
                density_normalize=field_cfg.density_normalize,
                append_type_counts=field_cfg.append_type_counts,
            )
        )

    features = {
        "B": np.asarray(feat_b, dtype=np.float64),
        "P": np.asarray(feat_p, dtype=np.float64),
        "L": np.asarray(feat_l, dtype=np.float64),
        "grid": grid,
    }
    if fit_rmsd_nm is not None:
        features["frame_fit_rmsd_ang"] = 10.0 * fit_rmsd_nm
    return features
