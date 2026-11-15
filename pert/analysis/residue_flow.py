"""Residue-level information-flow network (who drives whom, and what changed).

Complements the region-level estimands with residue resolution:

1. every aligned residue gets a scalar *motion signal* per frame
   (z-scored displacement from its mean position + weighted z-scored speed);
2. a pairwise lag-1 transfer-entropy proxy ``TE[i, j]`` (does residue ``i``'s
   past help predict residue ``j``?) is computed for WT and mutant;
3. ``delta TE = TE_mut - TE_wt`` is searched for routes from the *sources*
   (residues displaced by the mutation) to the *targets* (binding-site
   residues): direct edges, one-relay paths and per-residue relay scores;
4. a sliding-window version gives the source->site flow score over time.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from pert.structure.binding_site import read_binding_residue_keys
from pert.structure.comparison import infer_mutation_anchor_from_structure
from pert.structure.geometry import kabsch_align
from pert.utils.numeric import zscore_columns


# --- per-residue motion signals ---------------------------------------------------


def align_positions_and_velocities(
    positions_nm: np.ndarray,
    velocities_nmps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove global rotation/translation: superpose every frame on the first one (velocities are rotated too)."""
    ref = np.asarray(positions_nm[0], dtype=float)
    pos_aligned = np.zeros_like(positions_nm, dtype=float)
    vel_aligned = np.zeros_like(velocities_nmps, dtype=float)
    for frame in range(int(positions_nm.shape[0])):
        rot, trans = kabsch_align(np.asarray(positions_nm[frame], dtype=float), ref)
        pos_aligned[frame] = (rot @ positions_nm[frame].T).T + trans[None, :]
        vel_aligned[frame] = (rot @ velocities_nmps[frame].T).T
    return pos_aligned, vel_aligned


def build_residue_motion_signals(
    positions_nm: np.ndarray,
    velocities_nmps: np.ndarray,
    velocity_weight: float = 0.35,
) -> np.ndarray:
    """Scalar motion signal per residue and frame.

    ``signal = z(|r - <r>|) + velocity_weight * z(|v|)``: how far a residue is from
    its average position plus how fast it moves, both standardised per residue.
    """
    disp = np.linalg.norm(positions_nm - np.mean(positions_nm, axis=0, keepdims=True), axis=2)
    speed = np.linalg.norm(velocities_nmps, axis=2)
    disp_z = zscore_columns(disp)
    speed_z = zscore_columns(speed)
    return disp_z + float(velocity_weight) * speed_z


# --- pairwise transfer-entropy proxy ------------------------------------------------


def _residual_variance(x: np.ndarray, y: np.ndarray, ridge: float = 1e-5) -> float:
    """Variance of the residuals of a ridge regression ``y ~ x``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xtx = x.T @ x + ridge * np.eye(x.shape[1], dtype=float)
    xty = x.T @ y
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtx) @ xty
    resid = y - x @ beta
    var = float(np.var(resid))
    if not np.isfinite(var):
        return 1e-10
    return max(var, 1e-10)


def pairwise_te_matrix(signal: np.ndarray, lag: int = 1) -> np.ndarray:
    """Pairwise lag-1 transfer-entropy proxy between residue signals.

    ``te[i, j] = 1/2 ln(var(j | past j) / var(j | past j, past i))``, clipped at 0:
    how much the past of residue ``i`` helps to predict residue ``j``.
    """
    signal = np.asarray(signal, dtype=float)
    t, n = signal.shape
    te = np.zeros((n, n), dtype=float)
    if t <= lag + 3:
        return te
    y_all = signal[lag:, :]
    yp_all = signal[:-lag, :]
    ones = np.ones((t - lag, 1), dtype=float)
    for target in range(n):
        y = y_all[:, target]
        yp = yp_all[:, target : target + 1]
        xr = np.hstack([ones, yp])
        var_r = _residual_variance(xr, y)
        for source in range(n):
            if source == target:
                continue
            xp = yp_all[:, source : source + 1]
            xf = np.hstack([ones, yp, xp])
            var_f = _residual_variance(xf, y)
            value = 0.5 * np.log(var_r / var_f)
            if np.isfinite(value):
                te[source, target] = max(0.0, float(value))
    return te


# --- sources (mutation) and targets (binding site) ----------------------------------


def indices_from_mut_residue_keys(pair_records: Sequence[Dict[str, object]], keys: Sequence[str]) -> List[int]:
    """Indices of pair records matching residue keys (``LYS2``, ``A:LYS2`` or ``A:2``)."""
    keyset = {str(k).strip() for k in keys if str(k).strip()}
    indices: List[int] = []
    for idx, rec in enumerate(pair_records):
        chain_id = str(rec["chain_id"]).strip()
        resid = str(rec["resid"]).strip()
        mut_key = str(rec["mut_residue_key"]).strip()
        if (
            mut_key in keyset
            or (chain_id and f"{chain_id}:{mut_key}" in keyset)
            or (chain_id and f"{chain_id}:{resid}" in keyset)
        ):
            indices.append(idx)
    return sorted(set(indices))


def source_target_indices(
    pair_records: Sequence[Dict[str, object]],
    wt_pdb: str,
    mut_pdb: str,
    binding_csv: str,
    ligand=None,
) -> Tuple[List[int], List[int]]:
    """Source residues (largest structural displacement) and target residues (binding site)."""
    try:
        _, hotspot_keys, _ = infer_mutation_anchor_from_structure(wt_pdb, mut_pdb, top_n=4, ligand=ligand)
    except Exception:
        hotspot_keys = []
    binding_keys = read_binding_residue_keys(binding_csv)
    source_idx = indices_from_mut_residue_keys(pair_records, hotspot_keys)
    target_idx = indices_from_mut_residue_keys(pair_records, binding_keys)
    if not source_idx and len(pair_records) > 0:
        source_idx = [0]
    if not target_idx and len(pair_records) > 1:
        target_idx = [len(pair_records) - 1]
    src_set = set(source_idx)
    filtered_targets = [idx for idx in target_idx if idx not in src_set]
    if filtered_targets:
        target_idx = filtered_targets
    if not target_idx and source_idx:
        target_idx = [source_idx[0]]
    return source_idx, target_idx


# --- flow scores, relays and paths ---------------------------------------------------


def flow_score(te: np.ndarray, source_idx: Sequence[int], target_idx: Sequence[int]) -> float:
    """Total source -> target flow: direct TE plus half of the best one-relay contributions."""
    if not source_idx or not target_idx:
        return 0.0
    src = np.asarray(source_idx, dtype=int)
    tgt = np.asarray(target_idx, dtype=int)
    direct = float(np.sum(te[np.ix_(src, tgt)]))
    src_set = set(int(i) for i in src.tolist())
    tgt_set = set(int(i) for i in tgt.tolist())
    relay = 0.0
    for k in range(int(te.shape[0])):
        if k in src_set or k in tgt_set:
            continue
        smax = float(np.max(te[src, k])) if len(src) else 0.0
        tmax = float(np.max(te[k, tgt])) if len(tgt) else 0.0
        relay += smax * tmax
    return direct + 0.5 * relay


def relay_scores(te_delta_pos: np.ndarray, source_idx: Sequence[int], target_idx: Sequence[int]) -> np.ndarray:
    """How much each residue acts as an intermediate between sources and targets (normalised to max 1)."""
    n = int(te_delta_pos.shape[0])
    relay = np.zeros(n, dtype=float)
    src_set = set(int(i) for i in source_idx)
    tgt_set = set(int(i) for i in target_idx)
    for s in source_idx:
        for t in target_idx:
            for k in range(n):
                if k in src_set or k in tgt_set:
                    continue
                relay[k] += min(float(te_delta_pos[s, k]), float(te_delta_pos[k, t]))
    scale = float(np.max(relay))
    if scale > 1e-12:
        relay = relay / scale
    return relay


def find_top_delta_edges(
    te_wt: np.ndarray,
    te_mut: np.ndarray,
    pair_records: Sequence[Dict[str, object]],
    top_n: int = 24,
    min_delta: float = 1e-5,
) -> List[Dict[str, object]]:
    """Residue pairs whose TE increased most in the mutant."""
    delta = te_mut - te_wt
    edges: List[Dict[str, object]] = []
    n = int(delta.shape[0])
    for source in range(n):
        for target in range(n):
            if source == target:
                continue
            dv = float(delta[source, target])
            if dv <= min_delta:
                continue
            edges.append(
                {
                    "src_index": source,
                    "dst_index": target,
                    "src_residue": str(pair_records[source]["node_label"]),
                    "dst_residue": str(pair_records[target]["node_label"]),
                    "te_wt": float(te_wt[source, target]),
                    "te_mut": float(te_mut[source, target]),
                    "delta": dv,
                }
            )
    edges.sort(key=lambda row: row["delta"], reverse=True)
    return edges[: max(1, int(top_n))]


def find_top_paths(
    te_delta_pos: np.ndarray,
    source_idx: Sequence[int],
    target_idx: Sequence[int],
    pair_records: Sequence[Dict[str, object]],
    top_n: int = 12,
) -> List[Dict[str, object]]:
    """Strongest direct and one-relay source -> target routes in the positive delta-TE network."""
    n = int(te_delta_pos.shape[0])
    candidates: List[Dict[str, object]] = []
    for s in source_idx:
        for t in target_idx:
            direct = float(te_delta_pos[s, t])
            if direct > 0.0:
                candidates.append(
                    {
                        "kind": "direct",
                        "path": [str(pair_records[s]["node_label"]), str(pair_records[t]["node_label"])],
                        "score": direct,
                    }
                )
            for k in range(n):
                if k == s or k == t:
                    continue
                score = min(float(te_delta_pos[s, k]), float(te_delta_pos[k, t]))
                if score <= 0.0:
                    continue
                candidates.append(
                    {
                        "kind": "relay",
                        "path": [
                            str(pair_records[s]["node_label"]),
                            str(pair_records[k]["node_label"]),
                            str(pair_records[t]["node_label"]),
                        ],
                        "score": score,
                    }
                )
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    return candidates[: max(1, int(top_n))]


def windowed_flow_timeseries(
    signal_wt: np.ndarray,
    signal_mut: np.ndarray,
    time_ps: np.ndarray,
    source_idx: Sequence[int],
    target_idx: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Source -> target flow score in sliding windows for WT, mutant and their difference."""
    n = min(int(signal_wt.shape[0]), int(signal_mut.shape[0]), int(time_ps.shape[0]))
    if n < 10:
        te_wt = pairwise_te_matrix(signal_wt[:n], lag=1)
        te_mut = pairwise_te_matrix(signal_mut[:n], lag=1)
        fw = flow_score(te_wt, source_idx, target_idx)
        fm = flow_score(te_mut, source_idx, target_idx)
        return (
            np.asarray([float(np.mean(time_ps[:n]))], dtype=float),
            np.asarray([fw], dtype=float),
            np.asarray([fm], dtype=float),
            np.asarray([fm - fw], dtype=float),
            {"window_frames": n, "step_frames": n},
        )

    window = max(24, min(90, n // 3))
    if window >= n:
        window = max(8, n - 1)
    step = max(3, window // 6)

    t_center: List[float] = []
    flow_wt: List[float] = []
    flow_mut: List[float] = []
    for start in range(0, n - window + 1, step):
        end = start + window
        te_w = pairwise_te_matrix(signal_wt[start:end], lag=1)
        te_m = pairwise_te_matrix(signal_mut[start:end], lag=1)
        t_center.append(float(np.mean(time_ps[start:end])))
        flow_wt.append(flow_score(te_w, source_idx, target_idx))
        flow_mut.append(flow_score(te_m, source_idx, target_idx))
    if not t_center:
        te_w = pairwise_te_matrix(signal_wt[:n], lag=1)
        te_m = pairwise_te_matrix(signal_mut[:n], lag=1)
        t_center = [float(np.mean(time_ps[:n]))]
        flow_wt = [flow_score(te_w, source_idx, target_idx)]
        flow_mut = [flow_score(te_m, source_idx, target_idx)]

    time_arr = np.asarray(t_center, dtype=float)
    wt_arr = np.asarray(flow_wt, dtype=float)
    mut_arr = np.asarray(flow_mut, dtype=float)
    return time_arr, wt_arr, mut_arr, mut_arr - wt_arr, {"window_frames": window, "step_frames": step}
