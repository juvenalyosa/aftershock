"""VMD scenes for mutation -> binding-site flow.

* ``write_mut_flow_vmd_script``: mutant trajectory with the pocket latent map;
* ``write_flow_network_vmd_script``: strongest inferred source->site route with
  a marker whose size/colour follows the windowed delta-flow score.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pert.structure.binding_site import read_binding_delta_csv
from pert.structure.comparison import infer_mutation_anchor_from_structure
from pert.structure.geometry import kabsch_align
from pert.structure.residues import protein_ca_records
from pert.visualization.vmd_common import selection_from_indices


def _latent_project_ca_vectors(
    wt_pdb: str,
    mut_pdb: str,
    mut_traj_npz: Optional[str],
    latents_npz: Optional[str],
    ridge: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """Per-residue displacement direction associated with the WT -> MUT shift of the pocket latents.

    C-alpha coordinates of the mutant trajectory are regressed on the pocket
    latents; the regression is evaluated along the mean latent shift (MUT - WT).
    """
    if not mut_traj_npz or not latents_npz:
        return {}
    if not os.path.exists(mut_traj_npz) or not os.path.exists(latents_npz):
        return {}

    mut_records = protein_ca_records(mut_pdb)
    wt_records = protein_ca_records(wt_pdb)
    if len(mut_records) < 3 or len(wt_records) < 3:
        return {}

    mut_ca_indices = np.asarray([int(rec["atom_index"]) for rec in mut_records], dtype=int)
    traj = np.load(mut_traj_npz)
    positions = np.asarray(traj["positions"], dtype=float)
    lat = np.load(latents_npz)
    u_mut = np.asarray(lat["u_b_mut"], dtype=float)
    u_wt = np.asarray(lat["u_b_wt"], dtype=float)
    n = min(int(positions.shape[0]), int(u_mut.shape[0]))
    if n < 8:
        return {}

    positions = positions[:n]
    u_mut = u_mut[:n]

    ca_xyz = positions[:, mut_ca_indices, :]
    ref = ca_xyz[0]
    ca_aligned = np.zeros_like(ca_xyz)
    for t in range(n):
        rot, trans = kabsch_align(ca_xyz[t], ref)
        ca_aligned[t] = (rot @ ca_xyz[t].T).T + trans[None, :]

    y = ca_aligned.reshape(n, -1)
    y = y - np.mean(y, axis=0, keepdims=True)
    x = u_mut - np.mean(u_mut, axis=0, keepdims=True)
    if x.shape[1] < 1:
        return {}

    xtx = x.T @ x + ridge * np.eye(x.shape[1], dtype=float)
    xty = x.T @ y
    try:
        w = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(xtx) @ xty

    q = np.mean(u_mut, axis=0) - np.mean(u_wt, axis=0)
    q_norm = float(np.linalg.norm(q))
    if q_norm < 1e-10:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        q = vt[0]
    proj = q @ w
    vecs_nm = proj.reshape(len(mut_records), 3)
    return {str(mut_records[i]["residue_key"]): vecs_nm[i].astype(float) for i in range(len(mut_records))}


def write_mut_flow_vmd_script(
    wt_pdb: str,
    mut_pdb: str,
    mut_dcd: Optional[str],
    mut_traj_npz: Optional[str],
    latents_npz: Optional[str],
    ligand_resname: str,
    binding_csv: str,
    output_tcl: str,
    cutoff_ang: float,
    analysis: Optional[Dict[str, object]] = None,
    max_vectors: int = 6,
) -> str:
    """Write the mutant-only scene with the pocket conformation map that follows the movie frame."""
    mut_records = protein_ca_records(mut_pdb)
    mut_key_to_coord = {str(rec["residue_key"]): np.asarray(rec["coord"], dtype=float) for rec in mut_records}
    mut_key_to_resid = {str(rec["residue_key"]): str(rec["resid"]) for rec in mut_records}
    mut_key_to_chain = {str(rec["residue_key"]): str(rec["chain_id"]).strip() for rec in mut_records}

    def _selection_from_key(residue_key: str) -> Optional[str]:
        resid = mut_key_to_resid.get(residue_key)
        if resid is None:
            return None
        chain_id = mut_key_to_chain.get(residue_key, "")
        if chain_id:
            return f"protein and name CA and chain {chain_id} and resid {resid}"
        return f"protein and name CA and resid {resid}"

    anchor_center, hotspot_keys, _ = infer_mutation_anchor_from_structure(wt_pdb, mut_pdb, top_n=3)
    delta_rows = read_binding_delta_csv(binding_csv)
    delta_rows = [(k, v) for (k, v) in delta_rows if k in mut_key_to_coord]

    proj_map = _latent_project_ca_vectors(
        wt_pdb=wt_pdb,
        mut_pdb=mut_pdb,
        mut_traj_npz=mut_traj_npz,
        latents_npz=latents_npz,
    )
    use_projection = any(key in proj_map for key, _ in delta_rows)

    if use_projection:
        delta_rows_sorted = sorted(
            [item for item in delta_rows if item[0] in proj_map],
            key=lambda item: float(np.linalg.norm(proj_map[item[0]])),
            reverse=True,
        )
    else:
        delta_rows_sorted = sorted(delta_rows, key=lambda item: abs(item[1]), reverse=True)
    delta_rows_draw = delta_rows_sorted[: max(1, int(max_vectors))]

    binding_coords = [mut_key_to_coord[k] for k, _ in delta_rows if k in mut_key_to_coord]
    if binding_coords:
        binding_center = np.mean(np.stack(binding_coords, axis=0), axis=0)
    else:
        binding_center = anchor_center.copy()

    hotspot_sels: List[str] = []
    for key in hotspot_keys:
        sel = _selection_from_key(key)
        if sel and sel not in hotspot_sels:
            hotspot_sels.append(sel)
    if not hotspot_sels and delta_rows_draw:
        fallback_sel = _selection_from_key(delta_rows_draw[0][0])
        if fallback_sel:
            hotspot_sels = [fallback_sel]
    anchor_sel_text = " or ".join(f"({sel})" for sel in hotspot_sels) if hotspot_sels else "protein and name CA"

    if use_projection:
        mags = [float(np.linalg.norm(proj_map[k])) for k, _ in delta_rows if k in proj_map]
    else:
        mags = [abs(v) for _, v in delta_rows]
    max_mag = max(mags) if mags else 1.0
    if max_mag < 1e-12:
        max_mag = 1.0
    mean_mag = float(np.mean(mags)) if mags else 0.0
    local_signal = mean_mag / max_mag
    sus_signal = 0.0
    te_signal = 0.0
    if isinstance(analysis, dict):
        try:
            sus_signal = float(analysis.get("susceptibility_delta_fro_norm", 0.0))
        except Exception:
            sus_signal = 0.0
        try:
            te_signal = abs(float(analysis.get("te_delta_mut_minus_wt", 0.0)))
        except Exception:
            te_signal = 0.0
    sus_norm = float(np.tanh(max(0.0, sus_signal) / 6.0))
    te_norm = float(np.tanh(max(0.0, te_signal) / 0.25))
    global_scale = float(np.clip(0.5 * local_signal + 0.3 * sus_norm + 0.2 * te_norm, 0.0, 1.0))
    disp_clip_ang = float(1.2 + 1.8 * max(global_scale, 0.25))

    flow_entries: List[Tuple[str, float, int]] = []
    for residue_key, delta in delta_rows_draw:
        sel_text = _selection_from_key(residue_key)
        if not sel_text:
            continue
        target = mut_key_to_coord[residue_key]
        radial = target - anchor_center
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 1e-10:
            radial_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            radial_dir = radial / radial_norm

        if use_projection and residue_key in proj_map:
            pv = np.asarray(proj_map[residue_key], dtype=float)
            proj_sign = float(np.dot(pv, radial_dir))
            raw_mag = float(np.linalg.norm(pv))
        else:
            proj_sign = float(delta)
            raw_mag = abs(float(delta))

        mag = float(np.clip(raw_mag / max_mag, 0.0, 1.0))
        sign_flag = 1 if proj_sign >= 0.0 else -1
        flow_entries.append((sel_text, mag, sign_flag))

    site_target_sels: List[str] = []
    for residue_key, _ in delta_rows:
        sel_text = _selection_from_key(residue_key)
        if sel_text and sel_text not in site_target_sels:
            site_target_sels.append(sel_text)
    for sel_text, _, _ in flow_entries:
        if sel_text not in site_target_sels:
            site_target_sels.append(sel_text)

    def _fmt_vec_ang(v_nm: np.ndarray) -> str:
        vv = 10.0 * np.asarray(v_nm, dtype=float)
        return f"{{{vv[0]:.4f} {vv[1]:.4f} {vv[2]:.4f}}}"

    flow_targets_tcl = " ".join(f"{{{{{sel}}} {mag:.6f} {sgn}}}" for sel, mag, sgn in flow_entries)
    site_targets_tcl = " ".join(f"{{{sel}}}" for sel in site_target_sels)

    latent_xy: List[Tuple[float, float]] = []
    if latents_npz and os.path.exists(latents_npz):
        try:
            lat_data = np.load(latents_npz)
            u_mut = np.asarray(lat_data["u_b_mut"], dtype=float)
            if u_mut.ndim == 2 and u_mut.shape[0] > 0:
                if u_mut.shape[1] >= 2:
                    u2 = u_mut[:, :2]
                else:
                    u2 = np.column_stack([u_mut[:, 0], np.zeros(u_mut.shape[0], dtype=float)])
                x = u2[:, 0]
                y = u2[:, 1]
                xmin = float(np.min(x))
                xmax = float(np.max(x))
                ymin = float(np.min(y))
                ymax = float(np.max(y))
                xrange = max(xmax - xmin, 1e-8)
                yrange = max(ymax - ymin, 1e-8)
                for row in u2:
                    latent_xy.append((float((row[0] - xmin) / xrange), float((row[1] - ymin) / yrange)))
        except Exception:
            latent_xy = []
    if not latent_xy:
        latent_xy = [(0.5, 0.5)]
    latent_xy_tcl = " ".join(f"{{{xy[0]:.6f} {xy[1]:.6f}}}" for xy in latent_xy)

    lines = [
        "# Auto-generated MUT-only flow view for VMD",
        "display projection Orthographic",
        "display depthcue off",
        "axes location Off",
        "color Display Background white",
        f"set mut_pdb {{{os.path.abspath(mut_pdb).replace(os.sep, '/')}}}",
        f"set mut_dcd {{{os.path.abspath(mut_dcd).replace(os.sep, '/') if mut_dcd else ''}}}",
        f"set ligand_sel {{resname {ligand_resname.upper()}}}",
        f"set site_cutoff {float(cutoff_ang):.3f}",
        f"set anchor_sel_text {{{anchor_sel_text}}}",
        f"set anchor_fallback {_fmt_vec_ang(anchor_center)}",
        f"set site_fallback {_fmt_vec_ang(binding_center)}",
        f"set latent_xy {{{latent_xy_tcl}}}",
        "",
        "mol new $mut_pdb type pdb waitfor all",
        "set mid [molinfo top]",
        "if {[string length $mut_dcd] > 0 && [file exists $mut_dcd]} {",
        "    mol addfile $mut_dcd type dcd waitfor all molid $mid",
        "}",
        "",
        "proc fit_trajectory_to_first {mid sel_text} {",
        "    set ref [atomselect $mid $sel_text frame 0]",
        "    if {[$ref num] < 3} {",
        "        $ref delete",
        "        return",
        "    }",
        "    set mob [atomselect $mid $sel_text]",
        "    set all [atomselect $mid all]",
        "    set nframes [molinfo $mid get numframes]",
        "    for {set f 0} {$f < $nframes} {incr f} {",
        "        $mob frame $f",
        "        $all frame $f",
        "        set xform [measure fit $mob $ref]",
        "        $all move $xform",
        "    }",
        "    $ref delete",
        "    $mob delete",
        "    $all delete",
        "}",
        "fit_trajectory_to_first $mid {protein and name CA}",
        "",
        "mol delrep 0 $mid",
        "",
        "mol representation NewCartoon 0.24 20.0 4.1 0",
        "mol color ColorID 6",
        "mol selection {protein}",
        "mol material AOShiny",
        "mol addrep $mid",
        "",
        "mol representation Licorice 0.20 18.0 18.0",
        "mol color Name",
        "mol selection $ligand_sel",
        "mol material Glossy",
        "mol addrep $mid",
        "",
        "mol representation QuickSurf 1.0 0.6 1.0 1.0",
        "mol color ColorID 10",
        "mol selection \"protein and within $site_cutoff of ($ligand_sel)\"",
        "mol material Transparent",
        "mol addrep $mid",
    ]

    lines += [
        "",
        "proc _clip01 {x} {",
        "    if {$x < 0.0} { return 0.0 }",
        "    if {$x > 1.0} { return 1.0 }",
        "    return $x",
        "}",
        "",
        "proc _latent_color {t} {",
        "    set tt [_clip01 $t]",
        "    set r [expr {int(46 + $tt * 194)}]",
        "    set g [expr {int(118 + $tt * -55)}]",
        "    set b [expr {int(190 + $tt * -158)}]",
        "    return [format \"#%02x%02x%02x\" $r $g $b]",
        "}",
        "",
        "proc _latent_canvas_xy {u1 u2} {",
        "    set w 560",
        "    set h 420",
        "    set m 58",
        "    set cx [expr {$m + $u1 * ($w - 2.0 * $m)}]",
        "    set cy [expr {$h - $m - $u2 * ($h - 2.0 * $m)}]",
        "    return [list $cx $cy]",
        "}",
        "",
        "proc setup_latent_window {} {",
        "    global latent_xy latent_canvas latent_marker latent_frame_text latent_plot_ready",
        "    set latent_plot_ready 0",
        "    if {[catch {",
        "        if {[winfo exists .latent2d]} {",
        "            destroy .latent2d",
        "        }",
        "        toplevel .latent2d",
        "        wm title .latent2d {MUT Latent Trajectory}",
        "        wm geometry .latent2d 560x420+80+80",
        "        canvas .latent2d.c -width 560 -height 420 -background #f4f8ff -highlightthickness 0",
        "        pack .latent2d.c -fill both -expand 1",
        "    } err]} {",
        "        puts [format {Latent 2D window not available: %s} $err]",
        "        return",
        "    }",
        "    set c .latent2d.c",
        "    set latent_canvas $c",
        "    set w 560",
        "    set h 420",
        "    set m 58",
        "    for {set i 0} {$i <= 6} {incr i} {",
        "        set x [expr {$m + $i * ($w - 2.0 * $m) / 6.0}]",
        "        set y [expr {$m + $i * ($h - 2.0 * $m) / 6.0}]",
        "        $c create line $x $m $x [expr {$h - $m}] -fill #d9e3f0 -width 1",
        "        $c create line $m $y [expr {$w - $m}] $y -fill #d9e3f0 -width 1",
        "    }",
        "    $c create line $m [expr {$h - $m}] [expr {$w - $m}] [expr {$h - $m}] -width 2 -fill black",
        "    $c create line $m [expr {$h - $m}] $m $m -width 2 -fill black",
        "    $c create text [expr {$w / 2}] 20 -text {Pocket Conformation Map} -font {Helvetica 15 bold} -fill #17324d",
        "    $c create text [expr {$w / 2}] [expr {$h - 20}] -text {State axis 1} -font {Helvetica 11} -fill #17324d",
        "    $c create text 24 [expr {$h / 2}] -text {State axis 2} -angle 90 -font {Helvetica 11} -fill #17324d",
        "    $c create text [expr {$w / 2}] 42 -text {Dot moves with the movie frame} -font {Helvetica 10 italic} -fill #45627d",
        "    set n [llength $latent_xy]",
        "    for {set i 1} {$i < $n} {incr i} {",
        "        set p0 [lindex $latent_xy [expr {$i - 1}]]",
        "        set p1 [lindex $latent_xy $i]",
        "        lassign [_latent_canvas_xy [lindex $p0 0] [lindex $p0 1]] x0 y0",
        "        lassign [_latent_canvas_xy [lindex $p1 0] [lindex $p1 1]] x1 y1",
        "        set t [expr {$i / double(max($n - 1, 1))}]",
        "        set col [_latent_color $t]",
        "        $c create line $x0 $y0 $x1 $y1 -fill $col -width 2",
        "    }",
        "    set p_init [lindex $latent_xy 0]",
        "    lassign [_latent_canvas_xy [lindex $p_init 0] [lindex $p_init 1]] x y",
        "    $c create oval [expr {$x - 5}] [expr {$y - 5}] [expr {$x + 5}] [expr {$y + 5}] -fill #22c55e -outline #14532d -width 1.5",
        "    $c create text [expr {$x + 20}] [expr {$y - 10}] -text {Start} -font {Helvetica 9 bold} -fill #14532d",
        "    set p_last [lindex $latent_xy [expr {$n - 1}]]",
        "    lassign [_latent_canvas_xy [lindex $p_last 0] [lindex $p_last 1]] xL yL",
        "    $c create oval [expr {$xL - 5}] [expr {$yL - 5}] [expr {$xL + 5}] [expr {$yL + 5}] -fill #f97316 -outline #7c2d12 -width 1.5",
        "    $c create text [expr {$xL + 22}] [expr {$yL + 12}] -text {End} -font {Helvetica 9 bold} -fill #7c2d12",
        "    set latent_marker [$c create oval [expr {$x - 8}] [expr {$y - 8}] [expr {$x + 8}] [expr {$y + 8}] -fill #2563eb -outline #0f172a -width 2]",
        "    set latent_frame_text [$c create text [expr {$w - 96}] 20 -text {Frame 0} -font {Helvetica 11 bold} -fill #17324d]",
        "    set latent_progress_text [$c create text [expr {$w - 98}] 42 -text {Progress 0%} -font {Helvetica 10} -fill #2f4b69]",
        "    set latent_plot_ready 1",
        "}",
        "",
        "proc update_latent_window {frame} {",
        "    global latent_xy latent_canvas latent_marker latent_frame_text latent_progress_text latent_plot_ready",
        "    if {![info exists latent_plot_ready] || !$latent_plot_ready} {",
        "        return",
        "    }",
        "    if {![winfo exists .latent2d] || ![winfo exists $latent_canvas]} {",
        "        set latent_plot_ready 0",
        "        return",
        "    }",
        "    set n [llength $latent_xy]",
        "    if {$n < 1} {",
        "        return",
        "    }",
        "    set idx $frame",
        "    if {$idx < 0} {",
        "        set idx 0",
        "    }",
        "    if {$idx >= $n} {",
        "        set idx [expr {$n - 1}]",
        "    }",
        "    set p [lindex $latent_xy $idx]",
        "    lassign [_latent_canvas_xy [lindex $p 0] [lindex $p 1]] x y",
        "    $latent_canvas coords $latent_marker [expr {$x - 8}] [expr {$y - 8}] [expr {$x + 8}] [expr {$y + 8}]",
        "    set tt [expr {$idx / double(max($n - 1, 1))}]",
        "    set col [_latent_color $tt]",
        "    $latent_canvas itemconfigure $latent_marker -fill $col",
        "    $latent_canvas itemconfigure $latent_frame_text -text [format {Frame %d/%d} $idx [expr {$n - 1}]]",
        "    $latent_canvas itemconfigure $latent_progress_text -text [format {Progress %.1f%%} [expr {100.0 * $tt}]]",
        "}",
        "",
        "proc draw_mut_flow_frame {mid frame} {",
        "    # Keep 3D view clean: no extra vectors or geometric overlays.",
        "    return",
        "}",
        "",
        "proc _refresh_mut_flow {name index op} {",
        "    global mid",
        "    if {[catch {molinfo $mid get frame} frame]} {",
        "        return",
        "    }",
        "    draw_mut_flow_frame $mid $frame",
        "    update_latent_window $frame",
        "}",
        "",
        "setup_latent_window",
        "draw_mut_flow_frame $mid 0",
        "update_latent_window 0",
        "catch {trace remove variable vmd_frame($mid) write _refresh_mut_flow}",
        "trace add variable vmd_frame($mid) write _refresh_mut_flow",
    ]

    lines += [
        "",
        "animate style Loop",
        "animate speed 0.7",
        "animate goto 0",
        "display resetview",
        "puts \"Loaded MUT-only perturbation-flow view.\"",
        "puts \"3D window: protein + ligand + transparent binding-site volume.\"",
        "puts \"Frames are aligned to remove global translation/rotation.\"",
        "puts \"2D window: latent conformation map, moving dot = current frame.\"",
        "puts \"Path color encodes progression from early (blue) to late (orange).\"",
    ]

    os.makedirs(os.path.dirname(output_tcl) or ".", exist_ok=True)
    with open(output_tcl, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return output_tcl


def write_flow_network_vmd_script(
    mut_pdb: str,
    mut_dcd: Optional[str],
    ligand_resname: str,
    pair_records: Sequence[Dict[str, object]],
    top_edges: Sequence[Dict[str, object]],
    source_idx: Sequence[int],
    target_idx: Sequence[int],
    relay_scores: np.ndarray,
    flow_delta_series: np.ndarray,
    output_tcl: str,
    top_paths: Optional[Sequence[Dict[str, object]]] = None,
) -> str:
    """Write the scene that animates the strongest mutation -> site route and the delta-flow time series."""
    source_sel = selection_from_indices(pair_records, source_idx)
    target_sel = selection_from_indices(pair_records, target_idx)

    def _fmt_vec_ang(v_nm: np.ndarray) -> str:
        vv = 10.0 * np.asarray(v_nm, dtype=float)
        return f"{{{vv[0]:.4f} {vv[1]:.4f} {vv[2]:.4f}}}"

    def _mean_coord(indices: Sequence[int]) -> Optional[np.ndarray]:
        coords = []
        for idx in indices:
            if idx < 0 or idx >= len(pair_records):
                continue
            coords.append(np.asarray(pair_records[idx]["mut_coord_nm"], dtype=float))
        if not coords:
            return None
        return np.mean(np.stack(coords, axis=0), axis=0)

    source_center = _mean_coord(source_idx)
    target_center = _mean_coord(target_idx)
    label_to_coord = {
        str(rec["node_label"]): np.asarray(rec["mut_coord_nm"], dtype=float)
        for rec in pair_records
    }
    waypoint_labels: List[str] = []
    if top_paths:
        for row in top_paths:
            path_raw = row.get("path") if isinstance(row, dict) else None
            if not isinstance(path_raw, list):
                continue
            candidate = [str(label) for label in path_raw if str(label) in label_to_coord]
            if len(candidate) >= 2:
                waypoint_labels = candidate
                break
    if not waypoint_labels and source_idx and target_idx:
        src_label = str(pair_records[int(source_idx[0])]["node_label"])
        dst_label = str(pair_records[int(target_idx[0])]["node_label"])
        if src_label in label_to_coord and dst_label in label_to_coord:
            waypoint_labels = [src_label, dst_label]

    waypoint_coords: List[np.ndarray] = []
    for label in waypoint_labels:
        if label not in label_to_coord:
            continue
        coord = label_to_coord[label]
        if waypoint_coords and np.linalg.norm(coord - waypoint_coords[-1]) < 1e-10:
            continue
        waypoint_coords.append(coord)
    if len(waypoint_coords) < 2 and source_center is not None and target_center is not None:
        waypoint_coords = [source_center, target_center]
    if len(waypoint_coords) < 2 and len(pair_records) >= 2:
        waypoint_coords = [
            np.asarray(pair_records[0]["mut_coord_nm"], dtype=float),
            np.asarray(pair_records[-1]["mut_coord_nm"], dtype=float),
        ]
    if not waypoint_coords and len(pair_records) > 0:
        waypoint_coords = [np.asarray(pair_records[0]["mut_coord_nm"], dtype=float)]
    if len(waypoint_coords) == 1:
        waypoint_coords = [waypoint_coords[0], waypoint_coords[0]]

    flow_waypoints_tcl = " ".join(_fmt_vec_ang(coord) for coord in waypoint_coords)

    flow_delta = np.asarray(flow_delta_series, dtype=float).reshape(-1)
    if flow_delta.size < 1:
        flow_delta = np.asarray([0.0], dtype=float)
    flow_delta_tcl = " ".join(f"{float(v):.6f}" for v in flow_delta.tolist())

    lines = [
        "# Auto-generated mutation-flow network view for VMD",
        "display projection Orthographic",
        "display depthcue off",
        "axes location Off",
        "color Display Background white",
        f"set mut_pdb {{{os.path.abspath(mut_pdb).replace(os.sep, '/')}}}",
        f"set mut_dcd {{{os.path.abspath(mut_dcd).replace(os.sep, '/') if mut_dcd else ''}}}",
        f"set ligand_sel {{resname {ligand_resname.upper()}}}",
        f"set source_sel {{{source_sel}}}",
        f"set target_sel {{{target_sel}}}",
        f"set flow_waypoints {{{flow_waypoints_tcl}}}",
        f"set flow_delta {{{flow_delta_tcl}}}",
        "",
        "mol new $mut_pdb type pdb waitfor all",
        "set mid [molinfo top]",
        "if {[string length $mut_dcd] > 0 && [file exists $mut_dcd]} {",
        "    mol addfile $mut_dcd type dcd waitfor all molid $mid",
        "}",
        "",
        "proc fit_to_first {mid sel_text} {",
        "    set ref [atomselect $mid $sel_text frame 0]",
        "    if {[$ref num] < 3} {",
        "        $ref delete",
        "        return",
        "    }",
        "    set mob [atomselect $mid $sel_text]",
        "    set all [atomselect $mid all]",
        "    set nf [molinfo $mid get numframes]",
        "    for {set f 0} {$f < $nf} {incr f} {",
        "        $mob frame $f",
        "        $all frame $f",
        "        set xf [measure fit $mob $ref]",
        "        $all move $xf",
        "    }",
        "    $ref delete",
        "    $mob delete",
        "    $all delete",
        "}",
        "fit_to_first $mid {protein and name CA}",
        "",
        "mol delrep 0 $mid",
        "mol representation NewCartoon 0.22 24.0 4.2 0",
        "mol color Structure",
        "mol selection {protein}",
        "mol material AOChalky",
        "mol addrep $mid",
        "",
        "mol representation Licorice 0.24 20.0 20.0",
        "mol color Name",
        "mol selection $ligand_sel",
        "mol material Glossy",
        "mol addrep $mid",
    ]

    if source_sel != "none":
        lines += [
            "",
            "mol representation Licorice 0.22 20.0 20.0",
            "mol color ColorID 1",
            "mol selection ($source_sel) and not hydrogen",
            "mol material Glossy",
            "mol addrep $mid",
        ]
    if target_sel != "none":
        lines += [
            "",
            "mol representation Licorice 0.22 20.0 20.0",
            "mol color ColorID 10",
            "mol selection ($target_sel) and not hydrogen",
            "mol material Glossy",
            "mol addrep $mid",
        ]
    if target_sel != "none":
        lines += [
            "",
            "mol representation QuickSurf 1.0 0.5 1.0 1.0",
            "mol color ColorID 10",
            "mol selection $target_sel",
            "mol material Transparent",
            "mol addrep $mid",
        ]

    lines += [
        "",
        "proc _clip01 {x} {",
        "    if {$x < 0.0} { return 0.0 }",
        "    if {$x > 1.0} { return 1.0 }",
        "    return $x",
        "}",
        "",
        "proc _frame_to_flow_index {frame} {",
        "    global mid flow_delta",
        "    set n [llength $flow_delta]",
        "    if {$n < 1} { return 0 }",
        "    if {[catch {set nf [molinfo $mid get numframes]}]} { return 0 }",
        "    if {$nf < 2} {",
        "        set idx 0",
        "    } else {",
        "        set idx [expr {int(round($frame * ($n - 1) / double($nf - 1)))}]",
        "    }",
        "    if {$idx < 0} { set idx 0 }",
        "    if {$idx >= $n} { set idx [expr {$n - 1}] }",
        "    return $idx",
        "}",
        "",
        "proc _interp_flow_point {alpha} {",
        "    global flow_waypoints",
        "    set n [llength $flow_waypoints]",
        "    if {$n < 1} { return {0.0 0.0 0.0} }",
        "    if {$n == 1} { return [lindex $flow_waypoints 0] }",
        "    set a [_clip01 $alpha]",
        "    set seg [expr {$n - 1}]",
        "    set x [expr {$a * $seg}]",
        "    set i [expr {int(floor($x))}]",
        "    if {$i >= $seg} {",
        "        set i [expr {$seg - 1}]",
        "        set frac 1.0",
        "    } else {",
        "        set frac [expr {$x - $i}]",
        "    }",
        "    set p0 [lindex $flow_waypoints $i]",
        "    set p1 [lindex $flow_waypoints [expr {$i + 1}]]",
        "    return [vecadd [vecscale [expr {1.0 - $frac}] $p0] [vecscale $frac $p1]]",
        "}",
        "",
        "proc draw_flow_path {mid} {",
        "    global flow_waypoints",
        "    set n [llength $flow_waypoints]",
        "    if {$n < 2} { return }",
        "    for {set i 1} {$i < $n} {incr i} {",
        "        set p0 [lindex $flow_waypoints [expr {$i - 1}]]",
        "        set p1 [lindex $flow_waypoints $i]",
        "        graphics $mid color 17",
        "        graphics $mid cylinder $p0 $p1 radius 0.18 resolution 28 filled yes",
        "    }",
        "    set p_start [lindex $flow_waypoints 0]",
        "    set p_end [lindex $flow_waypoints [expr {$n - 1}]]",
        "    graphics $mid color 1",
        "    graphics $mid sphere $p_start radius 0.34 resolution 26",
        "    graphics $mid color 10",
        "    graphics $mid sphere $p_end radius 0.34 resolution 26",
        "}",
        "",
        "proc update_flow_ball {frame} {",
        "    global mid flow_ball_id flow_delta",
        "    set idx [_frame_to_flow_index $frame]",
        "    set n [llength $flow_delta]",
        "    if {$n < 1} {",
        "        set val 0.0",
        "    } else {",
        "        set val [lindex $flow_delta $idx]",
        "    }",
        "    if {[catch {set nf [molinfo $mid get numframes]}]} {",
        "        set alpha 0.0",
        "    } elseif {$nf < 2} {",
        "        set alpha 0.0",
        "    } else {",
        "        set alpha [expr {$frame / double($nf - 1)}]",
        "    }",
        "    set p [_interp_flow_point $alpha]",
        "    if {[info exists flow_ball_id] && $flow_ball_id >= 0} {",
        "        catch {graphics $mid delete $flow_ball_id}",
        "    }",
        "    set mag [expr {abs($val)}]",
        "    set rad [expr {0.38 + 0.55 * min(1.0, $mag)}]",
        "    if {$val >= 0.0} {",
        "        graphics $mid color 1",
        "    } else {",
        "        graphics $mid color 0",
        "    }",
        "    set flow_ball_id [graphics $mid sphere $p radius $rad resolution 30]",
        "}",
        "",
        "proc _flow_canvas_xy {idx val} {",
        "    global flow_delta_ymax",
        "    set w 560",
        "    set h 290",
        "    set m 52",
        "    set n [llength $::flow_delta]",
        "    if {$n < 2} {",
        "        set frac 0.0",
        "    } else {",
        "        set frac [expr {$idx / double($n - 1)}]",
        "    }",
        "    set x [expr {$m + $frac * ($w - 2.0 * $m)}]",
        "    set yy [expr {$val / max($flow_delta_ymax, 1e-6)}]",
        "    set y [expr {$h / 2.0 - $yy * (($h / 2.0) - $m)}]",
        "    return [list $x $y]",
        "}",
        "",
        "proc setup_flow_panel {} {",
        "    global flow_delta flow_canvas flow_marker flow_frame_text flow_mag_text flow_delta_ymax flow_ready",
        "    set flow_ready 0",
        "    if {[catch {",
        "        if {[winfo exists .flowpanel]} { destroy .flowpanel }",
        "        toplevel .flowpanel",
        "        wm title .flowpanel {Mutation->Binding Flow (Delta)}",
        "        wm geometry .flowpanel 520x260+90+520",
        "        canvas .flowpanel.c -width 520 -height 260 -background #f7fbff -highlightthickness 0",
        "        pack .flowpanel.c -fill both -expand 1",
        "    } err]} {",
        "        puts [format {Flow panel unavailable: %s} $err]",
        "        return",
        "    }",
        "    set c .flowpanel.c",
        "    set flow_canvas $c",
        "    set flow_delta_ymax 1e-6",
        "    foreach v $flow_delta {",
        "        set av [expr {abs($v)}]",
        "        if {$av > $flow_delta_ymax} {",
        "            set flow_delta_ymax $av",
        "        }",
        "    }",
        "    set w 560",
        "    set h 290",
        "    set m 52",
        "    set hmid [expr {$h / 2.0}]",
        "    $c create line $m [expr {$h / 2.0}] [expr {$w - $m}] [expr {$h / 2.0}] -fill #5f7891 -width 2",
        "    $c create line $m $m $m [expr {$h - $m}] -fill #5f7891 -width 2",
        "    $c create text [expr {$w / 2.0}] 16 -text {Delta Flow Over Time (MUT - WT)} -font {Helvetica 12 bold} -fill #17324d",
        "    $c create text [expr {$w / 2.0}] [expr {$h - 16}] -text {X axis: Time window index} -font {Helvetica 11 bold} -fill #0f2a43",
        "    $c create text 78 24 -text {Y axis: Delta flow (MUT-WT)} -font {Helvetica 11 bold} -fill #0f2a43",
        "    $c create text [expr {$m - 10}] [expr {$m + 4}] -text [format {%.2f} $flow_delta_ymax] -font {Helvetica 9} -fill #577590",
        "    $c create text [expr {$m - 10}] [expr {$h - $m - 4}] -text [format {%.2f} [expr {-$flow_delta_ymax}]] -font {Helvetica 9} -fill #577590",
        "    $c create text [expr {$m + 8}] [expr {$h - $m + 12}] -text {0} -font {Helvetica 9} -fill #577590",
        "    $c create text [expr {$w - $m}] [expr {$h - $m + 12}] -text [format {%d} [expr {[llength $flow_delta] - 1}]] -font {Helvetica 9} -fill #577590",
        "    set n [llength $flow_delta]",
        "    for {set i 1} {$i < $n} {incr i} {",
        "        set v0 [lindex $flow_delta [expr {$i - 1}]]",
        "        set v1 [lindex $flow_delta $i]",
        "        lassign [_flow_canvas_xy [expr {$i - 1}] $v0] x0 y0",
        "        lassign [_flow_canvas_xy $i $v1] x1 y1",
        "        set col [expr {$v1 >= 0.0 ? \"#d94841\" : \"#3182bd\"}]",
        "        $c create line $x0 $y0 $x1 $y1 -fill $col -width 2",
        "    }",
        "    set v0 [lindex $flow_delta 0]",
        "    lassign [_flow_canvas_xy 0 $v0] x y",
        "    set flow_marker [$c create oval [expr {$x - 7}] [expr {$y - 7}] [expr {$x + 7}] [expr {$y + 7}] -fill #f59e0b -outline #7c2d12 -width 2]",
        "    set flow_frame_text [$c create text [expr {$w - 88}] 16 -text {Frame 0} -font {Helvetica 10 bold} -fill #17324d]",
        "    set flow_mag_text [$c create text [expr {$w - 90}] 36 -text {Delta 0.00} -font {Helvetica 10} -fill #2f4b69]",
        "    set flow_ready 1",
        "}",
        "",
        "proc update_flow_panel {frame} {",
        "    global mid flow_delta flow_canvas flow_marker flow_frame_text flow_mag_text flow_ready",
        "    if {![info exists flow_ready] || !$flow_ready} { return }",
        "    if {![winfo exists .flowpanel] || ![winfo exists $flow_canvas]} {",
        "        set flow_ready 0",
        "        return",
        "    }",
        "    set n [llength $flow_delta]",
        "    if {$n < 1} { return }",
        "    set idx [_frame_to_flow_index $frame]",
        "    set v [lindex $flow_delta $idx]",
        "    lassign [_flow_canvas_xy $idx $v] x y",
        "    $flow_canvas coords $flow_marker [expr {$x - 7}] [expr {$y - 7}] [expr {$x + 7}] [expr {$y + 7}]",
        "    $flow_canvas itemconfigure $flow_marker -fill [expr {$v >= 0.0 ? \"#d94841\" : \"#3182bd\"}]",
        "    $flow_canvas itemconfigure $flow_frame_text -text [format {Frame %d} $frame]",
        "    $flow_canvas itemconfigure $flow_mag_text -text [format {Delta %.3f} $v]",
        "}",
        "",
        "proc _refresh_flow_network {name index op} {",
        "    global mid",
        "    if {[catch {set f [molinfo $mid get frame]}]} { return }",
        "    update_flow_panel $f",
        "    update_flow_ball $f",
        "}",
        "proc _poll_flow_network {} {",
        "    global mid",
        "    if {[catch {set f [molinfo $mid get frame]}]} {",
        "        after 80 _poll_flow_network",
        "        return",
        "    }",
        "    update_flow_panel $f",
        "    update_flow_ball $f",
        "    after 80 _poll_flow_network",
        "}",
        "",
        "set flow_ball_id -1",
        "draw_flow_path $mid",
        "setup_flow_panel",
        "update_flow_panel 0",
        "update_flow_ball 0",
        "catch {trace remove variable vmd_frame($mid) write _refresh_flow_network}",
        "trace add variable vmd_frame($mid) write _refresh_flow_network",
        "after 80 _poll_flow_network",
        "",
        "animate style Loop",
        "animate speed 0.7",
        "animate goto 0",
        "after 200 {animate forward}",
        "display resetview",
        "puts \"Loaded residue-flow network view.\"",
        "puts \"Red spheres: mutation source residues. Cyan spheres: binding-site targets.\"",
        "puts \"Gold ball moves along the strongest inferred mutation->site route.\"",
        "puts \"Bottom panel shows dynamic Δflow score with labeled axes.\"",
    ]

    os.makedirs(os.path.dirname(output_tcl) or ".", exist_ok=True)
    with open(output_tcl, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return output_tcl
