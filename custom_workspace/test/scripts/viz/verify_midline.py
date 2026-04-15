"""
Verify midline correction: plot wrist X trajectory against sternum midline.
If working correctly, wrist X should converge to sternum X at the drinking peak.

Usage:
  python scripts/viz/verify_midline.py                                # default FMA_50
  python scripts/viz/verify_midline.py output/generated/trc/FMA_30.trc
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import mujoco
import myosuite
from myosuite.physics import sim_scene
try:
    from myosuite.utils.trc_parser import TRCParser
except ImportError:
    from src.utils.trc_parser import TRCParser
from scipy.spatial.transform import Rotation

from src.utils.config import get_path, get
from src.inverse_kinematics import trc_data_scaler
from src.inverse_kinematics.convert_trc2mot import (
    load_reference_pose_from_mot, apply_hard_lock, solve_ik_multi_site,
    detect_drinking_peak, solve_bidirectional, apply_midline_correction
)


def get_wrist_x_trajectory(sim, qs):
    """Forward-sim each frame and record wrist site X position."""
    raw_model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    raw_data = sim.data.ptr if hasattr(sim.data, 'ptr') else sim.data
    sid = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_SITE, 'V_Wrist')
    xs = []
    for q in qs:
        raw_data.qpos[:] = q
        mujoco.mj_forward(raw_model, raw_data)
        xs.append(raw_data.site_xpos[sid][0])
    return np.array(xs)


def prepare_and_solve(trc_path):
    """Run the full NEW IK pipeline and return targets, solved qs, and metadata."""
    MODEL_PATH = get_path("mujoco_arm_model")
    REF_MOT = get_path("reference_mot")

    sim = sim_scene.SimScene.get_sim(MODEL_PATH).sim
    trc = TRCParser(trc_path)
    trc = trc_data_scaler.apply_retargeting(sim, trc)

    load_reference_pose_from_mot(sim, REF_MOT)

    lock_kw = get("pipeline", "locked_joint_keywords", default=[])
    zero_kw = get("pipeline", "zero_lock_keywords", default=[])
    zero_joints = get("pipeline", "zero_lock_joints", default=[])
    locked_dofs = apply_hard_lock(sim, lock_kw, zero_kw, zero_joints)

    site_weights = get("pipeline", "site_weights", default=None)
    peak_idx = detect_drinking_peak(trc)

    # 1-vec alignment
    sid_s = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Shoulder')
    sid_e = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Elbow')
    r_vec = sim.data.site_xpos[sid_e] - sim.data.site_xpos[sid_s]
    t_s = trc.get_marker_data('V_Shoulder')[0] / 1000.0
    t_e = trc.get_marker_data('V_Elbow')[0] / 1000.0
    t_vec = np.array([t_e[0], -t_e[2], t_e[1]]) - np.array([t_s[0], -t_s[2], t_s[1]])
    rot, _ = Rotation.align_vectors(a=[r_vec], b=[t_vec])

    # Build targets
    markers = [m for m in trc.get_marker_names() if m != 'V_Sternum']
    ref_s_traj = trc.get_marker_data('V_Shoulder')
    robot_s = sim.data.site_xpos[sid_s].copy()

    targets_raw = []
    for i in range(trc.get_num_frames()):
        frame_t = {}
        for m in markers:
            raw = trc.get_marker_data(m)[i]
            rel = (raw - ref_s_traj[i]) / 1000.0
            vec = np.array([rel[0], -rel[2], rel[1]])
            frame_t[m] = robot_s + rot.apply(vec)
        targets_raw.append(frame_t)

    # Save uncorrected target wrist X
    target_x_raw = np.array([t['V_Wrist'][0] for t in targets_raw])

    # Apply midline correction
    import copy
    targets = copy.deepcopy(targets_raw)
    raw_model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    sid_sternum = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_SITE, 'V_Sternum')
    sternum_x = sim.data.site_xpos[sid_sternum][0]
    targets = apply_midline_correction(targets, peak_idx, sternum_x)

    target_x_corrected = np.array([t['V_Wrist'][0] for t in targets])

    # Solve
    max_iters = get("ik_solver", "max_iterations", default=50)
    max_iters_first = get("ik_solver", "max_iterations_first_frame", default=500)
    retry_enabled = get("ik_solver", "retry_enabled", default=True)
    retry_max = get("ik_solver", "retry_max_attempts", default=3)
    warn_thresh = get("ik_solver", "error_warn_threshold", default=0.030)
    retry_thresh = get("ik_solver", "error_retry_threshold", default=0.100)

    qs, errs = solve_bidirectional(
        sim, targets, locked_dofs, peak_idx,
        max_iters=max_iters, max_iters_peak=max_iters_first,
        site_weights=site_weights,
        retry_enabled=retry_enabled, retry_max=retry_max,
        warn_thresh=warn_thresh, retry_thresh=retry_thresh)

    wrist_x_solved = get_wrist_x_trajectory(sim, qs)

    return {
        'target_x_raw': target_x_raw,
        'target_x_corrected': target_x_corrected,
        'wrist_x_solved': wrist_x_solved,
        'sternum_x': sternum_x,
        'peak_idx': peak_idx,
        'n_frames': trc.get_num_frames(),
    }


def main():
    PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    trc_path = os.path.join(get_path("output_dir"), "generated/trc/FMA_50.trc")
    if len(sys.argv) > 1:
        trc_path = os.path.join(PROJECT_ROOT, sys.argv[1])

    name = os.path.basename(trc_path).replace('.trc', '')
    print(f"Verifying midline correction for: {name}")

    r = prepare_and_solve(trc_path)

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    frames = np.arange(r['n_frames'])

    # Top: IK targets
    ax = axes[0]
    ax.plot(frames, r['target_x_raw'], 'r-', lw=1.5, alpha=0.6, label='Target wrist X (before correction)')
    ax.plot(frames, r['target_x_corrected'], 'g-', lw=1.5, alpha=0.8, label='Target wrist X (midline-corrected)')
    ax.axhline(r['sternum_x'], color='cyan', ls='--', lw=2, label=f'Sternum midline X = {r["sternum_x"]:.4f}')
    ax.axvline(r['peak_idx'], color='orange', ls=':', lw=1.5, alpha=0.7, label=f'Drinking peak (frame {r["peak_idx"]})')
    ax.set_ylabel('X position (m)')
    ax.set_title(f'{name} — IK Targets (before vs after midline correction)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom: solved wrist position
    ax = axes[1]
    ax.plot(frames, r['wrist_x_solved'], 'g-', lw=2, alpha=0.8, label='Solved wrist X')
    ax.axhline(r['sternum_x'], color='cyan', ls='--', lw=2, label=f'Sternum midline X = {r["sternum_x"]:.4f}')
    ax.axvline(r['peak_idx'], color='orange', ls=':', lw=1.5, alpha=0.7, label=f'Drinking peak (frame {r["peak_idx"]})')
    ax.set_xlabel('Frame')
    ax.set_ylabel('X position (m)')
    ax.set_title(f'{name} — Solved Wrist Position (after IK)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Stats
    solved_at_peak = r['wrist_x_solved'][r['peak_idx']]
    offset_mm = abs(solved_at_peak - r['sternum_x']) * 1000
    fig.text(0.5, -0.02,
             f'At peak — Solved wrist X: {solved_at_peak:.4f}m | '
             f'Sternum X: {r["sternum_x"]:.4f}m | '
             f'Offset from midline: {offset_mm:.1f}mm',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    out_dir = os.path.join(get_path("output_dir"), "generated/plots/findings")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_midline_verify.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_path}")
    print(f"  Sternum X:          {r['sternum_x']:.4f} m")
    print(f"  Solved wrist X@peak: {solved_at_peak:.4f} m  (offset: {offset_mm:.1f}mm)")


if __name__ == '__main__':
    main()
