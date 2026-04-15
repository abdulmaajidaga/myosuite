"""
===============================================================================
FILE: convert_trc2mot.py
===============================================================================
"""
import numpy as np
import myosuite
from myosuite.physics import sim_scene
try:
    from myosuite.utils.trc_parser import TRCParser
except ImportError:
    from src.utils.trc_parser import TRCParser
import os
import sys
import collections
import mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# --- HELPER IMPORTS ---
from src.inverse_kinematics import interactive_alignment
from src.inverse_kinematics import trc_data_scaler
# ----------------------

from src.utils.config import get_path, get

# CONFIG
MODEL_PATH = get_path("mujoco_arm_model")
TRC_PATH = os.path.join(get_path("output_dir"), "01_12_1.trc")
OUTPUT_PATH = os.path.join(get_path("output_dir"), "01_12_1.mot")
REFERENCE_MOT_PATH = os.environ.get("IK_REFERENCE_MOT", get_path("reference_mot"))

INTERACTIVE_ALIGN = os.environ.get("IK_INTERACTIVE_ALIGN", "True").lower() != "false"  # env override for batch
SCALE_DATA = True
LOCK_SHOULDER = True
LOCKED_JOINT_KEYWORDS = get("pipeline", "locked_joint_keywords", default=["shoulder", "clavicle", "scapula"])
ZERO_LOCK_KEYWORDS = get("pipeline", "zero_lock_keywords", default=[])
ZERO_LOCK_JOINTS = get("pipeline", "zero_lock_joints", default=[])

if len(sys.argv) > 3:
    MODEL_PATH = sys.argv[1]
    TRC_PATH = sys.argv[2]
    OUTPUT_PATH = sys.argv[3]

IKResult = collections.namedtuple('IKResult', ['qpos', 'err_norm', 'steps', 'success'])

def load_reference_pose_from_mot(sim, mot_path):
    if not os.path.exists(mot_path):
        return False
    try:
        with open(mot_path, 'r') as f:
            lines = f.readlines()
        header_end_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == 'endheader':
                header_end_idx = i
                break
        col_names = lines[header_end_idx + 1].strip().split('\t')
        first_frame_vals = lines[header_end_idx + 2].strip().split('\t')
        
        raw_model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
        raw_data = sim.data.ptr if hasattr(sim.data, 'ptr') else sim.data
        
        for i, name in enumerate(col_names):
            if name == 'time': continue
            joint_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id != -1:
                raw_data.qpos[raw_model.jnt_qposadr[joint_id]] = float(first_frame_vals[i])
        sim.forward()
        return True
    except: return False

def apply_hard_lock(sim, keywords, zero_keywords=None, zero_lock_joints=None):
    raw_model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    raw_data = sim.data.ptr if hasattr(sim.data, 'ptr') else sim.data
    raw_model.opt.gravity[:] = 0.0
    if zero_keywords is None:
        zero_keywords = []
    if zero_lock_joints is None:
        zero_lock_joints = []
    locked_dofs = []
    for i in range(raw_model.njnt):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not name: continue
        matched = any(k in name.lower() for k in keywords) or name in zero_lock_joints
        if matched:
            qpos_adr = raw_model.jnt_qposadr[i]
            # Zero out joints that should be locked straight (e.g. hand/fingers/wrist)
            if any(k in name.lower() for k in zero_keywords) or name in zero_lock_joints:
                raw_data.qpos[qpos_adr] = 0.0
            val = raw_data.qpos[qpos_adr]
            raw_model.jnt_range[i] = [val, val]
            raw_model.dof_damping[raw_model.jnt_dofadr[i]] = 1000.0
            locked_dofs.append(raw_model.jnt_dofadr[i])
    sim.forward()
    return np.array(locked_dofs, dtype=int)

def detect_drinking_peak(trc):
    """Find the frame where wrist is highest (drinking moment).
    Uses smoothed wrist Y trajectory (TRC Y = height)."""
    wrist_y = trc.get_marker_data('V_Wrist')[:, 1]  # Y = height in TRC
    # Smooth with moving average to avoid noise peaks
    kernel = min(11, len(wrist_y) // 5 * 2 + 1)  # odd, at most ~20% of signal
    if kernel >= 3:
        pad = kernel // 2
        padded = np.pad(wrist_y, pad, mode='edge')
        smoothed = np.convolve(padded, np.ones(kernel)/kernel, mode='valid')
    else:
        smoothed = wrist_y
    peak = int(np.argmax(smoothed))
    # Boundary check: if peak is at 0 or last frame, fallback to middle
    if peak <= 1 or peak >= len(wrist_y) - 2:
        peak = len(wrist_y) // 2
    return peak


def apply_midline_correction(targets, peak_idx, sternum_x):
    """Shift wrist targets toward sternum midline (X axis).
    Blend: 0 at frame 0/last, 1.0 at peak (triangular ramp)."""
    n = len(targets)
    for i in range(n):
        if 'V_Wrist' not in targets[i]:
            continue
        if i <= peak_idx:
            blend = i / max(peak_idx, 1)
        else:
            blend = (n - 1 - i) / max(n - 1 - peak_idx, 1)
        wrist = targets[i]['V_Wrist']
        targets[i]['V_Wrist'] = wrist.copy()
        targets[i]['V_Wrist'][0] += blend * (sternum_x - wrist[0])
    return targets


def _clamp_joint_limits(model, data):
    """Clamp all limited joints to their model-defined ranges."""
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            qidx = model.jnt_qposadr[j]
            lo, hi = model.jnt_range[j]
            data.qpos[qidx] = np.clip(data.qpos[qidx], lo, hi)


def _apply_joint_range_overrides(model):
    """Apply task-specific joint range overrides from config."""
    overrides = get("ik_solver", "joint_range_overrides", default=None)
    if not overrides:
        return
    for jname, (lo, hi) in overrides.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid != -1:
            model.jnt_range[jid] = [lo, hi]
            model.jnt_limited[jid] = 1
            print(f"    Joint range override: {jname} -> [{lo:.3f}, {hi:.3f}]")


def solve_ik_multi_site(sim, targets, locked_dofs=None, max_steps=None,
                        convergence_threshold=None, max_step_size=None,
                        site_weights=None, **_kwargs):
    # Load config defaults
    ik_cfg_get = lambda key, default: get("ik_solver", key, default=default)
    if max_steps is None:
        max_steps = ik_cfg_get("max_iterations", 100)
    if convergence_threshold is None:
        convergence_threshold = ik_cfg_get("convergence_threshold", 0.001)
    if max_step_size is None:
        max_step_size = ik_cfg_get("max_step_size", 0.5)

    DAMPING = 0.05  # fixed damping (proven effective in OLD solver)

    model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    data = sim.data.ptr if hasattr(sim.data, 'ptr') else sim.data
    site_ids, target_vec, weight_vec = [], [], []
    for n, p in targets.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        if sid != -1:
            w = site_weights.get(n, 1.0) if site_weights else 1.0
            if w <= 0.0:
                continue  # skip zero-weight sites (e.g. fixed-body markers)
            site_ids.append(sid)
            target_vec.append(p)
            weight_vec.extend([w, w, w])  # same weight for x, y, z
    if not site_ids: return IKResult(data.qpos.copy(), 999.0, 0, False)
    target_vec = np.concatenate(target_vec)
    weight_vec = np.array(weight_vec)
    nv = model.nv

    for step in range(max_steps):
        curr = np.concatenate([data.site_xpos[sid] for sid in site_ids])
        err = target_vec - curr
        err_norm = np.linalg.norm(err)
        if err_norm < convergence_threshold:
            return IKResult(data.qpos.copy(), err_norm, step, True)

        jac = np.zeros((3*len(site_ids), nv))
        for i, sid in enumerate(site_ids):
            mujoco.mj_jacSite(model, data, jac[3*i:3*i+3], None, sid)

        # Apply site weights to Jacobian rows and error vector
        w_jac = jac * weight_vec[:, None]
        w_err = err * weight_vec

        dq = np.linalg.solve(w_jac.T @ w_jac + np.eye(nv)*DAMPING, w_jac.T @ w_err)
        if locked_dofs is not None and len(locked_dofs) > 0:
            dq[locked_dofs] = 0
        mujoco.mj_integratePos(model, data.qpos, np.clip(dq, -max_step_size, max_step_size), 1.0)
        _clamp_joint_limits(model, data)
        mujoco.mj_forward(model, data)

    return IKResult(data.qpos.copy(), err_norm, max_steps, False)


def solve_bidirectional(sim, targets, locked_dofs, peak_idx,
                        max_iters, max_iters_peak, site_weights,
                        retry_enabled, retry_max, warn_thresh, retry_thresh):
    """Solve IK from the peak frame outward in both directions."""
    raw_data = sim.data.ptr if hasattr(sim.data, 'ptr') else sim.data
    raw_model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    n = len(targets)
    qs = [None] * n
    errs = [0.0] * n

    # 1) Solve peak frame with many iterations
    res = solve_ik_multi_site(sim, targets[peak_idx], locked_dofs,
                              max_steps=max_iters_peak, site_weights=site_weights)
    qs[peak_idx] = res.qpos.copy()
    errs[peak_idx] = res.err_norm
    last_good = res.qpos.copy()

    # Helper for retry logic
    def _solve_with_retry(frame_targets, last_good_qpos):
        res = solve_ik_multi_site(sim, frame_targets, locked_dofs,
                                  max_steps=max_iters, site_weights=site_weights)
        if retry_enabled and res.err_norm > retry_thresh:
            for attempt in range(retry_max):
                if attempt == 0:
                    res = solve_ik_multi_site(sim, frame_targets, locked_dofs,
                                              max_steps=max_iters*2, site_weights=site_weights)
                else:
                    raw_data.qpos[:] = last_good_qpos
                    mujoco.mj_forward(raw_model, raw_data)
                    res = solve_ik_multi_site(sim, frame_targets, locked_dofs,
                                              max_steps=max_iters*2, site_weights=site_weights)
                if res.err_norm <= retry_thresh:
                    break
        return res

    # 2) Forward: peak+1 → end
    raw_data.qpos[:] = qs[peak_idx]
    mujoco.mj_forward(raw_model, raw_data)
    last_good = qs[peak_idx].copy()
    for i in range(peak_idx + 1, n):
        res = _solve_with_retry(targets[i], last_good)
        qs[i] = res.qpos.copy()
        errs[i] = res.err_norm
        if res.err_norm <= warn_thresh:
            last_good = res.qpos.copy()

    # 3) Backward: peak-1 → 0
    raw_data.qpos[:] = qs[peak_idx]
    mujoco.mj_forward(raw_model, raw_data)
    last_good = qs[peak_idx].copy()
    for i in range(peak_idx - 1, -1, -1):
        res = _solve_with_retry(targets[i], last_good)
        qs[i] = res.qpos.copy()
        errs[i] = res.err_norm
        if res.err_norm <= warn_thresh:
            last_good = res.qpos.copy()

    return qs, errs


def main():
    if OUTPUT_PATH: os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sim = sim_scene.SimScene.get_sim(MODEL_PATH).sim
    trc = TRCParser(TRC_PATH)
    if SCALE_DATA: trc = trc_data_scaler.apply_retargeting(sim, trc)
    
    load_reference_pose_from_mot(sim, REFERENCE_MOT_PATH)

    # Apply task-specific joint range overrides before IK
    raw_model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    _apply_joint_range_overrides(raw_model)

    locked_dofs = None
    if LOCK_SHOULDER: locked_dofs = apply_hard_lock(sim, LOCKED_JOINT_KEYWORDS, ZERO_LOCK_KEYWORDS, ZERO_LOCK_JOINTS)
    
    # IK solver config (loaded early — needed by alignment refinement)
    ik_cfg_get = lambda key, default: get("ik_solver", key, default=default)
    max_iters = ik_cfg_get("max_iterations", 50)
    max_iters_first = ik_cfg_get("max_iterations_first_frame", 500)
    retry_enabled = ik_cfg_get("retry_enabled", True)
    retry_max = ik_cfg_get("retry_max_attempts", 3)
    warn_thresh = ik_cfg_get("error_warn_threshold", 0.030)
    retry_thresh = ik_cfg_get("error_retry_threshold", 0.100)
    fail_thresh = ik_cfg_get("error_fail_threshold", 0.200)
    target_offset = np.array(ik_cfg_get("target_offset", [0.0, 0.0, 0.0]))
    correction_deg = np.array(ik_cfg_get("alignment_correction_deg", [0.0, 0.0, 0.0]))
    correction_rot = Rotation.from_euler('xyz', correction_deg, degrees=True)

    # --- Site weights from config ---
    site_weights = get("pipeline", "site_weights", default=None)

    # --- 1-vec alignment at frame 0 (most reliable for medial reach) ---
    peak_idx = detect_drinking_peak(trc)
    print(f"  Drinking peak detected at frame {peak_idx}/{trc.get_num_frames()}")

    sid_s = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Shoulder')
    sid_e = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Elbow')

    # Exclude fixed-body markers from IK targets
    markers = [m for m in trc.get_marker_names() if m != 'V_Sternum']
    ref_s_traj = trc.get_marker_data('V_Shoulder')
    robot_s = sim.data.site_xpos[sid_s].copy()

    # 1-vec alignment: shoulder→elbow at frame 0
    r_vec = sim.data.site_xpos[sid_e] - sim.data.site_xpos[sid_s]
    t_s = trc.get_marker_data('V_Shoulder')[0] / 1000.0
    t_e = trc.get_marker_data('V_Elbow')[0] / 1000.0
    t_vec = np.array([t_e[0], -t_e[2], t_e[1]]) - np.array([t_s[0], -t_s[2], t_s[1]])
    rot, _ = Rotation.align_vectors(a=[r_vec], b=[t_vec])
    rot = correction_rot * rot

    if INTERACTIVE_ALIGN:
        rot, _ = interactive_alignment.run_interactive_alignment(sim, trc, auto_rot=rot)

    # Build targets for all frames using peak-refined rotation
    targets = []
    for i in range(trc.get_num_frames()):
        frame_t = {}
        for m in markers:
            raw = trc.get_marker_data(m)[i]
            rel = (raw - ref_s_traj[i]) / 1000.0
            vec = np.array([rel[0], -rel[2], rel[1]])
            frame_t[m] = robot_s + rot.apply(vec) + target_offset
        targets.append(frame_t)

    # Shift wrist toward sternum midline at peak (drinking reaches chest center)
    raw_model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    sid_sternum = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_SITE, 'V_Sternum')
    if sid_sternum != -1:
        sternum_x = sim.data.site_xpos[sid_sternum][0]
        targets = apply_midline_correction(targets, peak_idx, sternum_x)
        print(f"  Midline correction: wrist X → sternum X = {sternum_x:.4f}")

    # Solve bidirectionally from peak frame
    qs, frame_errors = solve_bidirectional(
        sim, targets, locked_dofs, peak_idx,
        max_iters=max_iters, max_iters_peak=max_iters_first,
        site_weights=site_weights,
        retry_enabled=retry_enabled, retry_max=retry_max,
        warn_thresh=warn_thresh, retry_thresh=retry_thresh)

    bad_frames = sum(1 for e in frame_errors if e > warn_thresh)

    # Error report
    frame_errors_mm = np.array(frame_errors) * 1000
    mean_err = np.mean(frame_errors_mm)
    p95_err = np.percentile(frame_errors_mm, 95)
    max_err = np.max(frame_errors_mm)
    warn_count = np.sum(frame_errors_mm > warn_thresh * 1000)
    fail_count = np.sum(frame_errors_mm > fail_thresh * 1000)

    print(f"FINAL_MEAN_ERROR: {mean_err:.4f}")
    print(f"  P95: {p95_err:.4f} mm | Max: {max_err:.4f} mm")
    print(f"  Frames > {warn_thresh*1000:.0f}mm: {warn_count}/{len(targets)} | "
          f"> {fail_thresh*1000:.0f}mm: {fail_count}/{len(targets)}")
    if bad_frames > 0:
        print(f"  Bad frames (after retries): {bad_frames}")
    
    # Save
    names = [mujoco.mj_id2name(sim.model.ptr, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(sim.model.njnt)]
    time = np.arange(len(qs)) / trc.get_data_rate()
    with open(OUTPUT_PATH, 'w') as f:
        f.write("dataset\nversion=1\nnRows={}\nnColumns={}\ninDegrees=no\nendheader\n".format(len(qs), len(names)+1))
        f.write("time\t" + "\t".join(names) + "\n")
        for i in range(len(qs)):
            f.write(f"{time[i]:.6f}\t" + "\t".join([f"{x:.6f}" for x in qs[i]]) + "\n")

if __name__ == '__main__': main()