"""
Clean IK solver: CSV markers -> joint angles (MOT).

Scale data to model proportions, align shoulder->elbow vector to model frame,
lock phantom (correction) joints at 0, multi-start frame 0, warm-started sequential.
V_Sternum excluded — it's a chest-fixed marker; the shoulder-relative coordinate
frame used for arm markers maps it to the wrong position and breaks convergence.

Weighted Jacobian IK: per-DOF diagonal damping encodes CNS distal-to-proximal
cost hierarchy (Cirstea & Levin 2000).  Arm DOFs are cheap (0.05), SC girdle
is penalised 4x (0.20).  Trunk is prescribed from CS plate normal before each
solve — TR_DAMPING is unused and kept only as a CLI placeholder.
"""
import os
import sys
import collections
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import myosuite
from myosuite.physics import sim_scene
try:
    from myosuite.utils.trc_parser import TRCParser
except ImportError:
    from src.utils.trc_parser import TRCParser

from src.utils.config import get_path, get, get_project_root
from src.inverse_kinematics import trc_data_scaler

MODEL_PATH = get_path("mujoco_arm_model").replace("myoarm.xml", "myoarm_trunk.xml")
TRC_PATH = os.path.join(get_path("output_originals_trc"), "S5_12_1.trc")
OUTPUT_PATH = os.path.join(get_path("output_originals_mot"), "S5_12_1.mot")

MAX_ITERS_OVERRIDE = None
if len(sys.argv) > 3:
    MODEL_PATH = sys.argv[1]
    TRC_PATH = sys.argv[2]
    OUTPUT_PATH = sys.argv[3]
if len(sys.argv) > 4 and sys.argv[4] != "0":
    MAX_ITERS_OVERRIDE = int(sys.argv[4])

IKResult = collections.namedtuple('IKResult', ['qpos', 'err_norm', 'steps', 'success'])

DAMPING    = 0.05   # base — arm DOFs (Tier 1)
SC_DAMPING = float(sys.argv[6]) if len(sys.argv) > 6 else 0.20   # shoulder girdle (Tier 2)
TR_DAMPING = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0    # trunk (Tier 3, unused — trunk is pinned)

N_STARTS = 20
IK_SKIP_MARKERS = {'V_Sternum'}

# CS markers are evaluated in the shoulder-relative frame to drive trunk_forward_lean.
CS_GLOBAL_MARKERS = {'CS_1', 'CS_2', 'CS_3', 'CS_4'}

# Grouped joint lists used to build per-DOF damping inside solve_ik.
TRUNK_JOINTS = ['trunk_forward_lean']
SC_JOINTS    = ['sternoclavicular_r2', 'sternoclavicular_r3']

# Mathematical correction joints — pinned at 0, excluded from Jacobian.
# trunk_lateral_lean is pinned: drinking task is sagittal; lateral lean is not
# a documented compensatory strategy for hemiparetic reaching.
PHANTOM_JOINTS = [
    'unrotscap_r2', 'unrotscap_r3',
    'acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3',
    'unrothum_r1', 'unrothum_r2', 'unrothum_r3',
    'shoulder1_r2',
    'trunk_lateral_lean',
]


def _unwrap(sim):
    model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    data  = sim.data.ptr  if hasattr(sim.data,  'ptr') else sim.data
    return model, data


def _get_dof_indices(model, joint_names):
    """Return DOF address array for the given joint names (silently skips missing)."""
    idxs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid != -1:
            idxs.append(model.jnt_dofadr[jid])
    return np.array(idxs, dtype=int)


def _build_phantom_arrays(model):
    """Return (dof_indices, qpos_indices) for phantom joints using _get_dof_indices."""
    dof_idxs = _get_dof_indices(model, PHANTOM_JOINTS)
    qpos_idxs = []
    for name in PHANTOM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid != -1:
            qpos_idxs.append(model.jnt_qposadr[jid])
    return dof_idxs, np.array(qpos_idxs, dtype=int)


def _build_limit_arrays(model):
    """Return (qpos_indices, lows, highs) for vectorized joint-limit clamping."""
    qidxs, lows, highs = [], [], []
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            qidxs.append(model.jnt_qposadr[j])
            lo, hi = model.jnt_range[j]
            lows.append(lo)
            highs.append(hi)
    return np.array(qidxs, dtype=int), np.array(lows), np.array(highs)


def _build_ik_cache(model, marker_names, dof_idxs):
    """Precompute site_ids and dof_mask (constant across all frames)."""
    site_ids, ordered_names = [], []
    for name in marker_names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid != -1:
            site_ids.append(sid)
            ordered_names.append(name)
    dof_mask = np.ones(model.nv)
    dof_mask[dof_idxs] = 0.0
    return site_ids, ordered_names, dof_mask


def solve_ik(model, data, target_vec, site_ids, dof_mask, phantom_qpos_idxs,
             limit_qidxs, limit_lo, limit_hi, sc_dofs, trunk_dofs,
             max_steps=100, convergence_threshold=0.001, max_step_size=0.5,
             pinned_dofs=None, n_cs_sites=0):
    """
    n_cs_sites: CS markers at the END of site_ids.
    Only their Jacobian rows drive trunk DOFs — arm-marker rows have trunk
    columns zeroed to prevent arm residuals from competing with CS signal.
    """
    nv = model.nv
    n_sites = len(site_ids)
    n_arm   = n_sites - n_cs_sites

    if pinned_dofs is not None and len(pinned_dofs):
        working_mask = dof_mask.copy()
        working_mask[pinned_dofs] = 0.0
    else:
        working_mask = dof_mask

    damping_diag = np.ones(nv) * DAMPING
    if sc_dofs.size:
        damping_diag[sc_dofs]    = SC_DAMPING
    if trunk_dofs.size:
        damping_diag[trunk_dofs] = TR_DAMPING
    damping_matrix = np.diag(damping_diag)

    for step in range(max_steps):
        if pinned_dofs is not None and len(pinned_dofs):
            data.qpos[pinned_dofs] = 0.0
        data.qpos[phantom_qpos_idxs] = 0.0
        mujoco.mj_forward(model, data)

        curr     = np.concatenate([data.site_xpos[sid] for sid in site_ids])
        err      = target_vec - curr
        err_norm = np.linalg.norm(err)
        if err_norm < convergence_threshold:
            return IKResult(data.qpos.copy(), err_norm, step, True)

        jac = np.zeros((3 * n_sites, nv))
        for i, sid in enumerate(site_ids):
            mujoco.mj_jacSite(model, data, jac[3*i:3*i+3], None, sid)
        jac *= working_mask[np.newaxis, :]

        if n_cs_sites > 0 and trunk_dofs.size:
            for d in trunk_dofs:
                jac[:3 * n_arm, d] = 0.0

        dq = np.linalg.solve(jac.T @ jac + damping_matrix, jac.T @ err)
        dq *= working_mask
        mujoco.mj_integratePos(model, data.qpos, np.clip(dq, -max_step_size, max_step_size), 1.0)
        data.qpos[limit_qidxs] = np.clip(data.qpos[limit_qidxs], limit_lo, limit_hi)
        mujoco.mj_forward(model, data)

    return IKResult(data.qpos.copy(), err_norm, max_steps, False)


def _load_cs_tilt(trc_path, n_frames):
    """
    Load the matching raw CSV, compute per-frame CS plate forward tilt (radians)
    using the SVD normal method, and return a delta-from-frame-0 array clipped to
    the joint range.  Returns None if CSV cannot be found or CS markers are missing.
    """
    group_dir = os.path.dirname(os.path.dirname(os.path.abspath(trc_path)))
    group     = os.path.basename(group_dir)
    root_dir  = get_project_root()
    trial     = os.path.splitext(os.path.basename(trc_path))[0]
    csv_path  = None
    for candidate in [
        os.path.join(root_dir, 'data', 'kinematic', group, trial + '.csv'),
        os.path.join(root_dir, 'data', group, trial + '.csv'),
    ]:
        if os.path.exists(candidate):
            csv_path = candidate
            break
    if csv_path is None:
        print(f"  [CS plate] no CSV found for {trial}; trunk will be zero")
        return None
    try:
        import pandas as pd
        from scripts.inject_cs_trunk import load_csv, wrist_fwd_axis, cs_lean_rad
        df      = load_csv(csv_path)
        fwd     = wrist_fwd_axis(df)
        tilt    = cs_lean_rad(df, fwd)
        if tilt is None:
            print(f"  [CS plate] <3 CS markers in {trial}")
            return None
        baseline = float(np.nanmean(tilt[:5]))
        delta    = tilt - baseline
        delta    = pd.Series(delta).interpolate(limit_direction='both').values
        nf       = min(len(delta), n_frames)
        out      = np.zeros(n_frames)
        out[:nf] = np.clip(delta[:nf], -0.70, 0.70)
        print(f"  [CS plate] {trial}: range [{np.degrees(out.min()):.1f}, {np.degrees(out.max()):.1f}]°")
        return out
    except Exception as e:
        print(f"  [CS plate] failed ({e}); trunk will be zero")
        return None


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    sim = sim_scene.SimScene.get_sim(MODEL_PATH).sim
    trc = TRCParser(TRC_PATH)
    trc = trc_data_scaler.apply_retargeting(sim, trc)

    max_iters       = MAX_ITERS_OVERRIDE or get("ik_solver", "max_iterations", default=100)
    max_iters_first = MAX_ITERS_OVERRIDE * 3 if MAX_ITERS_OVERRIDE else get("ik_solver", "max_iterations_first_frame", default=1000)
    warn_thresh     = get("ik_solver", "error_warn_threshold",  default=0.030)
    fail_thresh     = get("ik_solver", "error_fail_threshold",  default=0.200)
    max_step_size   = get("ik_solver", "max_step_size",         default=0.5)
    convergence_threshold = get("ik_solver", "convergence_threshold", default=0.001)

    model_ptr, data_ptr = _unwrap(sim)

    # Alignment: rotate mocap frame so shoulder->elbow vector matches model
    markers   = trc.get_marker_names()
    ref_s_traj = trc.get_marker_data('V_Shoulder')
    sid_s = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Shoulder')
    sid_e = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Elbow')
    robot_s = data_ptr.site_xpos[sid_s].copy()

    r_vec = data_ptr.site_xpos[sid_e] - data_ptr.site_xpos[sid_s]
    t_s   = ref_s_traj[0] / 1000.0
    t_e   = trc.get_marker_data('V_Elbow')[0] / 1000.0
    t_vec = np.array([t_e[0], -t_e[2], t_e[1]]) - np.array([t_s[0], -t_s[2], t_s[1]])
    rot, _ = Rotation.align_vectors(a=[r_vec], b=[t_vec])

    active_markers = [m for m in markers if m not in IK_SKIP_MARKERS]

    # Precompute IK arrays — constant across frames and files
    phantom_dofs, phantom_qpos_idxs = _build_phantom_arrays(model_ptr)
    sc_dofs    = _get_dof_indices(model_ptr, SC_JOINTS)
    trunk_dofs = _get_dof_indices(model_ptr, TRUNK_JOINTS)
    limit_qidxs, limit_lo, limit_hi = _build_limit_arrays(model_ptr)
    site_ids, ordered_names, dof_mask = _build_ik_cache(model_ptr, active_markers, phantom_dofs)

    # Default model CS site positions (trunk=0, for delta-anchored targets)
    mujoco.mj_forward(model_ptr, data_ptr)
    model_cs_default = {}
    for name in ordered_names:
        if name in CS_GLOBAL_MARKERS:
            sid = mujoco.mj_name2id(model_ptr, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid != -1:
                model_cs_default[name] = data_ptr.site_xpos[sid].copy()

    # Build per-frame target position arrays
    marker_trajs = {m: trc.get_marker_data(m) for m in active_markers}
    n = trc.get_num_frames()

    # Frame-0 CS-shoulder offsets: anatomical baseline removed via delta motion
    cs_initial_rel = {
        name: (marker_trajs[name][0] - ref_s_traj[0]) / 1000.0
        for name in ordered_names if name in CS_GLOBAL_MARKERS
    }

    target_vecs = []
    for i in range(n):
        positions = []
        for name in ordered_names:
            raw = marker_trajs[name][i]
            if name in CS_GLOBAL_MARKERS:
                # Delta motion from frame 0: strips anatomical baseline so that
                # the trunk angle at frame 0 is 0 by construction. Only the CHANGE
                # in CS position relative to shoulder drives trunk_forward_lean.
                cs_rel_i = (raw - ref_s_traj[i]) / 1000.0
                delta = cs_rel_i - cs_initial_rel[name]
                delta_model = np.array([delta[0], -delta[2], delta[1]])
                anchor = model_cs_default.get(name, robot_s)
                positions.append(anchor + delta_model)
            else:
                rel = (raw - ref_s_traj[i]) / 1000.0
                vec = np.array([rel[0], -rel[2], rel[1]])
                positions.append(robot_s + rot.apply(vec))
        target_vecs.append(np.concatenate(positions))

    default_qpos = data_ptr.qpos.copy()
    RNG = np.random.default_rng(42)

    phantom_qpos_set = set(phantom_qpos_idxs.tolist())
    # For hinge joints jnt_dofadr == jnt_qposadr, so trunk_dofs doubles as
    # qpos addresses. Exclude trunk from random initialisation: frame 0 always
    # starts with trunk at 0 to enforce the neutral-start assumption.
    trunk_qpos_set = set(trunk_dofs.tolist())
    free_joints = [
        (model_ptr.jnt_qposadr[j],
         bool(model_ptr.jnt_limited[j]),
         float(model_ptr.jnt_range[j][0]),
         float(model_ptr.jnt_range[j][1]))
        for j in range(model_ptr.njnt)
        if model_ptr.jnt_qposadr[j] not in phantom_qpos_set
        and model_ptr.jnt_qposadr[j] not in trunk_qpos_set
    ]

    def _random_qpos():
        q = default_qpos.copy()
        q[trunk_dofs] = 0.0          # trunk always neutral at frame 0
        for qidx, limited, lo, hi in free_joints:
            q[qidx] = RNG.uniform(lo, hi) if limited else RNG.uniform(-1.0, 1.0)
        return q

    # Load CS plate tilt trajectory — used as direct trunk signal in Pass 1
    cs_tilt_rad = _load_cs_tilt(TRC_PATH, n)

    # Two-pass IK: Pass 1 sets trunk from CS plate normal (closed-form);
    # Pass 2 solves arm/SC DOFs with trunk pinned.
    arm_only_mask = dof_mask.copy(); arm_only_mask[trunk_dofs] = 0.0

    def _set_trunk(frame_i):
        """Set trunk DOFs directly from CS plate tilt at this frame."""
        val = float(cs_tilt_rad[frame_i]) if cs_tilt_rad is not None else 0.0
        data_ptr.qpos[trunk_dofs] = val
        mujoco.mj_forward(model_ptr, data_ptr)
        return val

    def _solve_arms(tv, max_s):
        return solve_ik(model_ptr, data_ptr, tv, site_ids,
                        arm_only_mask, phantom_qpos_idxs,
                        limit_qidxs, limit_lo, limit_hi, sc_dofs, trunk_dofs,
                        max_steps=max_s, convergence_threshold=convergence_threshold,
                        max_step_size=max_step_size, n_cs_sites=0)

    def _solve_twopass(tv, frame_i, max_s):
        trunk_val = _set_trunk(frame_i)
        res = _solve_arms(tv, max_s)
        data_ptr.qpos[trunk_dofs] = trunk_val  # restore after arm IK
        mujoco.mj_forward(model_ptr, data_ptr)
        return IKResult(data_ptr.qpos.copy(), res.err_norm, res.steps, res.success)

    def solve_frame0_multistart(target_vec):
        trunk_val = _set_trunk(0)
        print(f"  Frame 0 trunk={np.degrees(trunk_val):.2f}°")
        best_a = None
        for s in range(N_STARTS):
            q = default_qpos.copy() if s == 0 else _random_qpos()
            q[trunk_dofs] = trunk_val
            data_ptr.qpos[:] = q
            mujoco.mj_forward(model_ptr, data_ptr)
            res = _solve_arms(target_vec, max_iters_first)
            if best_a is None or res.err_norm < best_a.err_norm:
                best_a = res
        final_qpos = best_a.qpos.copy()
        final_qpos[trunk_dofs] = trunk_val
        data_ptr.qpos[:] = final_qpos
        mujoco.mj_forward(model_ptr, data_ptr)
        return IKResult(final_qpos, best_a.err_norm, best_a.steps, best_a.success)

    qs, frame_errors = [], []
    for i, tv in enumerate(target_vecs):
        if i == 0:
            res = solve_frame0_multistart(tv)
            print(f"  Frame 0 best ({N_STARTS} starts): {res.err_norm*1000:.2f} mm")
        else:
            res = _solve_twopass(tv, i, max_iters)
        qs.append(res.qpos.copy())
        frame_errors.append(res.err_norm)

    errs_mm = np.array(frame_errors) * 1000
    print(f"FINAL_MEAN_ERROR: {np.mean(errs_mm):.4f}")
    print(f"  P95: {np.percentile(errs_mm, 95):.4f} mm | Max: {np.max(errs_mm):.4f} mm")
    print(f"  Frames > {warn_thresh*1000:.0f}mm: {np.sum(errs_mm > warn_thresh*1000)}/{n} | "
          f"> {fail_thresh*1000:.0f}mm: {np.sum(errs_mm > fail_thresh*1000)}/{n}")

    names = [mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(model_ptr.njnt)]
    time_arr = np.arange(len(qs)) / trc.get_data_rate()
    with open(OUTPUT_PATH, 'w') as f:
        f.write("dataset\nversion=1\nnRows={}\nnColumns={}\ninDegrees=no\nendheader\n".format(
            len(qs), len(names) + 1))
        f.write("time\t" + "\t".join(names) + "\n")
        for i, q in enumerate(qs):
            f.write(f"{time_arr[i]:.6f}\t" + "\t".join(f"{x:.6f}" for x in q) + "\n")


if __name__ == '__main__':
    main()
