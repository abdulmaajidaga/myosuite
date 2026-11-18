"""
Clean IK solver: CSV markers -> joint angles (MOT).

Minimal pipeline — no task-specific tweaks:
  - Scale data: yes (required to map real arm proportions to model)
  - Alignment: 1-vec shoulder->elbow to bring mocap into model frame
  - Shoulder lock: NO — all DOFs free, observe natural solver behaviour
  - Reference pose seed: NO — starts from model default every file
  - Bidirectional/peak solving: NO
  - Midline correction: NO
  - Joint range overrides: NO
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
from src.utils.config import get_path, get
from src.inverse_kinematics import trc_data_scaler

MODEL_PATH = get_path("mujoco_arm_model")
TRC_PATH = os.path.join(get_path("output_originals_trc"), "S5_12_1.trc")
OUTPUT_PATH = os.path.join(get_path("output_originals_mot"), "S5_12_1.mot")

if len(sys.argv) > 3:
    MODEL_PATH = sys.argv[1]
    TRC_PATH = sys.argv[2]
    OUTPUT_PATH = sys.argv[3]

IKResult = collections.namedtuple('IKResult', ['qpos', 'err_norm', 'steps', 'success'])


def _clamp_joint_limits(model, data):
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            qidx = model.jnt_qposadr[j]
            lo, hi = model.jnt_range[j]
            data.qpos[qidx] = np.clip(data.qpos[qidx], lo, hi)


def solve_ik(sim, targets, max_steps=100, convergence_threshold=0.001, max_step_size=0.5):
    DAMPING = 0.05
    model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    data = sim.data.ptr if hasattr(sim.data, 'ptr') else sim.data

    site_ids, target_vec = [], []
    for name, pos in targets.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid != -1:
            site_ids.append(sid)
            target_vec.append(pos)
    if not site_ids:
        return IKResult(data.qpos.copy(), 999.0, 0, False)

    target_vec = np.concatenate(target_vec)
    nv = model.nv

    for step in range(max_steps):
        curr = np.concatenate([data.site_xpos[sid] for sid in site_ids])
        err = target_vec - curr
        err_norm = np.linalg.norm(err)
        if err_norm < convergence_threshold:
            return IKResult(data.qpos.copy(), err_norm, step, True)

        jac = np.zeros((3 * len(site_ids), nv))
        for i, sid in enumerate(site_ids):
            mujoco.mj_jacSite(model, data, jac[3*i:3*i+3], None, sid)

        dq = np.linalg.solve(jac.T @ jac + np.eye(nv) * DAMPING, jac.T @ err)
        mujoco.mj_integratePos(model, data.qpos,
                               np.clip(dq, -max_step_size, max_step_size), 1.0)
        _clamp_joint_limits(model, data)
        mujoco.mj_forward(model, data)

    return IKResult(data.qpos.copy(), err_norm, max_steps, False)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    sim = sim_scene.SimScene.get_sim(MODEL_PATH).sim
    trc = TRCParser(TRC_PATH)
    trc = trc_data_scaler.apply_retargeting(sim, trc)  # rescale to model proportions

    ik_cfg = lambda key, default: get("ik_solver", key, default=default)
    max_iters = ik_cfg("max_iterations", 100)
    max_iters_first = ik_cfg("max_iterations_first_frame", 1000)
    warn_thresh = ik_cfg("error_warn_threshold", 0.030)
    fail_thresh = ik_cfg("error_fail_threshold", 0.200)
    max_step_size = ik_cfg("max_step_size", 0.5)
    convergence_threshold = ik_cfg("convergence_threshold", 0.001)

    # Alignment: rotate mocap frame so shoulder->elbow vector matches model
    markers = [m for m in trc.get_marker_names() if m != 'V_Sternum']
    ref_s_traj = trc.get_marker_data('V_Shoulder')
    sid_s = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Shoulder')
    sid_e = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Elbow')
    robot_s = sim.data.site_xpos[sid_s].copy()

    r_vec = sim.data.site_xpos[sid_e] - sim.data.site_xpos[sid_s]
    t_s = trc.get_marker_data('V_Shoulder')[0] / 1000.0
    t_e = trc.get_marker_data('V_Elbow')[0] / 1000.0
    t_vec = np.array([t_e[0], -t_e[2], t_e[1]]) - np.array([t_s[0], -t_s[2], t_s[1]])
    rot, _ = Rotation.align_vectors(a=[r_vec], b=[t_vec])

    # Build IK targets for every frame
    targets = []
    for i in range(trc.get_num_frames()):
        frame_t = {}
        for m in markers:
            raw = trc.get_marker_data(m)[i]
            rel = (raw - ref_s_traj[i]) / 1000.0
            vec = np.array([rel[0], -rel[2], rel[1]])
            frame_t[m] = robot_s + rot.apply(vec)
        targets.append(frame_t)

    # Sequential solve: frame 0 → end, model default pose as starting point
    n = trc.get_num_frames()
    qs, frame_errors = [], []

    for i, frame_t in enumerate(targets):
        iters = max_iters_first if i == 0 else max_iters
        res = solve_ik(sim, frame_t,
                       max_steps=iters,
                       convergence_threshold=convergence_threshold,
                       max_step_size=max_step_size)
        qs.append(res.qpos.copy())
        frame_errors.append(res.err_norm)

    errs_mm = np.array(frame_errors) * 1000
    print(f"FINAL_MEAN_ERROR: {np.mean(errs_mm):.4f}")
    print(f"  P95: {np.percentile(errs_mm, 95):.4f} mm | Max: {np.max(errs_mm):.4f} mm")
    print(f"  Frames > {warn_thresh*1000:.0f}mm: {np.sum(errs_mm > warn_thresh*1000)}/{n} | "
          f"> {fail_thresh*1000:.0f}mm: {np.sum(errs_mm > fail_thresh*1000)}/{n}")

    names = [mujoco.mj_id2name(sim.model.ptr, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(sim.model.njnt)]
    time_arr = np.arange(len(qs)) / trc.get_data_rate()
    with open(OUTPUT_PATH, 'w') as f:
        f.write("dataset\nversion=1\nnRows={}\nnColumns={}\ninDegrees=no\nendheader\n".format(
            len(qs), len(names) + 1))
        f.write("time\t" + "\t".join(names) + "\n")
        for i, q in enumerate(qs):
            f.write(f"{time_arr[i]:.6f}\t" + "\t".join(f"{x:.6f}" for x in q) + "\n")


if __name__ == '__main__':
    main()
