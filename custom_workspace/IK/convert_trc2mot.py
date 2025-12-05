# import sys
# import os
# # This adds the root of the cloned repo to Python's search path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
 
# do this if myosuite is not found
#cd /home/abdul/Documents/myosuite/
#pip install -e .

import numpy as np
import myosuite
from myosuite.physics import sim_scene
from myosuite.utils.trc_parser import TRCParser
import os
import collections
import mujoco
import time
from scipy.spatial.transform import Rotation

# ========================================
# CONFIGURATION - Edit these paths
# ========================================
MODEL_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/model/myo_sim/arm/myoarm.xml'  # Path to MuJoCo model XML
TRC_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/S5_12_1.trc'  # Path to input TRC file
OUTPUT_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/S5_12_1.mot'  # Path to output MOT file (leave as None if only visualizing)
VISUALIZE = False  # Set to True to show real-time alignment visualization instead of solving IK
# ========================================

IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success'])

def solve_ik_multi_site(sim, site_targets, tol=1e-5, max_steps=500, regularization_strength=0.01):
    model = sim.model.ptr
    data = sim.data.ptr
    initial_qpos = data.qpos.copy()
    site_ids, target_pos_vec = [], []
    for name, pos in site_targets.items():
        try:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid == -1: raise KeyError
            site_ids.append(sid)
            target_pos_vec.append(pos)
        except KeyError:
            return None
    target_pos_vec = np.concatenate(target_pos_vec)
    for i in range(max_steps):
        current_pos = np.concatenate([data.site_xpos[sid] for sid in site_ids])
        jac_stack = np.zeros((3 * len(site_ids), model.nv))
        for idx, sid in enumerate(site_ids):
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, sid)
            jac_stack[3*idx:3*idx+3, :] = jacp
        pos_error = target_pos_vec - current_pos
        err_norm = np.linalg.norm(pos_error)
        if err_norm < tol:
            return IKResult(qpos=data.qpos.copy(), err_norm=err_norm, steps=i, success=True)
        hess = jac_stack.T @ jac_stack + np.eye(model.nv) * regularization_strength
        dq = np.linalg.solve(hess, jac_stack.T @ pos_error)
        mujoco.mj_integratePos(model, data.qpos, dq, 1.0)
        mujoco.mj_forward(model, data)
    return IKResult(qpos=data.qpos.copy(), err_norm=err_norm, steps=max_steps, success=False)

def visualize_alignment(sim, trajectories):
    print("\nStarting Alignment Visualization...")
    print("Red=Shoulder, Green=Elbow, Blue=Wrist")
    viewer = mujoco.viewer.launch_passive(sim.model.ptr, sim.data.ptr)
    colors = [[1, 0, 0, 0.7], [0, 1, 0, 0.7], [0, 0, 1, 0.7]]
    marker_names = list(trajectories.keys())
    for i in range(len(marker_names)):
        mujoco.mjv_initGeom(viewer.user_scn.geoms[i], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], pos=np.zeros(3), mat=np.eye(3).flatten(), rgba=colors[i])
    viewer.user_scn.ngeom = len(marker_names)
    num_frames = len(trajectories[marker_names[0]])
    while viewer.is_running():
        for i in range(num_frames):
            if not viewer.is_running(): break
            for marker_idx, name in enumerate(marker_names):
                viewer.user_scn.geoms[marker_idx].pos = trajectories[name][i]
            viewer.sync()
            time.sleep(1 / 100)
    viewer.close()

def main():
    # Create output directory if needed
    if OUTPUT_PATH:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    sim_wrapper = sim_scene.SimScene.get_sim(MODEL_PATH)
    sim = sim_wrapper.sim
    print(f"✓ Model loaded from: {MODEL_PATH}")
    
    trc_data = TRCParser(TRC_PATH)
    marker_names = trc_data.get_marker_names()
    print(f"✓ TRC file loaded from: {TRC_PATH}")

    print("\nCalculating optimal alignment...")
    model_s_pos = sim.data.site_xpos[sim.model.site('V_Shoulder').id].copy()
    model_e_pos = sim.data.site_xpos[sim.model.site('V_Elbow').id].copy()
    model_vec = model_e_pos - model_s_pos
    mocap_s_pos_raw = trc_data.get_marker_data('V_Shoulder')[0] / 1000.0
    mocap_e_pos_raw = trc_data.get_marker_data('V_Elbow')[0] / 1000.0
    mocap_s_pos = np.array([mocap_s_pos_raw[0], -mocap_s_pos_raw[2], mocap_s_pos_raw[1]])
    mocap_e_pos = np.array([mocap_e_pos_raw[0], -mocap_e_pos_raw[2], mocap_e_pos_raw[1]])
    mocap_vec = mocap_e_pos - mocap_s_pos
    rotation, _ = Rotation.align_vectors(a=[model_vec], b=[mocap_vec])
    print("✓ Optimal rotation calculated.")

    mocap_shoulder_origin = trc_data.get_marker_data('V_Shoulder')[0]
    processed_trajectories = {}
    for name in marker_names:
        relative_pos_m = (trc_data.get_marker_data(name) - mocap_shoulder_origin) / 1000.0
        processed_trajectories[name] = np.array([model_s_pos + rotation.apply([p[0], -p[2], p[1]]) for p in relative_pos_m])
    print("✓ Mocap data aligned to model coordinate system.")
    
    if VISUALIZE:
        visualize_alignment(sim, processed_trajectories)
        return

    if not OUTPUT_PATH:
        print("✗ ERROR: OUTPUT_PATH must be specified when not visualizing.")
        return

    print("\nStarting Inverse Kinematics solve...")
    num_frames = trc_data.get_num_frames()
    joint_pos_trajectory = []
    all_joint_names = [sim.model.joint(i).name for i in range(sim.model.njnt)]

    print("  > Solving first frame with increased iterations...")
    first_frame_targets = {name: processed_trajectories[name][0] for name in marker_names}
    ik_result = solve_ik_multi_site(sim, first_frame_targets, max_steps=1000)
    
    sim.data.qpos[:] = ik_result.qpos
    sim.forward()
    joint_pos_trajectory.append(ik_result.qpos)
    print(f"  > Initial solve completed (Error: {ik_result.err_norm*1000:.2f} mm). Proceeding...")

    for i in range(1, num_frames):
        target_pos_dict = {name: processed_trajectories[name][i] for name in marker_names}
        ik_result = solve_ik_multi_site(sim, target_pos_dict)
        joint_pos_trajectory.append(ik_result.qpos)
        sim.data.qpos[:] = ik_result.qpos
        sim.forward()
        if i % 100 == 0 or i == num_frames - 1:
            print(f"  > Solved frame {i+1}/{num_frames} (Error: {ik_result.err_norm*1000:.2f} mm)")

    print("✓ Inverse Kinematics solve completed.")

    try:
        qpos_trajectory = np.array(joint_pos_trajectory)
        time = np.arange(num_frames) / trc_data.get_data_rate()
        with open(OUTPUT_PATH, 'w') as f:
            f.write(f"{os.path.basename(OUTPUT_PATH)}\nversion=1\n")
            f.write(f"nRows={num_frames}\nnColumns={len(all_joint_names) + 1}\n")
            f.write("inDegrees=no\nendheader\n")
            f.write("time\t" + "\t".join(all_joint_names) + "\n")
            for i in range(num_frames):
                f.write(f"{time[i]:.8f}\t" + "\t".join([f"{q:.8f}" for q in qpos_trajectory[i]]) + "\n")
        print(f"✓ Results successfully saved to: {OUTPUT_PATH}")
    except Exception as e:
        print(f"✗ ERROR: Failed to save MOT file: {e}")

if __name__ == '__main__':
    main()
