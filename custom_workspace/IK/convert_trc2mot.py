"""
===============================================================================
FILE: convert_trc2mot.py
===============================================================================
"""
import numpy as np
import myosuite
from myosuite.physics import sim_scene
from myosuite.utils.trc_parser import TRCParser
import os
import collections
import mujoco
import sys
from scipy.spatial.transform import Rotation

# --- HELPER IMPORTS ---
import interactive_alignment
import trc_data_scaler
# ----------------------

# CONFIG
MODEL_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/model/myo_sim/arm/myoarm.xml'
TRC_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/01_12_1.trc'
OUTPUT_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/01_12_1.mot'
REFERENCE_MOT_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/S5_12_1.mot'

INTERACTIVE_ALIGN = True # <--- KEEP FALSE FOR BATCH
SCALE_DATA = True
LOCK_SHOULDER = True
LOCKED_JOINT_KEYWORDS = ["shoulder", "clavicle", "scapula"]

if len(sys.argv) > 1:
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

def apply_hard_lock(sim, keywords):
    raw_model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    raw_data = sim.data.ptr if hasattr(sim.data, 'ptr') else sim.data
    raw_model.opt.gravity[:] = 0.0
    locked_dofs = []
    for i in range(raw_model.njnt):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not name: continue
        if any(k in name.lower() for k in keywords):
            qpos_adr = raw_model.jnt_qposadr[i]
            val = raw_data.qpos[qpos_adr]
            raw_model.jnt_range[i] = [val, val]
            raw_model.dof_damping[raw_model.jnt_dofadr[i]] = 1000.0
            locked_dofs.append(raw_model.jnt_dofadr[i])
    sim.forward()
    return np.array(locked_dofs, dtype=int)

def solve_ik_multi_site(sim, targets, locked_dofs=None, max_steps=50):
    model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    data = sim.data.ptr if hasattr(sim.data, 'ptr') else sim.data
    site_ids, target_vec = [], []
    for n, p in targets.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        if sid != -1:
            site_ids.append(sid)
            target_vec.append(p)
    if not site_ids: return IKResult(data.qpos.copy(), 999.0, 0, False)
    target_vec = np.concatenate(target_vec)
    nv = model.nv
    
    for step in range(max_steps):
        curr = np.concatenate([data.site_xpos[sid] for sid in site_ids])
        err = target_vec - curr
        err_norm = np.linalg.norm(err)
        if err_norm < 1e-3: return IKResult(data.qpos.copy(), err_norm, step, True)
        
        jac = np.zeros((3*len(site_ids), nv))
        for i, sid in enumerate(site_ids):
            mujoco.mj_jacSite(model, data, jac[3*i:3*i+3], None, sid)
            
        dq = np.linalg.solve(jac.T @ jac + np.eye(nv)*0.05, jac.T @ err)
        if locked_dofs is not None and len(locked_dofs)>0: dq[locked_dofs] = 0
        mujoco.mj_integratePos(model, data.qpos, np.clip(dq, -0.5, 0.5), 1.0)
        mujoco.mj_forward(model, data)
        
    return IKResult(data.qpos.copy(), err_norm, max_steps, False)

def main():
    if OUTPUT_PATH: os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sim = sim_scene.SimScene.get_sim(MODEL_PATH).sim
    trc = TRCParser(TRC_PATH)
    if SCALE_DATA: trc = trc_data_scaler.apply_retargeting(sim, trc)
    
    load_reference_pose_from_mot(sim, REFERENCE_MOT_PATH)
    
    locked_dofs = None
    if LOCK_SHOULDER: locked_dofs = apply_hard_lock(sim, LOCKED_JOINT_KEYWORDS)
    
    # Auto Align (Static Frame 0)
    sid_s = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Shoulder')
    sid_e = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Elbow')
    r_vec = sim.data.site_xpos[sid_e] - sim.data.site_xpos[sid_s]
    
    t_s = trc.get_marker_data('V_Shoulder')[0]/1000.0
    t_e = trc.get_marker_data('V_Elbow')[0]/1000.0
    t_vec = np.array([t_e[0], -t_e[2], t_e[1]]) - np.array([t_s[0], -t_s[2], t_s[1]])
    
    rot, _ = Rotation.align_vectors(a=[r_vec], b=[t_vec])
    if INTERACTIVE_ALIGN: rot, _ = interactive_alignment.run_interactive_alignment(sim, trc, auto_rot=rot)

    # Process
    markers = trc.get_marker_names()
    targets = []
    ref_s_traj = trc.get_marker_data('V_Shoulder')
    robot_s = sim.data.site_xpos[sid_s].copy()
    
    for i in range(trc.get_num_frames()):
        frame_t = {}
        for m in markers:
            raw = trc.get_marker_data(m)[i]
            rel = (raw - ref_s_traj[i]) / 1000.0
            vec = np.array([rel[0], -rel[2], rel[1]])
            frame_t[m] = robot_s + rot.apply(vec)
        targets.append(frame_t)

    # Solve
    qs = []
    errs = 0
    solve_ik_multi_site(sim, targets[0], locked_dofs, 500)
    
    for i, t in enumerate(targets):
        res = solve_ik_multi_site(sim, t, locked_dofs, 50)
        qs.append(res.qpos.copy())
        errs += res.err_norm
        
    print(f"FINAL_MEAN_ERROR: {(errs/len(targets))*1000:.4f}") # <--- FIXED FORMAT
    
    # Save
    names = [mujoco.mj_id2name(sim.model.ptr, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(sim.model.njnt)]
    time = np.arange(len(qs)) / trc.get_data_rate()
    with open(OUTPUT_PATH, 'w') as f:
        f.write("dataset\nversion=1\nnRows={}\nnColumns={}\ninDegrees=no\nendheader\n".format(len(qs), len(names)+1))
        f.write("time\t" + "\t".join(names) + "\n")
        for i in range(len(qs)):
            f.write(f"{time[i]:.6f}\t" + "\t".join([f"{x:.6f}" for x in qs[i]]) + "\n")

if __name__ == '__main__': main()