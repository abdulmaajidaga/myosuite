"""
===============================================================================
FILE: batch_process_originals.py
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
import glob
from scipy.spatial.transform import Rotation
import time

# Add parent IK/ directory to path for helper imports
IK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if IK_DIR not in sys.path:
    sys.path.insert(0, IK_DIR)

# --- HELPER IMPORTS ---
try:
    import interactive_alignment
    import trc_data_scaler
    DEPENDENCIES_OK = True
except ImportError:
    print("Warning: Helper modules not found. Running basic mode.")
    DEPENDENCIES_OK = False

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/model/myo_sim/arm/myoarm.xml'

# Input: The folder where you saved the time-restored TRCs
# TRC_DIR = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/trc/p_originals_w_vector'
TRC_DIR = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/trc/augmented_w_vector'

# Output: Where the final MyoSuite motion files will go
OUTPUT_DIR = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/mot/augmented'

# Settings
SCALE_DATA = True          
LOCK_SHOULDER = True       
LOCKED_JOINT_KEYWORDS = ["shoulder", "clavicle", "scapula"]
# ==========================================

IKResult = collections.namedtuple('IKResult', ['qpos', 'err_norm', 'steps', 'success'])

def apply_hard_lock(sim, keywords):
    """ Locks shoulder base to prevent drift """
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
            dof_adr = raw_model.jnt_dofadr[i]
            if dof_adr != -1:
                raw_model.dof_damping[dof_adr] = 1000.0
                locked_dofs.append(dof_adr)
                
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
        if locked_dofs is not None and len(locked_dofs) > 0: dq[locked_dofs] = 0
            
        mujoco.mj_integratePos(model, data.qpos, np.clip(dq, -0.5, 0.5), 1.0)
        mujoco.mj_forward(model, data)
        
    return IKResult(data.qpos.copy(), err_norm, max_steps, False)

def process_file(sim, trc_path, out_path):
    print(f"Processing: {os.path.basename(trc_path)} ...", end="", flush=True)
    
    try:
        trc = TRCParser(trc_path)
        
        # Scale
        if SCALE_DATA and DEPENDENCIES_OK: 
            trc = trc_data_scaler.apply_retargeting(sim, trc)
        
        # Reset Sim
        mujoco.mj_resetData(sim.model.ptr, sim.data.ptr)
        sim.forward()
        
        # Lock
        locked_dofs = None
        if LOCK_SHOULDER: 
            locked_dofs = apply_hard_lock(sim, LOCKED_JOINT_KEYWORDS)

        # Align
        sid_s = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Shoulder')
        sid_e = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Elbow')
        
        r_vec = sim.data.site_xpos[sid_e] - sim.data.site_xpos[sid_s]
        t_s = trc.get_marker_data('V_Shoulder')[0] / 1000.0
        t_e = trc.get_marker_data('V_Elbow')[0] / 1000.0
        t_vec = np.array([t_e[0], -t_e[2], t_e[1]]) - np.array([t_s[0], -t_s[2], t_s[1]])
        
        rot, _ = Rotation.align_vectors(a=[r_vec], b=[t_vec])

        # IK Loop
        markers = trc.get_marker_names()
        ref_s_traj = trc.get_marker_data('V_Shoulder')
        robot_s_pos = sim.data.site_xpos[sid_s].copy()
        
        qs = []
        err_sum = 0
        
        # Warm start
        targets_0 = {}
        for m in markers:
            raw = trc.get_marker_data(m)[0]
            rel = (raw - ref_s_traj[0]) / 1000.0
            vec = np.array([rel[0], -rel[2], rel[1]])
            targets_0[m] = robot_s_pos + rot.apply(vec)
        solve_ik_multi_site(sim, targets_0, locked_dofs, 500)

        # Solve All Frames
        for i in range(trc.get_num_frames()):
            frame_t = {}
            for m in markers:
                raw = trc.get_marker_data(m)[i]
                rel = (raw - ref_s_traj[i]) / 1000.0
                vec = np.array([rel[0], -rel[2], rel[1]])
                frame_t[m] = robot_s_pos + rot.apply(vec)
            
            res = solve_ik_multi_site(sim, frame_t, locked_dofs, 50)
            qs.append(res.qpos.copy())
            err_sum += res.err_norm

        # Save
        names = [mujoco.mj_id2name(sim.model.ptr, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(sim.model.njnt)]
        times = np.arange(len(qs)) / trc.get_data_rate()
        
        with open(out_path, 'w') as f:
            f.write(f"dataset\nversion=1\nnRows={len(qs)}\nnColumns={len(names)+1}\ninDegrees=no\nendheader\n")
            f.write("time\t" + "\t".join(names) + "\n")
            for i in range(len(qs)):
                row_str = "\t".join([f"{x:.6f}" for x in qs[i]])
                f.write(f"{times[i]:.6f}\t{row_str}\n")
                
        print(f" Done. (Avg Error: {(err_sum/len(qs))*1000:.2f} mm)")
        
    except Exception as e:
        print(f" Failed! Error: {e}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Init Sim
    sim = sim_scene.SimScene.get_sim(MODEL_PATH).sim
    
    # Get Files
    files = glob.glob(os.path.join(TRC_DIR, "*.trc"))
    print(f"Found {len(files)} TRC files in {TRC_DIR}")
    
    start_time = time.time()
    for f in files:
        base = os.path.basename(f).replace('.trc', '.mot')
        out = os.path.join(OUTPUT_DIR, base)
        process_file(sim, f, out)
        
    print(f"\nBatch processing complete in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()