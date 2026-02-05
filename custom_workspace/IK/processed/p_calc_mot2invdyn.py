import os
import glob
import mujoco
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import osqp
import matplotlib.pyplot as plt
import skvideo.io
from PIL import Image, ImageDraw
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_XML_PATH = r"/home/abdul/Desktop/myosuite/custom_workspace/model/myo_sim/arm/myoarm.xml"
INPUT_PATH = r"/home/abdul/Desktop/myosuite/custom_workspace/IK/output/mot/p_originals"
OUTPUT_ROOT = r"/home/abdul/Desktop/myosuite/custom_workspace/IK/output/ID_results"

# 1. SOLVER WHITELIST (Muscles are solved for these joints)
SOLVE_JOINTS = ['elbow_flexion', 'pro_sup', 'deviation', 'flexion']

# 2. SCALING CONFIGURATION
# Shoulder was 1.8 Million. We multiply by 1e-5 to get ~18 Nm (Realistic)
SHOULDER_SCALE = 0.00001 

# Elbow is auto-calibrated below to hit ~12 Nm.
TARGET_PEAK_TORQUE = 12.0 

# List of high-torque joints to apply SHOULDER_SCALE to
BROKEN_JOINTS = [
    'elv_angle', 'shoulder_elv', 'shoulder1_r2', 'shoulder_rot',
    'sternoclavicular_r2', 'sternoclavicular_r3', 
    'unrotscap_r3', 'unrotscap_r2',
    'acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3',
    'unrothum_r1', 'unrothum_r2', 'unrothum_r3'
]

# SETTINGS
GENERATE_VIDEO = True
LOW_PASS_CUTOFF = 1.5

# =============================================================================
# UTILITY
# =============================================================================

def read_mot_file(filepath):
    if not os.path.exists(filepath): return None
    skiprows = 0
    with open(filepath, "r") as file:
        for line in file:
            if "endheader" in line: break
            skiprows += 1
    return pd.read_csv(filepath, sep=r'\s+', skiprows=skiprows + 1)

def compute_derivatives(data, dt):
    vel = np.gradient(data, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc

def apply_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1: normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if data.ndim == 1: return filtfilt(b, a, data)
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])
    return filtered

# =============================================================================
# SOLVER
# =============================================================================

def solve_muscle_activations_sliced(model, data, tau_full, target_indices):
    n_muscles = model.nu
    n_targets = len(target_indices)
    
    data.qvel[:] = 0.0 
    mujoco.mj_forward(model, data)
    moment_arm_full = data.actuator_moment.reshape(model.nv, n_muscles)
    
    # Slice matrices to only solve for Target Joints (Elbow/Wrist)
    # This prevents the Shoulder torque (even if scaled) from interfering 
    # with the clean elbow solution.
    moment_arm_sliced = moment_arm_full[target_indices, :]
    tau_sliced = tau_full[target_indices]
    
    epsilon = 0.001 
    A = np.hstack([moment_arm_sliced, epsilon * np.eye(n_targets)])
    lambda_damp = 1e-4
    
    try:
        gram = A @ A.T + lambda_damp * np.eye(n_targets)
        x_sol = A.T @ np.linalg.solve(gram, tau_sliced)
        return np.clip(x_sol[:n_muscles], 0, 1)
    except np.linalg.LinAlgError:
        return np.zeros(n_muscles)

# =============================================================================
# VISUALIZER
# =============================================================================

def render_video(model, motion_df, activations, out_dir):
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth, camera.elevation, camera.distance = 45, -30, 1.8
    camera.lookat = [0.0, 0.0, 0.9]
    data = mujoco.MjData(model)
    frames = []
    
    geom_map, site_map, tendon_map = {}, {}, {}
    for i in range(model.nu):
        act_name = model.actuator(i).name
        geom_map[i] = [g for g in range(model.ngeom) if act_name in (model.geom(g).name or "")]
        site_map[i] = [s for s in range(model.nsite) if act_name in (model.site(s).name or "")]
        tendon_map[i] = [t for t in range(model.ntendon) if act_name in (model.tendon(t).name or "")]

    print("   -> Rendering Video...")
    step = 3 
    
    for i in tqdm(range(0, len(motion_df), step), leave=False):
        for col in motion_df.columns:
            if col == 'time': continue
            try:
                jid = model.joint(col).id
                data.qpos[model.jnt_qposadr[jid]] = motion_df[col].iloc[i]
            except: pass
        mujoco.mj_forward(model, data)
        
        for mid, act in enumerate(activations[i]):
            # Color: Blue (0) -> Purple (0.5) -> Red (1.0)
            if act < 0.5:
                color = (1 - 2*act)*np.array([0.2, 0.2, 0.8, 1]) + (2*act)*np.array([0.6, 0.2, 0.6, 1])
            else:
                color = (1 - 2*(act-0.5))*np.array([0.6, 0.2, 0.6, 1]) + (2*(act-0.5))*np.array([0.9, 0.1, 0.1, 1])
            
            for gid in geom_map.get(mid, []): model.geom_rgba[gid] = color
            for sid in site_map.get(mid, []): model.site_rgba[sid] = color
            for tid in tendon_map.get(mid, []): model.tendon_rgba[tid] = color 

        renderer.update_scene(data, camera=camera)
        img = Image.fromarray(renderer.render())
        ImageDraw.Draw(img).text((10,10), f"Time: {motion_df['time'].iloc[i]:.2f}s", fill='white')
        frames.append(np.array(img))
        
    skvideo.io.vwrite(os.path.join(out_dir, 'muscle_activity.mp4'), np.array(frames), 
                      inputdict={'-r':'60'}, outputdict={'-r':'60', '-pix_fmt':'yuv420p'})

# =============================================================================
# MAIN PROCESS
# =============================================================================

def process_file(mot_path, model_path, output_dir):
    file_id = os.path.basename(mot_path).replace('.mot', '')
    session_dir = os.path.join(output_dir, file_id)
    os.makedirs(session_dir, exist_ok=True)
    
    print(f"\nProcessing: {file_id}")
    
    df = read_mot_file(mot_path)
    if df is None: return
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    model_joints = [model.joint(j).name for j in range(model.njnt)]
    valid_cols = [c for c in df.columns if c in model_joints]
    
    # 1. Map Joints & Indices
    target_dof_indices = []
    broken_dof_indices = []
    
    # Map Solver Joints (Elbow/Wrist)
    for jname in SOLVE_JOINTS:
        if jname in model_joints:
             jid = model.joint(jname).id
             dof_adr = model.jnt_dofadr[jid]
             target_dof_indices.append(dof_adr)
             
    # Map Broken Joints (Shoulder) for Scaling
    for jname in BROKEN_JOINTS:
        if jname in model_joints:
            jid = model.joint(jname).id
            jtype = model.jnt_type[jid]
            dim = 1
            if jtype == 0: dim = 6
            elif jtype == 1: dim = 3
            
            dof_start = model.jnt_dofadr[jid]
            for d in range(dof_start, dof_start + dim):
                broken_dof_indices.append(d)

    # 2. Kinematics
    time = df['time'].values
    dt = np.mean(np.diff(time))
    if dt == 0: dt = 0.005
    fs = 1.0 / dt
    
    qpos = df[valid_cols].values
    qvel, qacc = compute_derivatives(qpos, dt)
    qvel = apply_lowpass_filter(qvel, LOW_PASS_CUTOFF, fs)
    qacc = apply_lowpass_filter(qacc, LOW_PASS_CUTOFF, fs)
    
    # 3. AUTO-CALIBRATION
    print("   -> Auto-Calibrating Physics...")
    raw_torques = []
    for i in range(0, len(time), 10):
        for jname, val, vel, acc in zip(valid_cols, qpos[i], qvel[i], qacc[i]):
            jid = model.joint(jname).id
            data.qpos[model.jnt_qposadr[jid]] = val
            data.qvel[model.jnt_dofadr[jid]] = vel
            data.qacc[model.jnt_dofadr[jid]] = acc
        mujoco.mj_inverse(model, data)
        if target_dof_indices:
             raw_torques.append(np.max(np.abs(data.qfrc_inverse[target_dof_indices])))
             
    peak_raw_torque = np.max(raw_torques)
    calibration_scale = TARGET_PEAK_TORQUE / peak_raw_torque
    print(f"      Peak Elbow Torque: {peak_raw_torque:.1f} -> {TARGET_PEAK_TORQUE:.1f} (Scale: {calibration_scale:.4f})")
    
    # 4. PREPARE SCALES VECTOR
    # Initialize all to 0.0 (Hide joints we don't care about, e.g. fingers)
    scales = np.zeros(model.nv)
    
    # Set Shoulder Scale (Massive Reduction)
    if broken_dof_indices:
        scales[broken_dof_indices] = SHOULDER_SCALE
        
    # Set Elbow Scale (Calibrated)
    if target_dof_indices:
        scales[target_dof_indices] = calibration_scale

    # 5. DYNAMICS LOOP
    n_frames = len(time)
    act_history = np.zeros((n_frames, model.nu))
    tau_history = np.zeros((n_frames, model.nv))
    
    print("   -> Running Dynamics...")
    for i in tqdm(range(n_frames), leave=False):
        for jname, val, vel, acc in zip(valid_cols, qpos[i], qvel[i], qacc[i]):
            jid = model.joint(jname).id
            data.qpos[model.jnt_qposadr[jid]] = val
            data.qvel[model.jnt_dofadr[jid]] = vel
            data.qacc[model.jnt_dofadr[jid]] = acc
            
        mujoco.mj_inverse(model, data)
        tau_raw = data.qfrc_inverse.copy()
        
        # Apply Logic: Scale everything appropriately. NO ZEROING OUT.
        tau_scaled = tau_raw * scales
        tau_history[i] = tau_scaled
        
        # Solver only looks at target_dof_indices anyway
        act_history[i] = solve_muscle_activations_sliced(model, data, tau_scaled, target_dof_indices)

    # 6. Save
    act_history = apply_lowpass_filter(act_history, LOW_PASS_CUTOFF, fs)
    act_history = np.clip(act_history, 0, 1)
    
    avg_act = np.mean(act_history)
    print(f"   -> Result: Avg Act = {avg_act:.2f}")

    muscle_names = [model.actuator(i).name for i in range(model.nu)]
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    
    pd.DataFrame(act_history, columns=muscle_names).assign(time=time).to_csv(os.path.join(session_dir, 'activations.csv'), index=False)
    pd.DataFrame(tau_history, columns=joint_names).assign(time=time).to_csv(os.path.join(session_dir, 'torques.csv'), index=False)
    
    if GENERATE_VIDEO: render_video(model, df, act_history, session_dir)

if __name__ == "__main__":
    if os.path.isdir(INPUT_PATH):
        files = glob.glob(os.path.join(INPUT_PATH, "*.mot"))
    else:
        files = [INPUT_PATH]
    
    for f in tqdm(files, desc="Total"):
        try: process_file(f, MODEL_XML_PATH, OUTPUT_ROOT)
        except Exception as e: print(f"Error {f}: {e}")