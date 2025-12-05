import os
import mujoco
import numpy as np
import pandas as pd
import scipy.sparse as spa
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import osqp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import skvideo.io
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy

# =============================================================================
# USER CONFIGURATION
# =============================================================================

MOT_FILE_PATH = r"/home/abdul/Desktop/myosuite/custom_workspace/IK/output/S5_12_1.mot"
MODEL_XML_PATH = r"/home/abdul/Desktop/myosuite/custom_workspace/model/myo_sim/arm/myoarm.xml"
OUTPUT_DIRECTORY = r"/home/abdul/Desktop/myosuite/custom_workspace/IK/output/ID_results"

# Toggles
GENERATE_PLOTS = True   # Set to True to save graphs
GENERATE_VIDEO = True   # Set to True to render video

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def read_mot_file(filepath):
    """Read a .mot file and return a pandas DataFrame."""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return None
    
    skiprows = 0
    with open(filepath, "r") as file:
        for line in file:
            if "endheader" in line:
                break
            skiprows += 1
    
    return pd.read_csv(filepath, sep=r'\s+', skiprows=skiprows + 1)

def compute_derivatives(data, dt):
    """Compute velocity and acceleration using central difference."""
    vel = np.zeros_like(data)
    vel[0] = (data[1] - data[0]) / dt
    vel[-1] = (data[-1] - data[-2]) / dt
    vel[1:-1] = (data[2:] - data[:-2]) / (2 * dt)
    
    acc = np.zeros_like(data)
    acc[0] = (vel[1] - vel[0]) / dt
    acc[-1] = (vel[-1] - vel[0]) / dt
    acc[1:-1] = (vel[2:] - vel[:-2]) / (2 * dt)
    
    return vel, acc

def apply_lowpass_filter(data, cutoff, fs, order=2):
    """Apply a low-pass Butterworth filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    if data.shape[0] < 15:
        return data

    if data.ndim == 1:
        return filtfilt(b, a, data)
    
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = filtfilt(b, a, data[:, i])
    return filtered_data

# =============================================================================
# OPTIMIZATION SOLVERS
# =============================================================================

def solve_muscle_activations_tracking_qp(model, data, tau_residual):
    """Fallback Solver: Minimizes error if constraints cannot be perfectly met."""
    n_actuators = model.nu
    moment_arm = data.actuator_moment.reshape(model.nv, n_actuators).copy()
    regularization = 0.01
    
    A = moment_arm
    P = 2 * (A.T @ A + regularization * spa.eye(n_actuators))
    q = -2 * A.T @ tau_residual
    
    # Constraints: 0 <= x <= 1
    A_bounds = spa.eye(n_actuators)
    lb = np.zeros(n_actuators)
    ub = np.ones(n_actuators)
    
    try:
        m = osqp.OSQP()
        m.setup(P=spa.csc_matrix(P), q=q, A=spa.csc_matrix(A_bounds), l=lb, u=ub, 
                verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000)
        res = m.solve()
        return res.x if res.info.status == 'solved' else solve_muscle_activations_simple(model, data, tau_residual)
    except Exception:
        return solve_muscle_activations_simple(model, data, tau_residual)

def solve_muscle_activations_qp(model, data, tau_residual):
    """Primary Solver: Static Optimization QP."""
    n_actuators = model.nu
    n_dof = model.nv

    # Scale qvel to stabilize moment arms calculation
    qvel_scaler = 5.0 
    data_stable = deepcopy(data)
    data_stable.qvel /= qvel_scaler
    mujoco.mj_forward(model, data_stable)
    
    moment_arm = data_stable.actuator_moment.reshape(n_dof, n_actuators).copy()

    if np.all(moment_arm == 0):
        return solve_muscle_activations_simple(model, data, tau_residual)

    # Objective: min sum(act^2)
    P = 2 * spa.eye(n_actuators)
    q = np.zeros(n_actuators)
    
    # Constraints: moment_arm @ act = tau; 0 <= act <= 1
    A_matrix = spa.vstack([moment_arm, spa.eye(n_actuators)])
    l_vec = np.hstack([tau_residual, np.zeros(n_actuators)])
    u_vec = np.hstack([tau_residual, np.ones(n_actuators)])

    try:
        m = osqp.OSQP()
        m.setup(P=spa.csc_matrix(P), q=q, A=spa.csc_matrix(A_matrix), l=l_vec, u=u_vec, 
                verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000)
        res = m.solve()
        
        if res.info.status == 'solved':
            return res.x
        else:
            return solve_muscle_activations_tracking_qp(model, data_stable, tau_residual)
    except Exception:
        return solve_muscle_activations_tracking_qp(model, data_stable, tau_residual)

def solve_muscle_activations_simple(model, data, tau_residual):
    """Simple Fallback: Least Squares."""
    n_actuators = model.nu
    moment_arm = data.actuator_moment.reshape(model.nv, n_actuators).copy()
    
    if np.all(moment_arm == 0):
        moment_arm = np.eye(min(model.nv, n_actuators), n_actuators)
    
    regularization = 0.01
    A = moment_arm
    ATA = A.T @ A + regularization * np.eye(n_actuators)
    ATb = A.T @ tau_residual
    
    try:
        activations = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        activations = np.linalg.lstsq(A, tau_residual, rcond=None)[0]
    
    return np.clip(activations, 0.0, 1.0)

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def run_inverse_dynamics():
    print("="*60)
    print("INVERSE DYNAMICS PIPELINE")
    print("="*60)
    
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # 1. Load Data
    print("1. Loading Data...")
    motion_df = read_mot_file(MOT_FILE_PATH)
    if motion_df is None: return

    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    
    # 2. Map Joints
    model_joint_names = [model.joint(j).name for j in range(model.njnt)]
    motion_joint_names = [col for col in motion_df.columns if col != 'time' and col in model_joint_names]
    joint_indices = [model_joint_names.index(name) for name in motion_joint_names]
    
    time = motion_df['time'].values
    dt = np.mean(np.diff(time))
    qpos_traj = motion_df[motion_joint_names].values
    qvel_traj, qacc_traj = compute_derivatives(qpos_traj, dt)
    
    print(f"   ✓ {len(motion_df)} frames, {len(motion_joint_names)} mapped joints, {1/dt:.1f} Hz")

    # 3. Compute Dynamics
    n_frames = len(time)
    tau_gravity = np.zeros((n_frames, model.nv))
    tau_total = np.zeros((n_frames, model.nv))
    muscle_activations = np.zeros((n_frames, model.nu))
    
    print("\n2. Computing Inverse Dynamics...")
    for i in tqdm(range(n_frames)):
        data.qpos[joint_indices] = qpos_traj[i]
        data.qvel[joint_indices] = qvel_traj[i]
        data.qacc[joint_indices] = qacc_traj[i]
        
        mujoco.mj_forward(model, data)
        mujoco.mj_inverse(model, data)
        tau_total[i] = data.qfrc_inverse.copy()
        
        # Gravity compensation check
        data_temp = mujoco.MjData(model)
        data_temp.qpos[:] = data.qpos
        mujoco.mj_forward(model, data_temp)
        tau_gravity[i] = data_temp.qfrc_bias.copy()
        
        muscle_activations[i] = solve_muscle_activations_qp(model, data, tau_total[i])
    
    # 4. Filter and Save
    print("\n3. Saving Results...")
    fs = 1.0 / dt
    muscle_activations = apply_lowpass_filter(muscle_activations, 6, fs)
    muscle_activations = np.clip(muscle_activations, 0.0, 1.0)

    joint_names_full = [model.joint(j).name for j in range(model.nv)]
    pd.DataFrame(tau_total, columns=joint_names_full).assign(time=time).to_csv(os.path.join(OUTPUT_DIRECTORY, 'joint_torques.csv'), index=False)
    pd.DataFrame(tau_gravity, columns=joint_names_full).assign(time=time).to_csv(os.path.join(OUTPUT_DIRECTORY, 'gravity_torques.csv'), index=False)
    
    muscle_names = [model.actuator(i).name for i in range(model.nu)]
    pd.DataFrame(muscle_activations, columns=muscle_names).assign(time=time).to_csv(os.path.join(OUTPUT_DIRECTORY, 'muscle_activations.csv'), index=False)
    
    print(f"   ✓ Results saved to {OUTPUT_DIRECTORY}")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_results():
    print("\nGenerating Plots...")
    try:
        activations = pd.read_csv(os.path.join(OUTPUT_DIRECTORY, 'muscle_activations.csv'))
        torques = pd.read_csv(os.path.join(OUTPUT_DIRECTORY, 'joint_torques.csv'))
        time = activations['time'].values
        
        # Plot 1: Muscle Activations (Grid)
        muscle_names = [c for c in activations.columns if c != 'time']
        n_muscles = len(muscle_names)
        n_rows = int(np.ceil(n_muscles / 8))
        fig, axes = plt.subplots(n_rows, min(8, n_muscles), figsize=(16, 2*n_rows))
        axes = axes.flatten() if n_muscles > 1 else [axes]
        
        for i, muscle in enumerate(muscle_names):
            axes[i].plot(time, activations[muscle], linewidth=1)
            axes[i].set_title(muscle, fontsize=8)
            axes[i].set_ylim([0, 1])
            axes[i].grid(alpha=0.3)
            if i % 8 == 0: axes[i].set_ylabel('Act')
        for i in range(n_muscles, len(axes)): axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'muscle_activations.png'), dpi=150)
        plt.close()

        # Plot 2: Key Joint Torques
        key_joints = [c for c in torques.columns if c != 'time'][:6] # Plot first 6 joints
        fig, axes = plt.subplots(int(np.ceil(len(key_joints)/3)), 3, figsize=(15, 6))
        axes = axes.flatten()
        for i, joint in enumerate(key_joints):
            axes[i].plot(time, torques[joint])
            axes[i].set_title(joint)
            axes[i].grid(alpha=0.3)
        for i in range(len(key_joints), len(axes)): axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'joint_torques.png'), dpi=150)
        plt.close()
        print("   ✓ Plots saved.")
        
    except Exception as e:
        print(f"   x Plotting failed: {e}")

def create_video():
    print("\nRendering Video...")
    try:
        motion_df = read_mot_file(MOT_FILE_PATH)
        activations = pd.read_csv(os.path.join(OUTPUT_DIRECTORY, 'muscle_activations.csv'))
        model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
        data = mujoco.MjData(model)
        
        # Setup Video
        output_video = os.path.join(OUTPUT_DIRECTORY, 'muscle_driven_motion.mp4')
        fps = 1.0 / np.mean(np.diff(motion_df['time'].values))
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.azimuth, camera.elevation, camera.distance = 45, -20, 1.5
        camera.lookat = [0, 0, 0.3]

        # Map actuators to geoms for coloring
        muscle_names = [c for c in activations.columns if c != 'time']
        muscle_map = {}
        cylinders = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER]
        
        for act in muscle_names:
            muscle_map[act] = [g for g in cylinders if model.geom(g).name and act in model.geom(g).name]

        frames = []
        blue, red = np.array([0, 0, 1, 1]), np.array([1, 0, 0, 1])
        joint_cols = [c for c in motion_df.columns if c in [model.joint(j).name for j in range(model.njnt)]]

        for i in tqdm(range(len(motion_df))):
            # Update Joints
            for jname in joint_cols:
                data.qpos[model.jnt_qposadr[model.joint(jname).id]] = motion_df[jname].iloc[i]
            
            mujoco.mj_forward(model, data)

            # Update Colors based on activation
            current_acts = activations.iloc[i]
            for act, geoms in muscle_map.items():
                val = current_acts[act]
                color = (1.0 - val) * blue + val * red
                for g in geoms: model.geom_rgba[g] = color

            # Render
            renderer.update_scene(data, camera=camera)
            img = Image.fromarray(renderer.render())
            
            # Overlay Text
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Time: {motion_df['time'].iloc[i]:.2f}s", fill=(255, 255, 255))
            
            frames.append(np.array(img))

        skvideo.io.vwrite(output_video, np.array(frames), inputdict={'-r': str(fps)}, outputdict={'-r': str(fps), '-vcodec': 'libx264', '-crf': '18'})
        print(f"   ✓ Video saved to {output_video}")

    except Exception as e:
        print(f"   x Video generation failed: {e}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_inverse_dynamics()
    
    if GENERATE_PLOTS:
        plot_results()
        
    if GENERATE_VIDEO:
        create_video()
        
    print("\n" + "="*60)
    print("DONE.")
    print("="*60)