"""
Inverse Dynamics: MOT -> joint torques + muscle activations.

Supports single-file mode (via module-level vars or CLI) and batch mode
(directory of .mot files). Uses a sliced least-squares solver targeting
specific joints, with auto-calibration of torque scales.
"""
import os
import sys
import glob
import mujoco
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
import skvideo.io
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Single-file defaults (can be monkey-patched by run_generated_pipeline.py)
MOT_FILE_PATH = get_path("output_dir") + "/S5_12_1.mot"
MODEL_XML_PATH = get_path("mujoco_arm_model")
OUTPUT_DIRECTORY = get_path("output_originals_id")

# Batch defaults
INPUT_PATH = get_path("output_originals_mot")
OUTPUT_ROOT = get_path("output_originals_id")

# Solver: which joints to solve muscle activations for
SOLVE_JOINTS = ['elbow_flexion', 'pro_sup', 'deviation', 'flexion']

# Shoulder torques are unrealistically high in the model; scale them down
SHOULDER_SCALE = 0.00001
TARGET_PEAK_TORQUE = 12.0
BROKEN_JOINTS = [
    'elv_angle', 'shoulder_elv', 'shoulder1_r2', 'shoulder_rot',
    'sternoclavicular_r2', 'sternoclavicular_r3',
    'unrotscap_r3', 'unrotscap_r2',
    'acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3',
    'unrothum_r1', 'unrothum_r2', 'unrothum_r3',
]

# Toggles
GENERATE_PLOTS = True
GENERATE_VIDEO = True
LOW_PASS_CUTOFF = 1.5

# =============================================================================
# UTILITIES
# =============================================================================

def read_mot_file(filepath):
    """Read a .mot file and return a pandas DataFrame."""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return None
    skiprows = 0
    with open(filepath, "r") as f:
        for line in f:
            if "endheader" in line:
                break
            skiprows += 1
    return pd.read_csv(filepath, sep=r'\s+', skiprows=skiprows + 1)


def compute_derivatives(data, dt):
    """Compute velocity and acceleration using np.gradient."""
    vel = np.gradient(data, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def apply_lowpass_filter(data, cutoff, fs, order=4):
    """Apply a low-pass Butterworth filter column-wise."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if data.shape[0] < 15:
        return data
    if data.ndim == 1:
        return filtfilt(b, a, data)
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])
    return filtered

# =============================================================================
# SOLVER
# =============================================================================

def solve_muscle_activations_sliced(model, data, tau_full, target_indices):
    """
    Solve for muscle activations using only the target joint DOFs.

    Slices the moment-arm matrix to the target joints (e.g. elbow/wrist),
    preventing shoulder torques from interfering with the solution.
    """
    n_muscles = model.nu
    n_targets = len(target_indices)

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    moment_arm_full = data.actuator_moment.reshape(model.nv, n_muscles)

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
# CORE PROCESSING
# =============================================================================

def process_file(mot_path, model_path, output_dir):
    """
    Run full inverse dynamics on a single MOT file.

    Saves activations.csv and torques.csv to output_dir.
    Optionally renders a video (controlled by GENERATE_VIDEO).
    """
    file_id = os.path.basename(mot_path).replace('.mot', '')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing: {file_id}")

    df = read_mot_file(mot_path)
    if df is None:
        return
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    model_joints = [model.joint(j).name for j in range(model.njnt)]
    valid_cols = [c for c in df.columns if c in model_joints]

    # 1. Map DOF indices
    target_dof_indices = []
    for jname in SOLVE_JOINTS:
        if jname in model_joints:
            jid = model.joint(jname).id
            target_dof_indices.append(model.jnt_dofadr[jid])

    broken_dof_indices = []
    for jname in BROKEN_JOINTS:
        if jname in model_joints:
            jid = model.joint(jname).id
            jtype = model.jnt_type[jid]
            dim = {0: 6, 1: 3}.get(int(jtype), 1)
            dof_start = model.jnt_dofadr[jid]
            for d in range(dof_start, dof_start + dim):
                broken_dof_indices.append(d)

    # 2. Kinematics
    time = df['time'].values
    dt = np.mean(np.diff(time))
    if dt == 0:
        dt = 0.005
    fs = 1.0 / dt

    qpos = df[valid_cols].values
    qvel, qacc = compute_derivatives(qpos, dt)
    qvel = apply_lowpass_filter(qvel, LOW_PASS_CUTOFF, fs)
    qacc = apply_lowpass_filter(qacc, LOW_PASS_CUTOFF, fs)

    # 3. Auto-calibration: scale torques so peak elbow torque hits TARGET_PEAK_TORQUE
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

    peak_raw_torque = np.max(raw_torques) if raw_torques else 1.0
    calibration_scale = TARGET_PEAK_TORQUE / peak_raw_torque
    print(f"      Peak Elbow Torque: {peak_raw_torque:.1f} -> {TARGET_PEAK_TORQUE:.1f} (Scale: {calibration_scale:.4f})")

    # 4. Build per-DOF scale vector
    scales = np.zeros(model.nv)
    if broken_dof_indices:
        scales[broken_dof_indices] = SHOULDER_SCALE
    if target_dof_indices:
        scales[target_dof_indices] = calibration_scale

    # 5. Dynamics loop
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

        tau_scaled = tau_raw * scales
        tau_history[i] = tau_scaled
        act_history[i] = solve_muscle_activations_sliced(model, data, tau_scaled, target_dof_indices)

    # 6. Filter and save
    act_history = apply_lowpass_filter(act_history, LOW_PASS_CUTOFF, fs)
    act_history = np.clip(act_history, 0, 1)

    avg_act = np.mean(act_history)
    print(f"   -> Result: Avg Act = {avg_act:.2f}")

    muscle_names = [model.actuator(i).name for i in range(model.nu)]
    joint_names = [model.joint(i).name for i in range(model.njnt)]

    pd.DataFrame(act_history, columns=muscle_names).assign(time=time).to_csv(
        os.path.join(output_dir, 'activations.csv'), index=False)
    pd.DataFrame(tau_history, columns=joint_names).assign(time=time).to_csv(
        os.path.join(output_dir, 'torques.csv'), index=False)

    print(f"   -> Saved to {output_dir}")

    if GENERATE_VIDEO:
        render_video(model, df, act_history, output_dir)

# =============================================================================
# BACKWARD-COMPAT WRAPPERS (used by run_generated_pipeline.py)
# =============================================================================

def run_inverse_dynamics():
    """Process the single file specified by module-level MOT_FILE_PATH."""
    process_file(MOT_FILE_PATH, MODEL_XML_PATH, OUTPUT_DIRECTORY)

# =============================================================================
# VISUALIZATION
# =============================================================================

def render_video(model, motion_df, activations, out_dir):
    """Render muscle-activity video with blue->purple->red coloring."""
    print("   -> Rendering Video...")
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth, camera.elevation, camera.distance = 45, -30, 1.8
    camera.lookat = [0.0, 0.0, 0.9]
    data = mujoco.MjData(model)

    geom_map, site_map, tendon_map = {}, {}, {}
    for i in range(model.nu):
        act_name = model.actuator(i).name
        geom_map[i] = [g for g in range(model.ngeom) if act_name in (model.geom(g).name or "")]
        site_map[i] = [s for s in range(model.nsite) if act_name in (model.site(s).name or "")]
        tendon_map[i] = [t for t in range(model.ntendon) if act_name in (model.tendon(t).name or "")]

    frames = []
    step = 3

    for i in tqdm(range(0, len(motion_df), step), leave=False):
        for col in motion_df.columns:
            if col == 'time':
                continue
            try:
                jid = model.joint(col).id
                data.qpos[model.jnt_qposadr[jid]] = motion_df[col].iloc[i]
            except Exception:
                pass
        mujoco.mj_forward(model, data)

        for mid, act in enumerate(activations[i]):
            if act < 0.5:
                color = (1 - 2*act)*np.array([0.2, 0.2, 0.8, 1]) + (2*act)*np.array([0.6, 0.2, 0.6, 1])
            else:
                color = (1 - 2*(act-0.5))*np.array([0.6, 0.2, 0.6, 1]) + (2*(act-0.5))*np.array([0.9, 0.1, 0.1, 1])
            for gid in geom_map.get(mid, []):
                model.geom_rgba[gid] = color
            for sid in site_map.get(mid, []):
                model.site_rgba[sid] = color
            for tid in tendon_map.get(mid, []):
                model.tendon_rgba[tid] = color

        renderer.update_scene(data, camera=camera)
        img = Image.fromarray(renderer.render())
        ImageDraw.Draw(img).text((10, 10), f"Time: {motion_df['time'].iloc[i]:.2f}s", fill='white')
        frames.append(np.array(img))

    skvideo.io.vwrite(
        os.path.join(out_dir, 'muscle_activity.mp4'), np.array(frames),
        inputdict={'-r': '60'}, outputdict={'-r': '60', '-pix_fmt': 'yuv420p'})
    print(f"   -> Video saved.")


def plot_results(output_dir=None):
    """Generate activation and torque plots from saved CSVs."""
    if output_dir is None:
        output_dir = OUTPUT_DIRECTORY

    print("\nGenerating Plots...")
    try:
        activations = pd.read_csv(os.path.join(output_dir, 'activations.csv'))
        torques = pd.read_csv(os.path.join(output_dir, 'torques.csv'))
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
            if i % 8 == 0:
                axes[i].set_ylabel('Act')
        for i in range(n_muscles, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'activations.png'), dpi=150)
        plt.close()

        # Plot 2: Key Joint Torques
        key_joints = [c for c in torques.columns if c != 'time'][:6]
        fig, axes = plt.subplots(int(np.ceil(len(key_joints)/3)), 3, figsize=(15, 6))
        axes = axes.flatten()
        for i, joint in enumerate(key_joints):
            axes[i].plot(time, torques[joint])
            axes[i].set_title(joint)
            axes[i].grid(alpha=0.3)
        for i in range(len(key_joints), len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'torques.png'), dpi=150)
        plt.close()
        print("   -> Plots saved.")

    except Exception as e:
        print(f"   x Plotting failed: {e}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    if os.path.isdir(INPUT_PATH):
        files = sorted(glob.glob(os.path.join(INPUT_PATH, "*.mot")))
    else:
        files = [INPUT_PATH]

    print(f"Found {len(files)} MOT file(s)")

    for f in tqdm(files, desc="Total"):
        try:
            file_id = os.path.basename(f).replace('.mot', '')
            session_dir = os.path.join(OUTPUT_ROOT, file_id)
            process_file(f, MODEL_XML_PATH, session_dir)
            if GENERATE_PLOTS:
                plot_results(session_dir)
        except Exception as e:
            print(f"Error {f}: {e}")

    print("\nDONE.")
