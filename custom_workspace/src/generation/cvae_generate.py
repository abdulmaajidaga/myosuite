import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
import argparse
import os
import sys
import joblib
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

# --- CONFIGURATION ---
MODEL_PATH = get_path("cvae_model_best")
SCALER_PATH = get_path("cvae_scaler")
REFERENCE_PATH = get_path("reference_healthy_csv")
OUTPUT_DIR = get_path("output_generated_csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.generation.model import MotionCVAE, INPUT_DIM, CONDITION_DIM, HIDDEN_DIM, LATENT_DIM, NUM_HEADS, SEQ_LEN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Column names
ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
COLS = ARM_COLS + TRUNK_COLS


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_reference_pose():
    """Load the reference starting pose to add back to delta output.

    The model outputs deltas (differences from start position).
    This uses the average starting position across all training subjects.
    """
    # Average starting pose across all subjects in training data
    return {
        'Sh_x': -77.3, 'Sh_y': 643.0, 'Sh_z': 302.7,
        'El_x': -188.2, 'El_y': 474.4, 'El_z': 41.3,
        'Wr_x': -88.7, 'Wr_y': 241.2, 'Wr_z': 41.1,
        'WrVec_x': -37.0, 'WrVec_y': 12.0, 'WrVec_z': -33.1,
        'Trunk_x': 0.0, 'Trunk_y': 0.0, 'Trunk_z': 0.0,
    }


def smooth_trajectory(data, cutoff=6, fs=100):
    """Apply low-pass filter for smoother motion."""
    smoothed = data.copy()
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = signal.butter(2, normal_cutoff, btype='low')

    for i in range(data.shape[1]):
        smoothed[:, i] = signal.filtfilt(b, a, data[:, i])

    return smoothed


# ==========================================
# GENERATION FUNCTION
# ==========================================

def generate_motion(target_score, smooth=True, guidance_scale=2.0):
    """Generate motion for a target FMA score with classifier-free guidance."""

    # Try best model first, then regular
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        model_path = model_path.replace('_best', '')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run training first: python3 cvae_train_cutoff.py")
        return None

    print(f"Loading model from {model_path}...")
    print(f"Guidance scale: {guidance_scale}")

    # Load Model
    model = MotionCVAE().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Load Scaler
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler not found at {SCALER_PATH}")
        return None
    scaler = joblib.load(SCALER_PATH)

    # Load reference pose (to add back to delta output)
    ref_pose = load_reference_pose()

    # Prepare Input
    norm_score = target_score / 66.0
    print(f"Generating motion for FMA {target_score} (Normalized: {norm_score:.2f})...")

    c = torch.FloatTensor([[norm_score]]).to(DEVICE)

    # Generate with classifier-free guidance (output is in normalized delta space)
    generated_tensor = model.inference(c, seq_len=SEQ_LEN, guidance_scale=guidance_scale)
    data_norm = generated_tensor.squeeze(0).cpu().numpy()

    # Un-normalize (back to delta mm space)
    data_delta = scaler.inverse_transform(data_norm)

    # Add reference pose (convert delta to absolute coordinates)
    data_abs = data_delta.copy()
    for col_idx, col in enumerate(COLS):
        data_abs[:, col_idx] += ref_pose.get(col, 0)

    # Optional smoothing
    if smooth:
        data_abs = smooth_trajectory(data_abs)

    # Save to CSV
    df = pd.DataFrame(data_abs, columns=COLS)
    out_path = os.path.join(OUTPUT_DIR, f"FMA_{target_score}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    return data_abs


# ==========================================
# VISUALIZATION FUNCTION
# ==========================================

def visualise(data, title_score):
    # Extract Joints (arm only for visualization)
    sh = data[:, 0:3]
    el = data[:, 3:6]
    wr = data[:, 6:9]
    trunk = data[:, 12:15] if data.shape[1] >= 15 else None

    # Setup Plot Limits
    all_x = np.concatenate([sh[:,0], el[:,0], wr[:,0]])
    all_y = np.concatenate([sh[:,1], el[:,1], wr[:,1]])
    all_z = np.concatenate([sh[:,2], el[:,2], wr[:,2]])
    pad = 50
    xlims = [all_x.min()-pad, all_x.max()+pad]
    ylims = [all_y.min()-pad, all_y.max()+pad]
    zlims = [all_z.min()-pad, all_z.max()+pad]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Generated Motion: FMA {title_score}", fontsize=14)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(xlims); ax.set_ylim(ylims); ax.set_zlim(zlims)

    try: ax.set_box_aspect((np.ptp(xlims), np.ptp(ylims), np.ptp(zlims)))
    except: pass

    # Graphics
    line1, = ax.plot([], [], [], 'b-', lw=4, label='Upper Arm')
    line2, = ax.plot([], [], [], 'orange', lw=4, label='Forearm')
    pt_sh, = ax.plot([], [], [], 'ko', markersize=8)
    pt_el, = ax.plot([], [], [], 'bo', markersize=6)
    pt_wr, = ax.plot([], [], [], 'ro', markersize=6)
    trail, = ax.plot([], [], [], 'r:', lw=1, alpha=0.5)

    # 2D trajectory plot
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Y (Forward)'); ax2.set_ylabel('Z (Up)')
    ax2.set_title('Wrist Trajectory + Trunk Displacement')
    ax2.plot(wr[:,1], wr[:,2], 'b-', lw=2, label='Wrist')
    if trunk is not None:
        ax2.plot(trunk[:,1] * 10, trunk[:,2] * 10, 'g--', lw=1, alpha=0.7, label='Trunk (x10)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    def update(frame):
        s, e, w = sh[frame], el[frame], wr[frame]

        line1.set_data_3d([s[0], e[0]], [s[1], e[1]], [s[2], e[2]])
        line2.set_data_3d([e[0], w[0]], [e[1], w[1]], [e[2], w[2]])
        pt_sh.set_data_3d([s[0]], [s[1]], [s[2]])
        pt_el.set_data_3d([e[0]], [e[1]], [e[2]])
        pt_wr.set_data_3d([w[0]], [w[1]], [w[2]])
        trail.set_data_3d(wr[:frame+1, 0], wr[:frame+1, 1], wr[:frame+1, 2])

        return line1, line2, pt_sh, pt_el, pt_wr, trail

    ani = FuncAnimation(fig, update, frames=len(data), interval=50, blit=False)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# ANALYSIS FUNCTION
# ==========================================

def analyze_motion(data, fma_score):
    """Quick analysis of generated motion including trunk."""
    print(f"\n=== Motion Analysis (FMA {fma_score}) ===")

    sh = data[:, 0:3]
    el = data[:, 3:6]
    wr = data[:, 6:9]

    # Arm segment lengths
    upper_arm = np.linalg.norm(el - sh, axis=1)
    forearm = np.linalg.norm(wr - el, axis=1)
    print(f"Upper arm: {upper_arm.mean():.1f} +/- {upper_arm.std():.1f} mm")
    print(f"Forearm:   {forearm.mean():.1f} +/- {forearm.std():.1f} mm")

    # Motion range (arm)
    print(f"\nArm Motion range (Y-axis):")
    print(f"  Shoulder: {sh[:,1].max() - sh[:,1].min():.1f} mm")
    print(f"  Elbow:    {el[:,1].max() - el[:,1].min():.1f} mm")
    print(f"  Wrist:    {wr[:,1].max() - wr[:,1].min():.1f} mm")

    # Trunk compensation (if available)
    if data.shape[1] >= 15:
        trunk = data[:, 12:15]
        trunk_disp = np.linalg.norm(trunk - trunk[0], axis=1).max()
        wrist_disp = np.linalg.norm(wr - wr[0], axis=1).max()
        trunk_ratio = trunk_disp / wrist_disp if wrist_disp > 0 else 0

        print(f"\nTrunk Compensation:")
        print(f"  Max trunk displacement: {trunk_disp:.1f} mm")
        print(f"  Max wrist displacement: {wrist_disp:.1f} mm")
        print(f"  Trunk/Wrist ratio:      {trunk_ratio:.3f}")


# ==========================================
# MAIN INTERFACE
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Motion for a specific FMA Score")
    parser.add_argument("--fma", type=int, default=50, help="Target FMA Score (0-66)")
    parser.add_argument("--guidance", type=float, default=2.0, help="Guidance scale for conditioning strength (default: 2.0)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    target_fma_score = args.fma

    # Generate
    motion_data = generate_motion(target_fma_score, guidance_scale=args.guidance)

    if motion_data is not None:
        analyze_motion(motion_data, target_fma_score)
        if not args.no_viz:
            visualise(motion_data, target_fma_score)
