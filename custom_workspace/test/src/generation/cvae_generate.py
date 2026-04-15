import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
import argparse
from scipy import signal

# Add project root to path
# __file__ is test/src/generation/cvae_generate.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.utils.config import get_path
from src.generation.minimalist_models import STAGE_MODELS

# Constants
SEQ_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
COLS = ARM_COLS + TRUNK_COLS


def load_reference_pose():
    """Average starting pose in mm (absolute body frame) to add back to delta outputs.

    Training data stores positions as deltas from the resting start pose.
    After inverse_transform we're back in delta-mm space; adding this reference
    converts to absolute mm positions that the IK solver expects.
    """
    return {
        'Sh_x': -77.3, 'Sh_y': 643.0, 'Sh_z': 302.7,
        'El_x': -188.2, 'El_y': 474.4, 'El_z': 41.3,
        'Wr_x': -88.7, 'Wr_y': 241.2, 'Wr_z': 41.1,
        'WrVec_x': -37.0, 'WrVec_y': 12.0, 'WrVec_z': -33.1,
        'Trunk_x': 0.0, 'Trunk_y': 0.0, 'Trunk_z': 0.0,
    }


def smooth_trajectory(data, cutoff=6, fs=100):
    """Low-pass Butterworth filter for smoother motion."""
    smoothed = data.copy()
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = signal.butter(2, normal_cutoff, btype='low')
    for i in range(data.shape[1]):
        smoothed[:, i] = signal.filtfilt(b, a, data[:, i])
    return smoothed


def generate(fma, stage=0, data_source="dtw", guidance=2.0, n_samples=1, output_dir=None, smooth=True):
    model_dir = os.path.join(BASE_DIR, "models/cvae")
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "output/generated/csv")
    os.makedirs(output_dir, exist_ok=True)

    # Load scaler
    scaler_path = os.path.join(model_dir, "scaler_cutoff_fma.pkl")
    scaler = joblib.load(scaler_path)

    # Reference pose: converts model delta-output → absolute mm positions
    ref_pose = load_reference_pose()

    # Load model
    ModelClass = STAGE_MODELS[stage]
    model = ModelClass().to(DEVICE)
    model_name = f"cvae_stage{stage}_{data_source}"
    ckpt_path = os.path.join(model_dir, f"{model_name}_best.pth")

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # Condition
    c = torch.FloatTensor([[fma / 66.0]]).to(DEVICE)

    with torch.no_grad():
        for i in range(n_samples):
            try:
                recon = model.inference(c, seq_len=SEQ_LEN, guidance_scale=guidance)
            except TypeError:
                recon = model.inference(c, seq_len=SEQ_LEN)

            recon_np = recon.squeeze(0).cpu().numpy()
            # inverse_transform → delta-mm space
            data_delta = scaler.inverse_transform(recon_np)

            # Add reference pose: delta-mm → absolute-mm (required by IK solver)
            data_abs = data_delta.copy()
            for col_idx, col in enumerate(COLS):
                data_abs[:, col_idx] += ref_pose.get(col, 0.0)

            # Optional smoothing
            if smooth:
                data_abs = smooth_trajectory(data_abs)

            # Save to CSV
            df = pd.DataFrame(data_abs, columns=COLS)
            out_path = os.path.join(output_dir, f"FMA_{fma}.csv" if n_samples == 1 else f"FMA_{fma}_s{i}.csv")
            df.to_csv(out_path, index=False)
            print(f"Generated: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fma", type=int, default=30)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--data-source", type=str, default="dtw")
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    
    generate(args.fma, args.stage, args.data_source, args.guidance, args.n_samples, args.output_dir)
