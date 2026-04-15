"""
generate.py — Generate marker-space CSVs from a trained test2 experiment.

Usage:
  # Generate for a range of FMA scores
  python test/generate.py --experiment A1_no_residual --fma 16-66

  # Generate for specific FMA scores
  python test/generate.py --experiment C3_both --fma 18,30,40,50,66

  # Use best checkpoint (default) or final
  python test/generate.py --experiment A1_no_residual --fma 40 --checkpoint final

Outputs to: test/output/{experiment}/csv/FMA_{score}.csv
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import joblib
from scipy import signal

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TEST_DIR)
sys.path.insert(0, ROOT_DIR)

from models import MotionCVAE, SEQ_LEN

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
            'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x','Trunk_y','Trunk_z']
COLS       = ARM_COLS + TRUNK_COLS

# Average starting pose in absolute mm (from real MHH recordings).
# Training data is in delta space (frame 0 subtracted).
# After inverse_transform we add this back to get absolute positions for IK.
REFERENCE_POSE = {
    'Sh_x': -77.3,  'Sh_y': 643.0, 'Sh_z': 302.7,
    'El_x': -188.2, 'El_y': 474.4, 'El_z':  41.3,
    'Wr_x':  -88.7, 'Wr_y': 241.2, 'Wr_z':  41.1,
    'WrVec_x': -37.0, 'WrVec_y': 12.0, 'WrVec_z': -33.1,
    'Trunk_x': 0.0, 'Trunk_y': 0.0, 'Trunk_z': 0.0,
}
REF_ARRAY = np.array([REFERENCE_POSE[c] for c in COLS])


def smooth(data, cutoff=6, fs=100):
    out = data.copy()
    nyq = 0.5 * fs
    b, a = signal.butter(2, min(cutoff / nyq, 0.99), btype='low')
    for i in range(data.shape[1]):
        out[:, i] = signal.filtfilt(b, a, data[:, i])
    return out


def generate_for_experiment(exp_name: str, fma_scores: list,
                             checkpoint: str = "best",
                             guidance: float = 2.0,
                             apply_smooth: bool = True,
                             seed: int = None,
                             n_samples: int = 1,
                             out_subdir: str = "csv"):

    out_dir = os.path.join(TEST_DIR, "output", exp_name)
    cfg_path = os.path.join(out_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"No config.json in {out_dir}. Run train.py first.")

    with open(cfg_path) as f:
        config = json.load(f)

    ckpt_name = "model_best.pth" if checkpoint == "best" else "model_final.pth"
    ckpt_path = os.path.join(out_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    scaler_path = os.path.join(out_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    model = MotionCVAE(config["model"]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    use_cfg = config["model"].get("use_cfg", False)
    gs = guidance if use_cfg else 1.0

    csv_dir = os.path.join(out_dir, out_subdir)
    os.makedirs(csv_dir, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"Generating {len(fma_scores)} FMA scores for [{exp_name}] "
          f"(guidance={gs:.1f}, ckpt={checkpoint}, seed={seed}, n_samples={n_samples})...")

    for fma in fma_scores:
        c = torch.FloatTensor([[fma / 66.0]]).to(DEVICE)
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = model.inference(c, seq_len=SEQ_LEN, guidance_scale=gs)
                raw = out.squeeze(0).cpu().numpy()
                samples.append(scaler.inverse_transform(raw))
        # Average across samples (reduces stochastic noise)
        delta_mm = np.mean(samples, axis=0)
        abs_mm   = delta_mm + REF_ARRAY
        if apply_smooth:
            abs_mm = smooth(abs_mm)
        df = pd.DataFrame(abs_mm, columns=COLS)
        path = os.path.join(csv_dir, f"FMA_{fma}.csv")
        df.to_csv(path, index=False)

    print(f"  Saved to {csv_dir}/")
    return csv_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", required=True)
    parser.add_argument("--fma", default="16-66",
                        help="FMA range '16-66' or list '18,30,40,50,66'")
    parser.add_argument("--checkpoint", default="best", choices=["best", "final"])
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible generation")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of samples to average per FMA score (reduces noise)")
    args = parser.parse_args()

    if "-" in args.fma:
        lo, hi = map(int, args.fma.split("-"))
        scores = list(range(lo, hi + 1))
    else:
        scores = [int(x) for x in args.fma.split(",")]

    generate_for_experiment(args.experiment, scores,
                            args.checkpoint, args.guidance,
                            not args.no_smooth, args.seed, args.n_samples)
