"""
cvae_generate.py — Generate FMA-conditioned motion CSVs from a trained MotionCVAE.

Loads model config from models/cvae/config.json (written by cvae_train.py).
Defaults to n=10 sample averaging for stable output (same as evaluation protocol).

Usage:
  python src/generation/cvae_generate.py --fma 50           # single FMA score
  python src/generation/cvae_generate.py --fma 16-66        # full range
  python src/generation/cvae_generate.py --fma 18,30,50,66  # specific scores
  python src/generation/cvae_generate.py --fma 50 --n-samples 1  # fast single sample
  python src/generation/cvae_generate.py --fma 50 --no-viz       # skip animation
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.generation.model import MotionCVAE, SEQ_LEN
from src.utils.config import get_path

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = None   # resolved at runtime from config
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARM_COLS   = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
              'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x','Trunk_y','Trunk_z']
COLS       = ARM_COLS + TRUNK_COLS

# Average starting pose in absolute mm (from real MHH recordings).
# Training data is in delta space (frame-0 subtracted).
# After inverse_transform we add this back to get absolute positions for IK.
REFERENCE_POSE = {
    'Sh_x': -77.3,  'Sh_y': 643.0, 'Sh_z': 302.7,
    'El_x': -188.2, 'El_y': 474.4, 'El_z':  41.3,
    'Wr_x':  -88.7, 'Wr_y': 241.2, 'Wr_z':  41.1,
    'WrVec_x': -37.0, 'WrVec_y': 12.0, 'WrVec_z': -33.1,
    'Trunk_x': 0.0, 'Trunk_y': 0.0, 'Trunk_z': 0.0,
}
REF_ARRAY = np.array([REFERENCE_POSE[c] for c in COLS])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_models_dir():
    import src.utils.config as _cfg
    return os.path.join(_cfg.get_project_root(), "models", "cvae")


def _load_model(models_dir, checkpoint="best"):
    config_path = os.path.join(models_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config.json found at {config_path}. Run cvae_train.py first.")

    with open(config_path) as f:
        saved = json.load(f)
    model_cfg = saved.get("model", {})

    ckpt_name = "cvae_cutoff_fma_best.pth" if checkpoint == "best" else "cvae_cutoff_fma.pth"
    ckpt_path = os.path.join(models_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    scaler_path = os.path.join(models_dir, "scaler_cutoff_fma.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = MotionCVAE(model_cfg).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    scaler = joblib.load(scaler_path)

    print(f"Model:  {model.describe()}")
    print(f"Ckpt:   {ckpt_path}")
    return model, scaler, model_cfg


def smooth(data, cutoff=6, fs=100):
    out = data.copy()
    nyq = 0.5 * fs
    b, a = signal.butter(2, min(cutoff / nyq, 0.99), btype='low')
    for i in range(data.shape[1]):
        out[:, i] = signal.filtfilt(b, a, data[:, i])
    return out


# ── Core generation ───────────────────────────────────────────────────────────

def generate_motion(fma_score, n_samples=10, guidance_scale=2.0,
                    apply_smooth=True, checkpoint="best",
                    output_dir=None, sample_label=None):
    """Generate motion for a target FMA score.

    Args:
        fma_score     : FMA score to generate (0–66)
        n_samples     : Samples to average (default 10 — matches eval protocol)
        guidance_scale: CFG guidance strength (default 2.0)
        apply_smooth  : Apply 6Hz low-pass filter to output
        checkpoint    : 'best' or 'final'
        output_dir    : Override default output/generated/csv directory
        sample_label  : If given, appended to filename (e.g. FMA_50_s00.csv)

    Returns:
        np.ndarray (SEQ_LEN, 15) in absolute mm
    """
    models_dir = _resolve_models_dir()
    model, scaler, _ = _load_model(models_dir, checkpoint)

    use_cfg = model.use_cfg
    gs = guidance_scale if use_cfg else 1.0

    print(f"Generating FMA {fma_score}  (n={n_samples}, guidance={gs:.1f})...")

    c = torch.FloatTensor([[fma_score / 66.0]]).to(DEVICE)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model.inference(c, seq_len=SEQ_LEN, guidance_scale=gs)
            raw = out.squeeze(0).cpu().numpy()
            samples.append(scaler.inverse_transform(raw))

    delta_mm = np.mean(samples, axis=0)
    abs_mm   = delta_mm + REF_ARRAY

    if apply_smooth:
        abs_mm = smooth(abs_mm)

    # Save CSV
    if output_dir is None:
        output_dir = get_path("output_generated_csv")
    os.makedirs(output_dir, exist_ok=True)

    if sample_label is not None:
        fname = f"FMA_{fma_score}_s{sample_label:02d}.csv"
    else:
        fname = f"FMA_{fma_score}.csv"

    out_path = os.path.join(output_dir, fname)
    pd.DataFrame(abs_mm, columns=COLS).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return abs_mm


def generate_range(fma_scores, n_samples=10, guidance_scale=2.0,
                   apply_smooth=True, checkpoint="best", output_dir=None):
    """Generate a batch of FMA scores. Loads model once for efficiency."""
    models_dir = _resolve_models_dir()
    model, scaler, _ = _load_model(models_dir, checkpoint)

    use_cfg = model.use_cfg
    gs = guidance_scale if use_cfg else 1.0

    if output_dir is None:
        output_dir = get_path("output_generated_csv")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {len(fma_scores)} FMA scores "
          f"(n={n_samples}, guidance={gs:.1f})...")

    for fma in fma_scores:
        c = torch.FloatTensor([[fma / 66.0]]).to(DEVICE)
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = model.inference(c, seq_len=SEQ_LEN, guidance_scale=gs)
                raw = out.squeeze(0).cpu().numpy()
                samples.append(scaler.inverse_transform(raw))

        delta_mm = np.mean(samples, axis=0)
        abs_mm   = delta_mm + REF_ARRAY
        if apply_smooth:
            abs_mm = smooth(abs_mm)

        out_path = os.path.join(output_dir, f"FMA_{fma}.csv")
        pd.DataFrame(abs_mm, columns=COLS).to_csv(out_path, index=False)

    print(f"Saved {len(fma_scores)} files to {output_dir}/")
    return output_dir


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_motion(data, fma_score):
    print(f"\n=== Motion Analysis (FMA {fma_score}) ===")
    sh, el, wr = data[:, 0:3], data[:, 3:6], data[:, 6:9]
    ua = np.linalg.norm(el - sh, axis=1)
    fa = np.linalg.norm(wr - el, axis=1)
    print(f"Upper arm: {ua.mean():.1f} ± {ua.std():.1f} mm")
    print(f"Forearm:   {fa.mean():.1f} ± {fa.std():.1f} mm")
    print(f"Wrist Y range: {wr[:, 1].max() - wr[:, 1].min():.1f} mm")
    if data.shape[1] >= 15:
        trunk = data[:, 12:15]
        trunk_disp = np.linalg.norm(trunk - trunk[0], axis=1).max()
        wrist_disp = np.linalg.norm(wr - wr[0], axis=1).max()
        ratio = trunk_disp / wrist_disp if wrist_disp > 0 else 0
        print(f"Trunk/Wrist ratio: {ratio:.3f} "
              f"(trunk={trunk_disp:.1f}mm, wrist={wrist_disp:.1f}mm)")


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise(data, fma_score):
    sh, el, wr = data[:, 0:3], data[:, 3:6], data[:, 6:9]
    trunk = data[:, 12:15] if data.shape[1] >= 15 else None
    pad = 50

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Generated Motion: FMA {fma_score}")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    all_pts = np.concatenate([sh, el, wr])
    ax.set_xlim([all_pts[:,0].min()-pad, all_pts[:,0].max()+pad])
    ax.set_ylim([all_pts[:,1].min()-pad, all_pts[:,1].max()+pad])
    ax.set_zlim([all_pts[:,2].min()-pad, all_pts[:,2].max()+pad])

    line1, = ax.plot([], [], [], 'b-',  lw=4, label='Upper Arm')
    line2, = ax.plot([], [], [], '-',   lw=4, color='orange', label='Forearm')
    pt_sh,  = ax.plot([], [], [], 'ko', markersize=8)
    pt_el,  = ax.plot([], [], [], 'bo', markersize=6)
    pt_wr,  = ax.plot([], [], [], 'ro', markersize=6)
    trail,  = ax.plot([], [], [], 'r:', lw=1, alpha=0.5)

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Y (Forward)'); ax2.set_ylabel('Z (Up)')
    ax2.set_title('Wrist Trajectory + Trunk')
    ax2.plot(wr[:, 1], wr[:, 2], 'b-', lw=2, label='Wrist')
    if trunk is not None:
        ax2.plot(trunk[:, 1] * 10, trunk[:, 2] * 10, 'g--', lw=1, alpha=0.7, label='Trunk (×10)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    def update(frame):
        s, e, w = sh[frame], el[frame], wr[frame]
        line1.set_data_3d([s[0], e[0]], [s[1], e[1]], [s[2], e[2]])
        line2.set_data_3d([e[0], w[0]], [e[1], w[1]], [e[2], w[2]])
        pt_sh.set_data_3d([s[0]], [s[1]], [s[2]])
        pt_el.set_data_3d([e[0]], [e[1]], [e[2]])
        pt_wr.set_data_3d([w[0]], [w[1]], [w[2]])
        trail.set_data_3d(wr[:frame+1, 0], wr[:frame+1, 1], wr[:frame+1, 2])
        return line1, line2, pt_sh, pt_el, pt_wr, trail

    FuncAnimation(fig, update, frames=len(data), interval=50, blit=False)
    plt.legend(); plt.tight_layout(); plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FMA-conditioned motion CSVs")
    parser.add_argument("--fma", default="50",
                        help="FMA score(s): single '50', range '16-66', or list '18,30,50,66'")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Samples to average per FMA score (default: 10)")
    parser.add_argument("--guidance", type=float, default=2.0,
                        help="CFG guidance scale (default: 2.0)")
    parser.add_argument("--checkpoint", default="best", choices=["best", "final"],
                        help="Which checkpoint to load (default: best)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Skip low-pass smoothing")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip 3D animation (for batch use)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Parse FMA scores
    fma_arg = args.fma.strip()
    if "-" in fma_arg and "," not in fma_arg:
        lo, hi = map(int, fma_arg.split("-"))
        scores = list(range(lo, hi + 1))
    elif "," in fma_arg:
        scores = [int(x) for x in fma_arg.split(",")]
    else:
        scores = [int(fma_arg)]

    if len(scores) == 1:
        data = generate_motion(
            fma_score=scores[0],
            n_samples=args.n_samples,
            guidance_scale=args.guidance,
            apply_smooth=not args.no_smooth,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
        )
        if data is not None:
            analyze_motion(data, scores[0])
            if not args.no_viz:
                visualise(data, scores[0])
    else:
        generate_range(
            fma_scores=scores,
            n_samples=args.n_samples,
            guidance_scale=args.guidance,
            apply_smooth=not args.no_smooth,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
        )
