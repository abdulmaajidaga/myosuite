"""
analyse_variance.py — Sample-level variance analysis for D_base (best model + n=10 averaging).

Generates N_SAMPLES independent samples per FMA level from D_base,
computes wrist_range per sample (no averaging), then plots:
  1. Bell curves at representative FMA levels (16, 30, 45, 66)
  2. Mean wrist_range ± std vs FMA score
  3. Coefficient of variation vs FMA score

Outputs: test/output/D_base/variance/
"""

import os, sys, json
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr, norm

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TEST_DIR)
sys.path.insert(0, ROOT_DIR)
from models import MotionCVAE, SEQ_LEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Config ---
EXPERIMENT   = "D_base"      # base model (weights live here)
OUTPUT_ALIAS = "D_base"      # output goes here
N_SAMPLES    = 50                        # independent draws per FMA level
FMA_SCORES   = list(range(16, 67))       # all 51 levels
BELL_FMAS    = [16, 30, 45, 66]          # FMA levels for bell-curve panel

ARM_COLS   = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
              'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x','Trunk_y','Trunk_z']
COLS       = ARM_COLS + TRUNK_COLS

REFERENCE_POSE = {
    'Sh_x': -77.3,  'Sh_y': 643.0, 'Sh_z': 302.7,
    'El_x': -188.2, 'El_y': 474.4, 'El_z':  41.3,
    'Wr_x':  -88.7, 'Wr_y': 241.2, 'Wr_z':  41.1,
    'WrVec_x': -37.0, 'WrVec_y': 12.0, 'WrVec_z': -33.1,
    'Trunk_x': 0.0, 'Trunk_y': 0.0, 'Trunk_z': 0.0,
}
REF_ARRAY = np.array([REFERENCE_POSE[c] for c in COLS])

WR_Y_IDX = COLS.index('Wr_y')


def smooth(data, cutoff=6, fs=100):
    nyq = 0.5 * fs
    b, a = signal.butter(2, min(cutoff / nyq, 0.99), btype='low')
    out = data.copy()
    for i in range(data.shape[1]):
        out[:, i] = signal.filtfilt(b, a, data[:, i])
    return out


def wrist_range(abs_mm):
    """Peak-to-peak range of Wr_y (mm)."""
    wy = abs_mm[:, WR_Y_IDX]
    return wy.max() - wy.min()


def main():
    out_dir = os.path.join(TEST_DIR, "output", OUTPUT_ALIAS, "variance")
    os.makedirs(out_dir, exist_ok=True)

    exp_dir = os.path.join(TEST_DIR, "output", EXPERIMENT)
    ckpt    = os.path.join(exp_dir, "model_best.pth")
    cfg_p   = os.path.join(exp_dir, "config.json")
    scaler  = joblib.load(os.path.join(exp_dir, "scaler.pkl"))

    with open(cfg_p) as f:
        config = json.load(f)

    model = MotionCVAE(config["model"]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    # Use CFG guidance=2.0 when model supports CFG
    if not config["model"].get("use_cfg", False):
        gs = 1.0
    else:
        gs = 2.0

    print(f"Model: {EXPERIMENT}, guidance={gs}, device={DEVICE}")
    print(f"Generating {N_SAMPLES} samples × {len(FMA_SCORES)} FMA levels...")

    # results[fma] = list of wrist_range values (length N_SAMPLES)
    results = {}
    for fma in FMA_SCORES:
        c = torch.FloatTensor([[fma / 66.0]]).to(DEVICE)
        ranges = []
        with torch.no_grad():
            for s in range(N_SAMPLES):
                torch.manual_seed(s)
                out = model.inference(c, seq_len=SEQ_LEN, guidance_scale=gs)
                raw = out.squeeze(0).cpu().numpy()
                delta = scaler.inverse_transform(raw)
                abs_mm = delta + REF_ARRAY
                abs_mm = smooth(abs_mm)
                ranges.append(wrist_range(abs_mm))
        results[fma] = np.array(ranges)
        if fma % 10 == 0:
            print(f"  FMA {fma}: mean={np.mean(ranges):.1f}  std={np.std(ranges):.1f}")

    # Save raw data
    np.save(os.path.join(out_dir, "sample_wrist_ranges.npy"), results)

    fma_arr   = np.array(FMA_SCORES)
    means     = np.array([results[f].mean() for f in FMA_SCORES])
    stds      = np.array([results[f].std()  for f in FMA_SCORES])
    cvs       = stds / (means + 1e-9)

    rho, pval = spearmanr(fma_arr, means)
    print(f"\nSpearman ρ (mean wrist_range vs FMA): {rho:.3f}  p={pval:.2e}")

    # ── Plot 1: Mean ± std ribbon ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(fma_arr, means - stds, means + stds, alpha=0.25,
                    color='steelblue', label='±1 SD')
    ax.fill_between(fma_arr, means - 2*stds, means + 2*stds, alpha=0.10,
                    color='steelblue', label='±2 SD')
    ax.plot(fma_arr, means, color='steelblue', lw=2, label='Mean')
    ax.set_xlabel("FMA Score", fontsize=12)
    ax.set_ylabel("Wrist Y Range (mm)", fontsize=12)
    ax.set_title(f"Wrist Range vs FMA  (ρ={rho:.3f}, N={N_SAMPLES} samples/level)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wrist_range_ribbon.png"), dpi=150)
    plt.close()
    print("Saved: wrist_range_ribbon.png")

    # ── Plot 2: Bell curves at BELL_FMAS ───────────────────────────────────
    fig, axes = plt.subplots(1, len(BELL_FMAS), figsize=(14, 4), sharey=False)
    colors = ['#d73027', '#fc8d59', '#4dac26', '#2c7bb6']
    for ax, fma, color in zip(axes, BELL_FMAS, colors):
        data  = results[fma]
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min() - 5, data.max() + 5, 200)
        ax.hist(data, bins=15, density=True, alpha=0.55, color=color, edgecolor='white')
        ax.plot(x, norm.pdf(x, mu, sigma), color=color, lw=2.5,
                label=f'μ={mu:.0f}\nσ={sigma:.0f}')
        ax.axvline(mu, color='black', lw=1.2, ls='--')
        ax.set_title(f"FMA {fma}", fontsize=12)
        ax.set_xlabel("Wrist Y Range (mm)", fontsize=10)
        ax.set_ylabel("Density" if fma == BELL_FMAS[0] else "", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Sample Distribution of Wrist Range  (N={N_SAMPLES} per level)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bell_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: bell_curves.png")

    # ── Plot 3: Coefficient of Variation vs FMA ────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(fma_arr, cvs * 100, color='darkorange', lw=2)
    ax.set_xlabel("FMA Score", fontsize=12)
    ax.set_ylabel("CV (%)", fontsize=12)
    ax.set_title("Model Uncertainty vs FMA Score (CV = σ/μ)", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_vs_fma.png"), dpi=150)
    plt.close()
    print("Saved: cv_vs_fma.png")

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = {
        "n_fma_levels": len(FMA_SCORES),
        "n_samples_per_level": N_SAMPLES,
        "spearman_rho_means": round(float(rho), 4),
        "spearman_p": float(pval),
        "mean_std_across_fma": round(float(stds.mean()), 2),
        "mean_cv_pct": round(float(cvs.mean()) * 100, 2),
        "worst_cv_fma": int(fma_arr[cvs.argmax()]),
        "best_cv_fma":  int(fma_arr[cvs.argmin()]),
    }
    with open(os.path.join(out_dir, "variance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nVariance summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
