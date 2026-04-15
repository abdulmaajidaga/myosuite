"""
train.py — Unified training script for test2 experiments.

Usage:
  python test/train.py --experiment A1_no_residual
  python test/train.py --experiment C1_seg_only --epochs 200

Outputs to: test/output/{experiment_name}/
  model_best.pth   — best validation checkpoint
  model_final.pth  — end of training
  config.json      — model + loss config (needed for generate.py)
  history.csv      — epoch-level train/val loss
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import joblib, glob, math
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

# ── Path setup ───────────────────────────────────────────────────────────────
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TEST_DIR)
sys.path.insert(0, ROOT_DIR)

from models import MotionCVAE, SEQ_LEN, INPUT_DIM
from experiments import EXPERIMENTS
from src.utils.config import get_path, get

# ── Defaults ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARM_COLS   = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
              'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x','Trunk_y','Trunk_z']
COLS       = ARM_COLS + TRUNK_COLS

AUG_DATA_PATHS = {
    "dtw":    "data_cutoff_augmented_dtw",
    "smote":  "data_cutoff_augmented_smote",
    "linear": "data_cutoff_augmented_linear",
}

# ── Dataset ──────────────────────────────────────────────────────────────────

class MotionDataset(Dataset):
    def __init__(self, data_dir, scaler, max_samples=15000, include_fma=None):
        """
        include_fma: if given, only load files whose FMA score is in this set.
                     Use to create train/val splits by FMA level (held-out split).
        """
        import re, random
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

        # Group by FMA score for stratified sampling
        groups = {}
        for f in all_files:
            m = re.search(r'_FMA(\d+)\.csv$', f)
            if m:
                s = int(m.group(1))
                if include_fma is not None and s not in include_fma:
                    continue
                groups.setdefault(s, []).append((f, float(s)))

        total = sum(len(v) for v in groups.values())
        if max_samples and total > max_samples:
            random.seed(42)
            per_score = max_samples // len(groups)
            files = []
            for s in sorted(groups):
                g = groups[s]
                files.extend(random.sample(g, min(per_score, len(g))))
            random.shuffle(files)
        else:
            files = [(f, s) for grp in groups.values() for f, s in grp]

        print(f"Loading {len(files)} files (total available: {total})...")
        motions, scores, skipped = [], [], 0
        for i, (path, score) in enumerate(files):
            try:
                df = pd.read_csv(path)
                for col in COLS:
                    if col not in df.columns:
                        df[col] = 0.0
                data = df[COLS].values
                if len(data) != SEQ_LEN:
                    data = resample(data, SEQ_LEN)
                motions.append(scaler.transform(data))
                scores.append(score / 66.0)
            except Exception:
                skipped += 1
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(files)} cached...", flush=True)

        self.motions = torch.FloatTensor(np.array(motions))
        self.scores  = torch.FloatTensor(np.array(scores)).unsqueeze(1)
        print(f"  Cached {len(self.motions)} samples"
              + (f", skipped {skipped}" if skipped else ""))

    def __len__(self):  return len(self.motions)
    def __getitem__(self, idx): return self.motions[idx], self.scores[idx]


# ── Loss function ─────────────────────────────────────────────────────────────

def pearson(a, b):
    if a.shape[0] < 4 or a.std() < 1e-6 or b.std() < 1e-6:
        return torch.tensor(0.0, device=a.device)
    a_c, b_c = a - a.mean(), b - b.mean()
    return (a_c * b_c).sum() / (torch.norm(a_c) * torch.norm(b_c) + 1e-8)


def dynamics_loss(recon_x, scores):
    """FMA → peak velocity ↑,  FMA → jerk ↓."""
    s = scores.squeeze(-1)
    wv = recon_x[:, 1:, 6:9] - recon_x[:, :-1, 6:9]
    speed = torch.norm(wv, dim=-1)
    peak  = speed.max(dim=1).values
    wa    = wv[:, 1:] - wv[:, :-1]
    wj    = wa[:, 1:] - wa[:, :-1]
    jerk  = (wj ** 2).sum(dim=-1).sum(dim=-1)
    path  = speed.sum(dim=1) + 1e-6
    T     = float(recon_x.shape[1])
    njerk = (T**5) / (2 * path**2) * jerk
    return (1.0 - pearson(peak, s)) + (1.0 + pearson(njerk, s))


def segment_length_loss(recon_x):
    """Penalise temporal variance of upper-arm and forearm lengths (SCALED SPACE).
    BROKEN — StandardScaler distorts Euclidean distances. Use segment_length_loss_physical instead.
    """
    Sh = recon_x[:, :, 0:3]
    El = recon_x[:, :, 3:6]
    Wr = recon_x[:, :, 6:9]
    ua_len = torch.norm(El - Sh, dim=-1)   # (B, T)
    fa_len = torch.norm(Wr - El, dim=-1)
    return ua_len.var(dim=1).mean() + fa_len.var(dim=1).mean()


def segment_length_loss_physical(recon_x, scaler_scale, scaler_mean):
    """Penalise temporal variance of upper-arm and forearm lengths in PHYSICAL mm space.

    Converts scaled features back to mm before computing Euclidean distances.
    This is the correct implementation — C1 failed because it computed distances
    in StandardScaler space where each axis has different effective units.

    scaler_scale: (15,) tensor of scaler.scale_ values
    scaler_mean:  (15,) tensor of scaler.mean_ values
    Col layout: Sh(0:3), El(3:6), Wr(6:9)
    """
    # Inverse-transform to physical mm: x_phys = x_scaled * scale + mean
    # broadcast: recon_x is (B, T, 15), scaler tensors are (15,)
    phys = recon_x * scaler_scale + scaler_mean   # (B, T, 15)

    Sh = phys[:, :, 0:3]   # shoulder
    El = phys[:, :, 3:6]   # elbow
    Wr = phys[:, :, 6:9]   # wrist

    ua_len = torch.norm(El - Sh, dim=-1)   # (B, T) upper-arm length in mm
    fa_len = torch.norm(Wr - El, dim=-1)   # (B, T) forearm length in mm

    # Minimise temporal variance — segment lengths should be constant (rigid body)
    return ua_len.var(dim=1).mean() + fa_len.var(dim=1).mean()


def sagittal_constraint_loss(recon_x):
    """Penalise lateral (X-axis) deviation of elbow and wrist from frame-0.

    The drinking/reaching task is sagittal-plane. Wrist and elbow should not
    drift laterally. This reduces pro/sup overestimation and IK routing errors.
    Col 3 = El_x,  col 6 = Wr_x  (in scaled feature space).
    """
    el_x = recon_x[:, :, 3]
    wr_x = recon_x[:, :, 6]
    el_dev = (el_x - el_x[:, 0:1]).pow(2).mean()
    wr_dev = (wr_x - wr_x[:, 0:1]).pow(2).mean()
    return el_dev + wr_dev



def compute_loss(recon_x, x, mu, logvar, scores, weights,
                 scaler_scale=None, scaler_mean=None):
    """Combined VAE loss with configurable component weights."""
    w = weights

    recon = nn.functional.mse_loss(recon_x, x)

    vel_r = x[:, 1:, :]     - x[:, :-1, :]
    vel_p = recon_x[:, 1:, :] - recon_x[:, :-1, :]
    vel   = nn.functional.mse_loss(vel_p, vel_r)

    acc_r = vel_r[:, 1:, :] - vel_r[:, :-1, :]
    acc_p = vel_p[:, 1:, :] - vel_p[:, :-1, :]
    acc   = nn.functional.mse_loss(acc_p, acc_r)

    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    dyn = dynamics_loss(recon_x, scores) if w.get('w_dyn', 0) > 0 else torch.tensor(0.0, device=x.device)

    total = (recon
             + w.get('w_vel', 10.0) * vel
             + w.get('w_acc',  5.0) * acc
             + w.get('w_kl',   0.1) * kld
             + w.get('w_dyn',  2.0) * dyn)
    return total, recon, vel, kld


# ── Training loop ─────────────────────────────────────────────────────────────

def train_experiment(exp_name: str, epochs: int = DEFAULT_EPOCHS,
                     max_samples: int = 15000, resume: bool = False):

    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{exp_name}'. "
                         f"Available: {list(EXPERIMENTS.keys())}")

    cfg = EXPERIMENTS[exp_name]
    out_dir = os.path.join(TEST_DIR, "output", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"  {cfg['desc']}")
    print(f"  Aug: {cfg['aug']}  |  Device: {DEVICE}")
    print("=" * 60)

    # Data
    data_dir = get_path(AUG_DATA_PATHS[cfg["aug"]])
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Loaded cached scaler.")
    else:
        print("Fitting scaler...")
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        step = 50 if len(all_files) > 10000 else max(1, len(all_files) // 200)
        raw = []
        for f in all_files[::step]:
            try:
                df = pd.read_csv(f)
                for col in COLS:
                    if col not in df.columns: df[col] = 0.0
                raw.append(df[COLS].values)
            except Exception: pass
        scaler = StandardScaler().fit(np.vstack(raw))
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler fitted on {len(raw)} files.")

    held_out_fma = cfg.get("held_out_fma", None)
    if held_out_fma:
        # FMA-level split: train on all FMA levels NOT in held_out, val on held_out
        all_fma = set(range(16, 67))
        train_fma = all_fma - set(held_out_fma)
        print(f"FMA-level split: train on {sorted(train_fma)}, val on {sorted(held_out_fma)}")
        train_ds = MotionDataset(data_dir, scaler, max_samples, include_fma=train_fma)
        val_ds   = MotionDataset(data_dir, scaler, max_samples=None, include_fma=set(held_out_fma))
    else:
        dataset = MotionDataset(data_dir, scaler, max_samples)
        train_n = int(0.9 * len(dataset))
        train_ds, val_ds = random_split(dataset, [train_n, len(dataset) - train_n])

    pin = DEVICE.type == 'cuda'
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}\n")

    # Model
    model = MotionCVAE(cfg["model"]).to(DEVICE)
    print(model.describe())

    best_path  = os.path.join(out_dir, "model_best.pth")
    final_path = os.path.join(out_dir, "model_final.pth")
    start_epoch = 0
    best_val = float('inf')

    if resume and os.path.exists(final_path):
        model.load_state_dict(torch.load(final_path, map_location=DEVICE))
        print(f"Resumed from {final_path}")

    # Save config alongside model so generate.py can reconstruct it
    config_out = {
        "experiment": exp_name,
        "model": cfg["model"],
        "aug":   cfg["aug"],
        "losses": cfg["losses"],
    }
    if held_out_fma:
        config_out["held_out_fma"] = held_out_fma
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    lr = cfg.get("learning_rate", LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    loss_weights = {
        "w_vel": 10.0, "w_acc": 5.0, "w_kl": 0.1,
        "w_dyn": 2.0,
    }
    loss_weights.update(cfg.get("losses", {}))

    # Pre-compute scaler tensors for physical-space segment loss
    scaler_scale_t = torch.FloatTensor(scaler.scale_).to(DEVICE)
    scaler_mean_t  = torch.FloatTensor(scaler.mean_).to(DEVICE)

    history = []

    for epoch in range(start_epoch, epochs):
        model.train()
        t_loss = 0
        for motion, score in train_loader:
            motion = motion.to(DEVICE, non_blocking=True)
            score  = score.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            recon, mu, logvar = model(motion, score)
            loss, *_ = compute_loss(recon, motion, mu, logvar, score, loss_weights,
                                    scaler_scale_t, scaler_mean_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for motion, score in val_loader:
                motion = motion.to(DEVICE, non_blocking=True)
                score  = score.to(DEVICE, non_blocking=True)
                recon, mu, logvar = model(motion, score)
                loss, *_ = compute_loss(recon, motion, mu, logvar, score, loss_weights,
                                        scaler_scale_t, scaler_mean_t)
                v_loss += loss.item()
        v_loss /= len(val_loader)
        scheduler.step(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), best_path)

        lr = optimizer.param_groups[0]['lr']
        history.append({"epoch": epoch+1, "train": t_loss, "val": v_loss, "lr": lr})

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {t_loss:.4f} | "
                  f"Val: {v_loss:.4f} | LR: {lr:.6f}", flush=True)

    torch.save(model.state_dict(), final_path)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    print(f"\nSaved: {final_path}")
    print(f"Best:  {best_path}  (val={best_val:.4f})")
    print(f"History: {os.path.join(out_dir, 'history.csv')}")
    return out_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a test2 experiment")
    parser.add_argument("--experiment", "-e", required=True,
                        help="Experiment name from experiments.py")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-samples", type=int, default=15000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    args = parser.parse_args()

    if args.list:
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name:25s}  {cfg['desc']}")
        sys.exit(0)

    train_experiment(args.experiment, args.epochs, args.max_samples, args.resume)
