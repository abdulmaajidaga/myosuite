"""
cvae_train.py — Train the FMA-conditioned motion CVAE.

Default config (D_base, best validated — wrist_rho=0.914, trunk_rho=-0.895):
  Architecture : FiLM + CFG, no residual (Stage 2)
  Data source  : SMOTE augmented (~58k files, 15k cap)
  Losses       : w_vel=10, w_acc=2.5, w_kl=0.2, w_dyn=2.0
  lr           : 1e-3, ReduceLROnPlateau patience=15

Outputs (models/cvae/):
  cvae_cutoff_fma_best.pth   — best validation checkpoint
  cvae_cutoff_fma.pth        — final checkpoint
  scaler_cutoff_fma.pkl      — StandardScaler (must be kept with model)
  config.json                — model + loss config (needed by cvae_generate.py)

Usage:
  python src/generation/cvae_train.py                        # defaults
  python src/generation/cvae_train.py -e 400 --data-source smote
  python src/generation/cvae_train.py --max-samples all      # full dataset
"""

import os, sys, json, argparse, glob, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.generation.model import MotionCVAE, SEQ_LEN, INPUT_DIM
from src.utils.config import get_path, get

# ── Constants ─────────────────────────────────────────────────────────────────
ARM_COLS   = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
              'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x','Trunk_y','Trunk_z']
COLS       = ARM_COLS + TRUNK_COLS

BATCH_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# D_base — best config from systematic ablation (Phases A–D in test/)
BEST_MODEL_CONFIG = {
    "use_film":          True,
    "use_cfg":           True,
    "use_residual":      False,
    "use_temporal_conv": False,
    "latent_dim":        32,
}

BEST_LOSS_WEIGHTS = {
    "w_vel": 10.0,
    "w_acc":  2.5,
    "w_kl":   0.2,
    "w_dyn":  2.0,
}

DATA_SOURCE_PATHS = {
    "smote":  "data_cutoff_augmented_smote",
    "dtw":    "data_cutoff_augmented_dtw",
    "linear": "data_cutoff_augmented_linear",
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class MotionDataset(Dataset):
    """Pre-caches all data into RAM tensors for fast GPU training."""

    def __init__(self, data_dir, scaler, max_samples=15000):
        import re, random
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

        groups = {}
        for f in all_files:
            m = re.search(r'_FMA(\d+)\.csv$', f)
            if m:
                s = int(m.group(1))
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


# ── Loss ──────────────────────────────────────────────────────────────────────

def _pearson(a, b):
    if a.shape[0] < 4 or a.std() < 1e-6 or b.std() < 1e-6:
        return torch.tensor(0.0, device=a.device)
    a_c, b_c = a - a.mean(), b - b.mean()
    return (a_c * b_c).sum() / (torch.norm(a_c) * torch.norm(b_c) + 1e-8)


def dynamics_loss(recon_x, scores):
    """FMA → peak velocity ↑,  FMA → jerk ↓."""
    s  = scores.squeeze(-1)
    wv = recon_x[:, 1:, 6:9] - recon_x[:, :-1, 6:9]
    speed = torch.norm(wv, dim=-1)
    peak  = speed.max(dim=1).values
    wa    = wv[:, 1:] - wv[:, :-1]
    wj    = wa[:, 1:] - wa[:, :-1]
    jerk  = (wj ** 2).sum(dim=-1).sum(dim=-1)
    path  = speed.sum(dim=1) + 1e-6
    T     = float(recon_x.shape[1])
    njerk = (T ** 5) / (2 * path ** 2) * jerk
    return (1.0 - _pearson(peak, s)) + (1.0 + _pearson(njerk, s))


def compute_loss(recon_x, x, mu, logvar, scores, weights):
    recon = nn.functional.mse_loss(recon_x, x)

    vel_r = x[:, 1:, :]       - x[:, :-1, :]
    vel_p = recon_x[:, 1:, :] - recon_x[:, :-1, :]
    vel   = nn.functional.mse_loss(vel_p, vel_r)

    acc_r = vel_r[:, 1:, :] - vel_r[:, :-1, :]
    acc_p = vel_p[:, 1:, :] - vel_p[:, :-1, :]
    acc   = nn.functional.mse_loss(acc_p, acc_r)

    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    dyn = dynamics_loss(recon_x, scores) if weights.get('w_dyn', 0) > 0 else \
          torch.tensor(0.0, device=x.device)

    total = (recon
             + weights.get('w_vel', 10.0) * vel
             + weights.get('w_acc',  2.5) * acc
             + weights.get('w_kl',   0.2) * kld
             + weights.get('w_dyn',  2.0) * dyn)
    return total, recon, vel, kld


# ── Scaler fitting ────────────────────────────────────────────────────────────

def fit_scaler(data_dir):
    print("Fitting scaler...")
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    step = 50 if len(all_files) > 10000 else max(1, len(all_files) // 200)
    raw = []
    for f in all_files[::step]:
        try:
            df = pd.read_csv(f)
            for col in COLS:
                if col not in df.columns:
                    df[col] = 0.0
            raw.append(df[COLS].values)
        except Exception:
            pass
    scaler = StandardScaler().fit(np.vstack(raw))
    print(f"  Scaler fitted on {len(raw)} files.")
    return scaler


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_source="smote", epochs=300, max_samples=15000,
          model_config=None, loss_weights=None, learning_rate=1e-3):

    model_config  = model_config  or BEST_MODEL_CONFIG
    loss_weights  = loss_weights  or BEST_LOSS_WEIGHTS
    data_dir      = get_path(DATA_SOURCE_PATHS[data_source])
    models_dir    = os.path.join(get_path("output_dir").replace("output", "models"), "cvae")
    # Resolve properly via config
    import src.utils.config as _cfg
    root = _cfg.get_project_root()
    models_dir    = os.path.join(root, "models", "cvae")

    best_path   = os.path.join(models_dir, "cvae_cutoff_fma_best.pth")
    final_path  = os.path.join(models_dir, "cvae_cutoff_fma.pth")
    scaler_path = os.path.join(models_dir, "scaler_cutoff_fma.pkl")
    config_path = os.path.join(models_dir, "config.json")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print(f"Training MotionCVAE  |  D_base config")
    print(f"  Data: {data_source}  |  Device: {DEVICE}")
    print(f"  Epochs: {epochs}  |  Max samples: {max_samples or 'ALL'}")
    print(f"  Model: {model_config}")
    print(f"  Losses: {loss_weights}")
    print("=" * 60)

    # Scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Loaded cached scaler.")
    else:
        scaler = fit_scaler(data_dir)
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved to {scaler_path}")

    # Dataset
    dataset  = MotionDataset(data_dir, scaler, max_samples)
    train_n  = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_n, len(dataset) - train_n])

    pin = DEVICE.type == 'cuda'
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}\n")

    # Model
    model = MotionCVAE(model_config).to(DEVICE)
    print(model.describe())

    # Persist config for generate.py
    with open(config_path, "w") as f:
        json.dump({"model": model_config, "losses": loss_weights,
                   "data_source": data_source}, f, indent=2)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    best_val = float('inf')
    history  = []

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for motion, score in train_loader:
            motion = motion.to(DEVICE, non_blocking=True)
            score  = score.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            recon, mu, logvar = model(motion, score)
            loss, *_ = compute_loss(recon, motion, mu, logvar, score, loss_weights)
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
                loss, *_ = compute_loss(recon, motion, mu, logvar, score, loss_weights)
                v_loss += loss.item()
        v_loss /= len(val_loader)
        scheduler.step(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), best_path)

        lr = optimizer.param_groups[0]['lr']
        history.append({"epoch": epoch + 1, "train": t_loss, "val": v_loss, "lr": lr})

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {t_loss:.4f} | "
                  f"Val: {v_loss:.4f} | LR: {lr:.6f}", flush=True)

    torch.save(model.state_dict(), final_path)
    pd.DataFrame(history).to_csv(
        os.path.join(models_dir, "training_history.csv"), index=False)

    print(f"\nFinal:  {final_path}")
    print(f"Best:   {best_path}  (val={best_val:.4f})")
    print(f"Scaler: {scaler_path}")
    print(f"Config: {config_path}")
    return best_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FMA-conditioned MotionCVAE (D_base config)")
    parser.add_argument("-e", "--epochs", type=int, default=300,
                        help="Training epochs (default: 300)")
    parser.add_argument("-n", "--max-samples", type=str, default="15000",
                        help="Max samples per run, or 'all' (default: 15000)")
    parser.add_argument("--data-source", default="smote",
                        choices=["smote", "dtw", "linear"],
                        help="Augmentation dataset (default: smote)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    args = parser.parse_args()

    max_samples = None if args.max_samples.lower() == "all" else int(args.max_samples)

    train(data_source=args.data_source,
          epochs=args.epochs,
          max_samples=max_samples,
          learning_rate=args.lr)
