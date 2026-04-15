import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
import glob
import joblib
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
import math
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path, get

# --- CONFIGURATION ---
# Data source is selected via --data-source flag (dtw or smote)
# Default: dtw (existing DTW-morphed augmented data)
DATA_DIR = get_path("data_cutoff_augmented")  # overridden by CLI flag

MODEL_SAVE_PATH = get_path("cvae_model")
SCALER_SAVE_PATH = get_path("cvae_scaler")

# Hyperparameters - Improved architecture
INPUT_DIM = 15       # 12 arm + 3 trunk
CONDITION_DIM = 1
HIDDEN_DIM = 256
LATENT_DIM = 32
NUM_HEADS = 4        # For self-attention
SEQ_LEN = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 12 arm columns + 3 trunk columns
ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
COLS = ARM_COLS + TRUNK_COLS

# --- 1. FMA-TARGETED DATASET (FMA scores embedded in filenames) ---

class CutoffDataset(Dataset):
    """Dataset that pre-caches all data into RAM for fast GPU training."""

    def __init__(self, data_dir, scaler, seq_len=100, max_samples=None):
        self.seq_len = seq_len
        import re
        import random

        # Load all CSV files - FMA score is in filename
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

        # Group files by FMA score
        fma_groups = {}
        for fpath in all_files:
            fname = os.path.basename(fpath)
            match = re.search(r'_FMA(\d+)\.csv$', fname)
            if match:
                score = int(match.group(1))
                if score not in fma_groups:
                    fma_groups[score] = []
                fma_groups[score].append((fpath, float(score)))

        total_files = sum(len(v) for v in fma_groups.values())

        # Stratified sampling: equal samples per FMA score
        if max_samples is not None and total_files > max_samples:
            random.seed(42)
            n_scores = len(fma_groups)
            samples_per_score = max_samples // n_scores

            files = []
            for score in sorted(fma_groups.keys()):
                files_for_score = fma_groups[score]
                if len(files_for_score) > samples_per_score:
                    files.extend(random.sample(files_for_score, samples_per_score))
                else:
                    files.extend(files_for_score)

            random.shuffle(files)
            print(f"Stratified sampling: {len(files)} from {total_files} files")
            print(f"  (~{samples_per_score} per FMA score, {n_scores} scores)")
        else:
            files = [(f, s) for flist in fma_groups.values() for f, s in flist]
            print(f"Loaded all {len(files)} files")

        # Score distribution summary
        if files:
            fma_counts = {}
            for _, score in files:
                fma_counts[int(score)] = fma_counts.get(int(score), 0) + 1
            scores_list = [f[1] for f in files]
            print(f"FMA range: {min(scores_list):.0f} - {max(scores_list):.0f}")
            print(f"Unique FMA scores: {len(fma_counts)}")

        # Pre-cache all data into RAM tensors (eliminates per-epoch CSV I/O)
        print(f"Pre-caching {len(files)} files into RAM...")
        motions = []
        scores = []
        n_skipped = 0
        for i, (path, score) in enumerate(files):
            try:
                df = pd.read_csv(path)
                for col in COLS:
                    if col not in df.columns:
                        df[col] = 0.0
                data = df[COLS].values
                if len(data) != seq_len:
                    data = resample(data, seq_len)
                data = scaler.transform(data)
                motions.append(data)
                scores.append(score / 66.0)
            except Exception:
                n_skipped += 1

            if (i + 1) % 500 == 0:
                print(f"  Cached {i+1}/{len(files)} files...", flush=True)

        self.motions = torch.FloatTensor(np.array(motions))  # (N, 100, 15)
        self.scores = torch.FloatTensor(np.array(scores)).unsqueeze(1)  # (N, 1)
        print(f"Pre-cached {len(self.motions)} samples into RAM "
              f"({self.motions.nbytes / 1e6:.0f} MB)"
              f"{f', skipped {n_skipped}' if n_skipped else ''}", flush=True)

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        return self.motions[idx], self.scores[idx]

# --- 2. MODEL (imported dynamically based on --model-version flag) ---
# Default import for backward compat; overridden by CLI flag
_MotionCVAE = None

# --- 3. LOSS FUNCTION (Position + Velocity + Acceleration) ---

def _pearson(a, b):
    """Pearson correlation between two 1D tensors. Returns 0 if degenerate."""
    if a.shape[0] < 4 or a.std() < 1e-6 or b.std() < 1e-6:
        return torch.tensor(0.0, device=a.device)
    a_c = a - a.mean()
    b_c = b - b.mean()
    return (a_c * b_c).sum() / (torch.norm(a_c) * torch.norm(b_c) + 1e-8)


def dynamics_correlation_loss(recon_x, scores):
    """Batch-level loss enforcing FMA-dependent temporal dynamics.

    Two terms:
      1. Peak velocity should positively correlate with FMA (healthy = faster)
      2. Normalized jerk should negatively correlate with FMA (healthy = smoother)

    Combined loss: (1 - vel_corr) + (1 + jerk_corr)
    Perfect score = 0 (vel_corr=+1, jerk_corr=-1).
    """
    scores_flat = scores.squeeze(-1)  # (B,)

    # Per-sample peak speed (wrist columns 6:9)
    wrist_vel = recon_x[:, 1:, 6:9] - recon_x[:, :-1, 6:9]  # (B, 99, 3)
    speed = torch.norm(wrist_vel, dim=-1)                      # (B, 99)
    peak_speed = speed.max(dim=1).values                        # (B,)

    # Per-sample normalized jerk (wrist only)
    wrist_acc = wrist_vel[:, 1:] - wrist_vel[:, :-1]           # (B, 98, 3)
    wrist_jerk = wrist_acc[:, 1:] - wrist_acc[:, :-1]          # (B, 97, 3)
    jerk_sq = (wrist_jerk ** 2).sum(dim=-1).sum(dim=-1)        # (B,)
    path_len = speed.sum(dim=1) + 1e-6                          # (B,)
    T = float(recon_x.shape[1])
    norm_jerk = (T ** 5) / (2 * path_len ** 2) * jerk_sq       # (B,)

    # Velocity: want positive correlation with FMA
    vel_corr = _pearson(peak_speed, scores_flat)

    # Jerk: want negative correlation with FMA (healthy = smooth = low jerk)
    jerk_corr = _pearson(norm_jerk, scores_flat)

    # Loss: 0 when vel_corr=+1 and jerk_corr=-1
    return (1.0 - vel_corr) + (1.0 + jerk_corr)


def spectral_smoothness_loss(recon_x, cutoff_hz=2.0, fs=200.0):
    """Penalize high-frequency energy above cutoff_hz in reconstructed signal.

    Uses FFT to compute power spectrum, then returns ratio of high-freq
    energy to total energy. Lower = smoother output.
    """
    spectrum = torch.fft.rfft(recon_x, dim=1)
    power = torch.abs(spectrum) ** 2
    freq_resolution = fs / recon_x.shape[1]  # Hz per bin
    k_cutoff = max(1, int(math.ceil(cutoff_hz / freq_resolution)))
    hf_energy = power[:, k_cutoff + 1:, :].sum(dim=1)
    total_energy = power.sum(dim=1)
    return (hf_energy / (total_energy + 1e-8)).mean()


def loss_function(recon_x, x, mu, logvar, scores=None, beta=0.1,
                  spectral_weight=0.0, spectral_cutoff=2.0):
    """VAE loss with reconstruction, velocity, acceleration, and dynamics correlation."""

    # Reconstruction loss (position)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')

    # Velocity loss (per-sample velocity profile matching)
    real_vel = x[:, 1:, :] - x[:, :-1, :]
    recon_vel = recon_x[:, 1:, :] - recon_x[:, :-1, :]
    vel_loss = nn.functional.mse_loss(recon_vel, real_vel, reduction='mean')

    # Acceleration loss
    real_acc = real_vel[:, 1:, :] - real_vel[:, :-1, :]
    recon_acc = recon_vel[:, 1:, :] - recon_vel[:, :-1, :]
    acc_loss = nn.functional.mse_loss(recon_acc, real_acc, reduction='mean')

    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Dynamics correlation (batch-level: FMA → peak velocity↑, jerk↓)
    dyn_corr = torch.tensor(0.0, device=recon_x.device)
    if scores is not None:
        dyn_corr = dynamics_correlation_loss(recon_x, scores)

    # Spectral smoothness loss (penalize high-freq jitter)
    spec_loss = torch.tensor(0.0, device=recon_x.device)
    if spectral_weight > 0:
        spec_loss = spectral_smoothness_loss(recon_x, cutoff_hz=spectral_cutoff)

    # Combined loss
    total = (recon_loss + 10.0 * vel_loss + 5.0 * acc_loss + beta * kld
             + 2.0 * dyn_corr + spectral_weight * spec_loss)

    return total, recon_loss, vel_loss, kld

# --- 4. TRAINING LOOP ---

def fit_scaler(data_dir=None):
    """Fit StandardScaler on all training data (sample for efficiency)."""
    if data_dir is None:
        data_dir = DATA_DIR
    print("Fitting scaler on sampled data...")
    all_data = []

    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    # Adaptive sampling: every 50th for large dirs, every 5th for small
    step = 50 if len(all_files) > 10000 else max(1, len(all_files) // 200)
    sample_files = all_files[::step]

    for f in sample_files:
        try:
            df = pd.read_csv(f)
            for col in COLS:
                if col not in df.columns:
                    df[col] = 0.0
            all_data.append(df[COLS].values)
        except:
            pass

    all_data = np.vstack(all_data)
    print(f"Scaler fitted on {len(all_data)} samples from {len(sample_files)} files")

    scaler = StandardScaler()
    scaler.fit(all_data)
    return scaler


def train(max_samples=None, epochs=EPOCHS, cond_drop_prob=0.1, data_dir=None,
          model_version="v2", spectral_weight=0.0, spectral_cutoff=2.0):
    global _MotionCVAE
    if data_dir is None:
        data_dir = DATA_DIR

    # Dynamic model import
    if model_version == "v1":
        from src.generation.model import MotionCVAE
        arch_label = "V1 (BiLSTM + SelfAttention + ResBlocks)"
    elif model_version == "v3":
        from src.generation.model_v3 import MotionCVAE
        arch_label = "V3 (BiLSTM Encoder + TimeVAE ConvDecoder + FiLM)"
    else:
        from src.generation.model_v2 import MotionCVAE
        arch_label = "V2 (BiLSTM + Attention Pooling + FiLM + TemporalConv)"
    _MotionCVAE = MotionCVAE

    print("=" * 60)
    print(f"Training CVAE FMA [{model_version.upper()}]")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Architecture: {arch_label}")
    print(f"Features: {INPUT_DIM} dims ({len(ARM_COLS)} arm + {len(TRUNK_COLS)} trunk)")
    print(f"Hidden: {HIDDEN_DIM}, Latent: {LATENT_DIM}, Heads: {NUM_HEADS}")
    print(f"Max samples: {max_samples if max_samples else 'ALL'}")
    print(f"Epochs: {epochs}")
    print(f"Condition dropout: {cond_drop_prob} (classifier-free guidance)")
    print("=" * 60 + "\n")

    # Fit and save scaler
    scaler = fit_scaler(data_dir=data_dir)
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}\n")

    # Load dataset (FMA scores from filenames)
    dataset = CutoffDataset(data_dir, scaler, SEQ_LEN, max_samples=max_samples)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    pin = DEVICE.type == 'cuda'
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=pin)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}\n", flush=True)

    # Versioned save paths: includes model version + data source
    data_tag = "smote" if "smote" in data_dir else ("linear" if "backup" in data_dir else "dtw")
    save_path = MODEL_SAVE_PATH.replace('.pth', f'_{model_version}_{data_tag}.pth')
    best_save_path = MODEL_SAVE_PATH.replace('.pth', f'_{model_version}_{data_tag}_best.pth')

    # Model with classifier-free guidance support
    model = _MotionCVAE(cond_drop_prob=cond_drop_prob).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for motion, score in train_loader:
            motion = motion.to(DEVICE, non_blocking=True)
            score = score.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            recon, mu, logvar = model(motion, score)
            loss, _, _, _ = loss_function(recon, motion, mu, logvar, scores=score,
                                          spectral_weight=spectral_weight,
                                          spectral_cutoff=spectral_cutoff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for motion, score in val_loader:
                motion = motion.to(DEVICE, non_blocking=True)
                score = score.to(DEVICE, non_blocking=True)
                recon, mu, logvar = model(motion, score)
                loss, _, _, _ = loss_function(recon, motion, mu, logvar, scores=score,
                                              spectral_weight=spectral_weight,
                                              spectral_cutoff=spectral_cutoff)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}",
                  flush=True)

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Best model saved to {best_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVAE FMA model")
    parser.add_argument("-n", "--num-samples", type=str, default="15000",
                        help="Number of samples to train on, or 'all' for full dataset (default: 15000)")
    parser.add_argument("-e", "--epochs", type=int, default=300,
                        help="Number of epochs (default: 300)")
    parser.add_argument("--cond-drop", type=float, default=0.1,
                        help="Condition dropout probability for classifier-free guidance (default: 0.1)")
    parser.add_argument("--data-source", type=str, default="dtw", choices=["dtw", "smote", "linear"],
                        help="Augmentation source: 'dtw' (default), 'smote', or 'linear'")
    parser.add_argument("--model-version", type=str, default="v2", choices=["v1", "v2", "v3"],
                        help="Model architecture: 'v1' (original), 'v2' (improved), or 'v3' (TimeVAE)")
    parser.add_argument("--spectral-weight", type=float, default=0.0,
                        help="Spectral smoothness loss weight (default: 0.0, auto-set to 1.0 for v3)")
    parser.add_argument("--spectral-cutoff", type=float, default=2.0,
                        help="Spectral loss cutoff frequency in Hz (default: 2.0)")
    args = parser.parse_args()

    # Select data directory based on source
    if args.data_source == "smote":
        DATA_DIR = get_path("data_cutoff_augmented_smote")
        print(f"[Data source: SMOTE augmented → {DATA_DIR}]")
    elif args.data_source == "linear":
        DATA_DIR = get_path("data_cutoff_augmented_linear")
        print(f"[Data source: Linear interpolation → {DATA_DIR}]")
    else:
        DATA_DIR = get_path("data_cutoff_augmented")
        print(f"[Data source: DTW augmented → {DATA_DIR}]")

    print(f"[Model version: {args.model_version.upper()}]")

    # Auto-set spectral weight for v3 if not explicitly provided
    spectral_weight = args.spectral_weight
    if args.model_version == "v3" and args.spectral_weight == 0.0:
        spectral_weight = 1.0
        print(f"[V3 auto-set: spectral_weight=1.0, cutoff={args.spectral_cutoff}Hz]")

    # Parse num_samples
    if args.num_samples.lower() == "all":
        max_samples = None
    else:
        max_samples = int(args.num_samples)

    train(max_samples=max_samples, epochs=args.epochs, cond_drop_prob=args.cond_drop,
          data_dir=DATA_DIR, model_version=args.model_version,
          spectral_weight=spectral_weight, spectral_cutoff=args.spectral_cutoff)