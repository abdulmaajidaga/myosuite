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
# Use FMA-targeted augmented data (FMA score embedded in filename)
DATA_DIR = get_path("data_cutoff_augmented")

MODEL_SAVE_PATH = get_path("cvae_model")
SCALER_SAVE_PATH = get_path("cvae_scaler")

# Hyperparameters - Improved architecture
INPUT_DIM = 15       # 12 arm + 3 trunk
CONDITION_DIM = 1
HIDDEN_DIM = 256
LATENT_DIM = 32
NUM_HEADS = 4        # For self-attention
SEQ_LEN = 100
BATCH_SIZE = 32 
LEARNING_RATE = 1e-3
EPOCHS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 12 arm columns + 3 trunk columns
ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
COLS = ARM_COLS + TRUNK_COLS

# --- 1. FMA-TARGETED DATASET (FMA scores embedded in filenames) ---

class CutoffDataset(Dataset):
    """Dataset that loads FMA-targeted augmented files with FMA score in filename."""

    def __init__(self, data_dir, scaler, seq_len=100, max_samples=None):
        self.files = []
        self.seq_len = seq_len
        self.scaler = scaler
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

            self.files = []
            for score in sorted(fma_groups.keys()):
                files_for_score = fma_groups[score]
                if len(files_for_score) > samples_per_score:
                    self.files.extend(random.sample(files_for_score, samples_per_score))
                else:
                    self.files.extend(files_for_score)

            random.shuffle(self.files)
            print(f"Stratified sampling: {len(self.files)} from {total_files} files")
            print(f"  (~{samples_per_score} per FMA score, {n_scores} scores)")
        else:
            self.files = [(f, s) for files in fma_groups.values() for f, s in files]
            print(f"Loaded all {len(self.files)} files")

        # Score distribution summary
        if self.files:
            fma_counts = {}
            for _, score in self.files:
                fma_counts[int(score)] = fma_counts.get(int(score), 0) + 1
            scores = [f[1] for f in self.files]
            print(f"FMA range: {min(scores):.0f} - {max(scores):.0f}")
            print(f"Unique FMA scores: {len(fma_counts)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, score = self.files[idx]

        df = pd.read_csv(path)
        for col in COLS:
            if col not in df.columns:
                df[col] = 0.0
        data = df[COLS].values

        if len(data) != self.seq_len:
            data = resample(data, self.seq_len)

        data = self.scaler.transform(data)

        motion_seq = torch.FloatTensor(data)
        score_val = torch.FloatTensor([score / 66.0])

        return motion_seq, score_val

# --- 2. MODEL (imported from shared module) ---
from src.generation.model import SelfAttention, Encoder, Decoder, ResidualBlock, MotionCVAE

# --- 3. LOSS FUNCTION (Position + Velocity + Acceleration) ---

def loss_function(recon_x, x, mu, logvar, beta=0.1):
    """VAE loss with reconstruction, velocity, and acceleration terms."""

    # Reconstruction loss (position)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')

    # Velocity loss (encourages smooth, realistic motion dynamics)
    real_vel = x[:, 1:, :] - x[:, :-1, :]
    recon_vel = recon_x[:, 1:, :] - recon_x[:, :-1, :]
    vel_loss = nn.functional.mse_loss(recon_vel, real_vel, reduction='mean')

    # Acceleration loss (even smoother motion)
    real_acc = real_vel[:, 1:, :] - real_vel[:, :-1, :]
    recon_acc = recon_vel[:, 1:, :] - recon_vel[:, :-1, :]
    acc_loss = nn.functional.mse_loss(recon_acc, real_acc, reduction='mean')

    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Combined loss (weighted sum)
    total = recon_loss + 10.0 * vel_loss + 5.0 * acc_loss + beta * kld

    return total, recon_loss, vel_loss, kld

# --- 4. TRAINING LOOP ---

def fit_scaler():
    """Fit StandardScaler on all training data (sample for efficiency)."""
    print("Fitting scaler on sampled data...")
    all_data = []

    # Sample every 50th file for scaler fitting (56k files -> ~1k samples)
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    sample_files = all_files[::50]

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


def train(max_samples=None, epochs=EPOCHS, cond_drop_prob=0.1):
    print("=" * 60)
    print("Training CVAE FMA: Direct FMA Score Targeting")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Features: {INPUT_DIM} dims ({len(ARM_COLS)} arm + {len(TRUNK_COLS)} trunk)")
    print(f"Architecture: BiLSTM + Self-Attention + Residual Blocks")
    print(f"Hidden: {HIDDEN_DIM}, Latent: {LATENT_DIM}, Heads: {NUM_HEADS}")
    print(f"Max samples: {max_samples if max_samples else 'ALL'}")
    print(f"Epochs: {epochs}")
    print(f"Condition dropout: {cond_drop_prob} (classifier-free guidance)")
    print("=" * 60 + "\n")

    # Fit and save scaler
    scaler = fit_scaler()
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}\n")

    # Load dataset (FMA scores from filenames)
    dataset = CutoffDataset(DATA_DIR, scaler, SEQ_LEN, max_samples=max_samples)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}\n")

    # Model with classifier-free guidance support
    model = MotionCVAE(cond_drop_prob=cond_drop_prob).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_val_loss = float('inf')
    best_model_path = MODEL_SAVE_PATH.replace('.pth', '_best.pth')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for motion, score in train_loader:
            motion = motion.to(DEVICE)
            score = score.to(DEVICE)

            optimizer.zero_grad()
            recon, mu, logvar = model(motion, score)
            loss, _, _, _ = loss_function(recon, motion, mu, logvar)
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
                motion = motion.to(DEVICE)
                score = score.to(DEVICE)
                recon, mu, logvar = model(motion, score)
                loss, _, _, _ = loss_function(recon, motion, mu, logvar)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 1 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")

    # Save final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVAE FMA model")
    parser.add_argument("-n", "--num-samples", type=str, default="15000",
                        help="Number of samples to train on, or 'all' for full dataset (default: 15000)")
    parser.add_argument("-e", "--epochs", type=int, default=300,
                        help="Number of epochs (default: 300)")
    parser.add_argument("--cond-drop", type=float, default=0.1,
                        help="Condition dropout probability for classifier-free guidance (default: 0.1)")
    args = parser.parse_args()

    # Parse num_samples
    if args.num_samples.lower() == "all":
        max_samples = None
    else:
        max_samples = int(args.num_samples)

    train(max_samples=max_samples, epochs=args.epochs, cond_drop_prob=args.cond_drop)