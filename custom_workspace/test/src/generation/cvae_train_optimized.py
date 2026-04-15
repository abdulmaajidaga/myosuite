import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
import glob
import re
import random
import math
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample

# --- Project Paths & Config ---
import sys
# Absolute path to 'test' directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.utils.config import get_path, get

# Hyperparameters for MAXIMUM SPEED
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 300
SEQ_LEN = 100
INPUT_DIM = 15
CONDITION_DIM = 1
HIDDEN_DIM = 256
LATENT_DIM = 32

# Device & Performance configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    from torch.amp import GradScaler, autocast
else:
    class autocast:
        def __init__(self, device_type, enabled=True): pass
        def __enter__(self): pass
        def __exit__(self, *args): pass
    class GradScaler:
        def __init__(self, device_type=None): pass
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass

# Feature columns
ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
COLS = ARM_COLS + TRUNK_COLS

# Save paths
MODEL_DIR = os.path.join(BASE_DIR, "models/cvae")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. DATASET ---

class CutoffDataset(Dataset):
    def __init__(self, data_dir, scaler, seq_len=100, max_samples=None):
        self.seq_len = seq_len
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        fma_groups = {}
        for fpath in all_files:
            fname = os.path.basename(fpath)
            match = re.search(r'FMA(\d+)', fname)
            if match:
                score = int(match.group(1))
                if score not in fma_groups: fma_groups[score] = []
                fma_groups[score].append((fpath, float(score)))

        total_files = sum(len(v) for v in fma_groups.values())
        if max_samples is not None and total_files > max_samples:
            n_scores = len(fma_groups)
            samples_per_score = max_samples // n_scores
            files = []
            for score in sorted(fma_groups.keys()):
                files_for_score = fma_groups[score]
                files.extend(random.sample(files_for_score, min(len(files_for_score), samples_per_score)))
            random.shuffle(files)
        else:
            files = [(f, s) for flist in fma_groups.values() for f, s in flist]

        print(f"Pre-caching {len(files)} files into RAM...")
        motions, scores = [], []
        for path, score in files:
            try:
                df = pd.read_csv(path)
                for c in COLS:
                    if c not in df.columns: df[c] = 0.0
                data = df[COLS].values
                if len(data) != seq_len: data = resample(data, seq_len)
                data = scaler.transform(data)
                motions.append(data)
                scores.append(score / 66.0)
            except: continue
        
        self.motions = torch.FloatTensor(np.array(motions))
        self.scores = torch.FloatTensor(np.array(scores)).unsqueeze(1)

    def __len__(self): return len(self.motions)
    def __getitem__(self, idx): return self.motions[idx], self.scores[idx]

# --- 2. MODEL ---
from src.generation.minimalist_models import STAGE_MODELS

# --- 3. LOSS FUNCTION ---

def _pearson(a, b):
    if a.shape[0] < 4 or a.std() < 1e-6 or b.std() < 1e-6:
        return torch.tensor(0.0, device=a.device)
    a_c, b_c = a - a.mean(), b - b.mean()
    return (a_c * b_c).sum() / (torch.norm(a_c) * torch.norm(b_c) + 1e-8)

def dynamics_correlation_loss(recon_x, scores):
    scores_flat = scores.squeeze(-1)
    wrist_vel = recon_x[:, 1:, 6:9] - recon_x[:, :-1, 6:9]
    speed = torch.norm(wrist_vel, dim=-1)
    peak_speed = speed.max(dim=1).values
    
    wrist_acc = wrist_vel[:, 1:] - wrist_vel[:, :-1]
    wrist_jerk = wrist_acc[:, 1:] - wrist_acc[:, :-1]
    jerk_sq = (wrist_jerk ** 2).sum(dim=-1).sum(dim=-1)
    path_len = speed.sum(dim=1) + 1e-6
    T = float(recon_x.shape[1])
    norm_jerk = (T ** 5) / (2 * path_len ** 2) * jerk_sq
    
    return (1.0 - _pearson(peak_speed, scores_flat)) + (1.0 + _pearson(norm_jerk, scores_flat))

def loss_function(recon_x, x, mu, logvar, scores=None, stage=0):
    recon_loss = F.mse_loss(recon_x, x)
    real_vel, recon_vel = x[:, 1:, :] - x[:, :-1, :], recon_x[:, 1:, :] - recon_x[:, :-1, :]
    vel_loss = F.mse_loss(recon_vel, real_vel)
    real_acc, recon_acc = real_vel[:, 1:, :] - real_vel[:, :-1, :], recon_vel[:, 1:, :] - recon_vel[:, :-1, :]
    acc_loss = F.mse_loss(recon_acc, real_acc)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    dyn_corr = dynamics_correlation_loss(recon_x, scores) if scores is not None else torch.tensor(0.0, device=recon_x.device)
    dyn_weight = 2.5 if stage == 3 else 2.0
    
    return (recon_loss + 10.0 * vel_loss + 5.0 * acc_loss + 0.1 * kld + dyn_weight * dyn_corr), recon_loss, vel_loss, kld

# --- 4. TRAINING LOOP ---

def train(max_samples=None, epochs=EPOCHS, stage=0, cond_drop=0.1, data_dir=None):
    print(f"\nTraining Stage {stage} on data: {os.path.basename(data_dir)}")
    scaler_path = os.path.join(MODEL_DIR, "scaler_cutoff_fma.pkl")
    if not os.path.exists(scaler_path):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))[:1000]
        all_data = []
        for f in all_files:
            df = pd.read_csv(f)
            for c in COLS:
                if c not in df.columns: df[c] = 0.0
            all_data.append(df[COLS].values)
        scaler = StandardScaler().fit(np.vstack(all_data))
        joblib.dump(scaler, scaler_path)
    else: scaler = joblib.load(scaler_path)

    dataset = CutoffDataset(data_dir, scaler, max_samples=max_samples)
    train_size = int(0.9 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)
    
    ModelClass = STAGE_MODELS[stage]
    try: model = ModelClass(cond_drop_prob=cond_drop).to(DEVICE)
    except TypeError: model = ModelClass().to(DEVICE)
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    scaler_amp = GradScaler('cuda') if torch.cuda.is_available() else GradScaler()
    
    best_val = float('inf')
    data_tag = "smote" if "smote" in data_dir else ("linear" if "backup" in data_dir or "linear" in data_dir else "dtw")
    model_name = f"cvae_stage{stage}_{data_tag}"
    
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_l = 0
        for x, c in train_loader:
            x, c = x.to(DEVICE, non_blocking=True), c.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, enabled=torch.cuda.is_available()):
                recon, mu, logvar = model(x, c)
                loss, _, _, _ = loss_function(recon, x, mu, logvar, scores=c, stage=stage)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_l += loss.item()
        
        model.eval()
        val_l = 0
        with torch.no_grad():
            for x, c in val_loader:
                x, c = x.to(DEVICE, non_blocking=True), c.to(DEVICE, non_blocking=True)
                with autocast(device_type=DEVICE.type, enabled=torch.cuda.is_available()):
                    recon, mu, logvar = model(x, c)
                    loss, _, _, _ = loss_function(recon, x, mu, logvar, scores=c, stage=stage)
                val_l += loss.item()
        
        avg_train = train_l / len(train_loader)
        avg_val = val_l / len(val_loader)
        history.append({'epoch': epoch, 'train': avg_train, 'val': avg_val})
        scheduler.step(avg_val)
        
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_name}_best.pth"))
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Train {avg_train:.4f}, Val {avg_val:.4f}")
            
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_name}.pth"))
    pd.DataFrame(history).to_csv(os.path.join(MODEL_DIR, f"{model_name}_history.csv"), index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples", type=int, default=15000)
    parser.add_argument("--data-source", type=str, default="dtw", choices=["dtw", "smote", "linear"])
    args = parser.parse_args()
    
    # Root relative to script: ../../../
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if args.data_source == "smote": 
        DATA_DIR = os.path.join(PROJECT_ROOT, "data/kinematic/cutoff/augmented_smote")
    elif args.data_source == "linear":
        DATA_DIR = os.path.join(PROJECT_ROOT, "data/kinematic/cutoff/augmented_backup")
        if not os.path.exists(DATA_DIR): DATA_DIR = os.path.join(PROJECT_ROOT, "data/kinematic/cutoff/augmented_linear")
    else: 
        DATA_DIR = os.path.join(PROJECT_ROOT, "data/kinematic/cutoff/augmented")
        
    train(max_samples=args.samples, epochs=args.epochs, stage=args.stage, data_dir=DATA_DIR)
