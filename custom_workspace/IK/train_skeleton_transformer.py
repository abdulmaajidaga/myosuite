import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import glob
import pandas as pd
import pickle
import math

# --- CONFIG ---
INPUT_DIM = 12
SEQ_LEN = 100
LATENT_DIM = 64
BATCH_SIZE = 32
LR = 0.0005
EPOCHS = 600

# --- RELAXED WEIGHTS (To fix "No Motion") ---
W_BOUNDARY = 20.0     # Down from 50.0 (Stop locking the arm)
SMOOTH_WEIGHT = 0.01  # Keep low
VEL_WEIGHT = 0.5      # Down from 10.0 (Let it move!)
MAX_KLD_WEIGHT = 0.005 # Final target for KLD

# --- DATASET (Balanced) ---
class SkeletonDataset(Dataset):
    def __init__(self, healthy_dir, stroke_dir, scores_file, output_dir):
        self.data = []
        self.scores_map = {}
        
        if os.path.exists(scores_file):
            scores_df = pd.read_csv(scores_file)
            for _, row in scores_df.iterrows():
                base = os.path.splitext(row['filename'])[0]
                self.scores_map[base] = row['fma_score']

        healthy_raw = []
        stroke_raw = []

        print("Loading processed skeleton data...")
        for folder in [healthy_dir, stroke_dir]:
            if not os.path.exists(folder): continue
            files = glob.glob(os.path.join(folder, "*_processed.csv"))
            
            for fpath in files:
                try:
                    df = pd.read_csv(fpath)
                    data = df.values.astype(np.float32)
                    if data.shape != (SEQ_LEN, INPUT_DIM): continue
                    
                    fname = os.path.basename(fpath).replace("_processed.csv", "")
                    if "Healthy" in folder: 
                        healthy_raw.append(data)
                    else: 
                        fma = self.scores_map.get(fname, 20.0)
                        stroke_raw.append((data, fma))
                except: continue

        # Balance Logic
        n_h = len(healthy_raw)
        n_s = len(stroke_raw)
        if n_h == 0 or n_s == 0:
            print("Error: Missing data class.")
            sys.exit(1)

        target_count = max(n_h, n_s)
        
        # Add Healthy
        h_repeats = int(target_count / n_h)
        all_data = []
        fma_list = []
        
        for data in healthy_raw:
            for _ in range(h_repeats):
                all_data.append(data)
                fma_list.append(66.0)
                
        # Add Stroke
        s_repeats = int(target_count / n_s)
        for data, fma in stroke_raw:
            for _ in range(s_repeats):
                all_data.append(data)
                fma_list.append(fma)

        # Fit Scaler
        all_stacked = np.vstack(all_data)
        self.mean = np.mean(all_stacked, axis=0)
        self.std = np.std(all_stacked, axis=0) + 1e-6
        
        # Check if there is actual movement in Z axis (drinking)
        z_range = np.max(all_stacked[:, 8]) - np.min(all_stacked[:, 8])
        print(f"Data Z-Range (Vertical Motion): {z_range:.2f}mm") 
        if z_range < 50:
            print("WARNING: Data seems very flat. Check preprocessing!")

        with open(os.path.join(output_dir, "skeleton_scaler.pkl"), 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)

        # Process
        for i, traj in enumerate(all_data):
            norm_traj = (traj - self.mean) / self.std
            fma = fma_list[i]
            aug_traj = norm_traj + np.random.normal(0, 0.01, norm_traj.shape)
            
            self.data.append({
                'x': torch.FloatTensor(aug_traj),
                'c': torch.FloatTensor([fma / 66.0])
            })
            
        print(f"Dataset Ready: {len(self.data)} samples.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]['x'], self.data[idx]['c']

# --- MODEL (Residual Conditioning) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        
        self.input_proj = nn.Linear(INPUT_DIM + 1, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, SEQ_LEN)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.fc_mu = nn.Linear(SEQ_LEN * self.d_model, LATENT_DIM)
        self.fc_logvar = nn.Linear(SEQ_LEN * self.d_model, LATENT_DIM)
        
        self.decoder_input = nn.Linear(LATENT_DIM + 1, SEQ_LEN * self.d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=4)
        
        self.output_proj = nn.Linear(self.d_model, INPUT_DIM)
        
        # NEW: Residual Bias for Condition (Forces FMA to affect output)
        self.condition_bias = nn.Linear(1, INPUT_DIM)

    def encode(self, x, c):
        c_expanded = c.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        x_in = torch.cat([x, c_expanded], dim=2)
        x_emb = self.input_proj(x_in)
        x_emb = self.pos_encoder(x_emb)
        h = self.transformer_encoder(x_emb)
        h_flat = h.flatten(1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        h_flat = self.decoder_input(z_c)
        h = h_flat.view(-1, SEQ_LEN, self.d_model)
        h = self.pos_encoder(h)
        out_emb = self.transformer_decoder(h)
        
        # Standard decode
        out = self.output_proj(out_emb)
        
        # NEW: Add condition bias to every frame
        # This makes sure "Stroke" condition physically shifts the coordinates
        c_bias = self.condition_bias(c).unsqueeze(1) # (Batch, 1, 12)
        
        return out + c_bias

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

# --- LOSS (With Annealing Support) ---
def loss_function(recon, x, mu, logvar, current_kld_weight):
    # Boundary (Relaxed)
    weights = torch.ones_like(x)
    weights[:, :5, :] = W_BOUNDARY
    weights[:, -5:, :] = W_BOUNDARY
    mse = torch.sum((recon - x).pow(2) * weights) / x.shape[0]

    # Velocity (Relaxed)
    real_vel = x[:, 1:, :] - x[:, :-1, :]
    recon_vel = recon[:, 1:, :] - recon[:, :-1, :]
    vel_loss = nn.SmoothL1Loss(reduction='sum')(recon_vel, real_vel) / x.shape[0]

    # KLD
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

    # Smoothness
    acc = recon[:, 2:, :] - 2 * recon[:, 1:-1, :] + recon[:, :-2, :]
    smooth_loss = torch.sum(acc.pow(2)) / x.shape[0]

    return mse + (VEL_WEIGHT * vel_loss) + (current_kld_weight * kld) + (SMOOTH_WEIGHT * smooth_loss)

# --- TRAIN ---
def train():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(base)
    h_dir = os.path.join(data_dir, "data/kinematic/Healthy/processed")
    s_dir = os.path.join(data_dir, "data/kinematic/Stroke/processed")
    scores = os.path.join(base, "output/scores.csv")
    out_dir = os.path.join(base, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = SkeletonDataset(h_dir, s_dir, scores, out_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = TransformerVAE()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5, verbose=True)
    
    print(f"Training on {device}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Cyclical Annealing: Ramp KLD up from 0 to MAX over first 100 epochs
        kld_w = min(MAX_KLD_WEIGHT, (epoch / 100.0) * MAX_KLD_WEIGHT)
        
        for x, c in loader:
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x, c)
            
            loss = loss_function(recon, x, mu, logvar, kld_w)
            
            if torch.isnan(loss): continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}: Loss {avg_loss:.4f} | KLD_W: {kld_w:.5f}")
            
        # Save every 50 epochs just in case
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, "transformer_skeleton.pth"))

    print("Training Complete.")

if __name__ == "__main__":
    train()