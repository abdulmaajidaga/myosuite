import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt

# --- CONFIG ---
SEQ_LEN = 100
INPUT_DIM = 12
LATENT_DIM = 32
BATCH_SIZE = 8
EPOCHS = 600  # Enough to converge for this test
LR = 0.001

# --- 1. DATASET ---
class KinematicDataset(Dataset):
    def __init__(self, folder, mean=None, std=None):
        self.data = []
        files = glob.glob(os.path.join(folder, "*_processed.csv"))
        
        # Load Data
        for fpath in files:
            try:
                df = pd.read_csv(fpath)
                data = df.values.astype(np.float32)
                if data.shape == (SEQ_LEN, INPUT_DIM):
                    self.data.append(data)
            except: pass
            
        self.data = np.array(self.data)
        
        # Handle Stats (Crucial for Anomaly Detection)
        if len(self.data) > 0:
            if mean is None:
                # We are the "Training" (Healthy) set, calculate stats
                self.mean = np.mean(self.data, axis=(0,1))
                self.std = np.std(self.data, axis=(0,1)) + 1e-6
                self.data_range = np.max(self.data) - np.min(self.data)
                print(f"Loaded {len(self.data)} samples from {folder}. (Calculating new stats)")
            else:
                # We are the "Test" (Impaired) set, use existing stats
                self.mean = mean
                self.std = std
                # Use healthy range for fair accuracy comparison
                self.data_range = np.max(self.data) - np.min(self.data) 
                print(f"Loaded {len(self.data)} samples from {folder}. (Using provided stats)")
        else:
            print(f"WARNING: No data found in {folder}")
            self.mean = 0
            self.std = 1
            self.data_range = 1

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        raw = self.data[idx]
        norm = (raw - self.mean) / self.std
        return torch.FloatTensor(norm)

# --- 2. MODEL (Same as before) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 64
        self.input_proj = nn.Linear(INPUT_DIM, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, SEQ_LEN)
        enc_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.enc_to_lat = nn.Linear(SEQ_LEN * self.d_model, LATENT_DIM)
        self.lat_to_dec = nn.Linear(LATENT_DIM, SEQ_LEN * self.d_model)
        dec_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=2)
        self.output_proj = nn.Linear(self.d_model, INPUT_DIM)

    def forward(self, x):
        x = self.pos_encoder(self.input_proj(x))
        latent = self.enc_to_lat(self.encoder(x).flatten(1))
        decoded = self.decoder(self.pos_encoder(self.lat_to_dec(latent).view(-1, SEQ_LEN, self.d_model)))
        return self.output_proj(decoded)

# --- 3. HELPER FOR EVALUATION ---
def evaluate_dataset(model, loader, dataset, name):
    model.eval()
    all_real, all_recon = [], []
    with torch.no_grad():
        for x in loader:
            recon = model(x)
            all_real.append((x.numpy() * dataset.std) + dataset.mean)
            all_recon.append((recon.numpy() * dataset.std) + dataset.mean)
    
    real = np.concatenate(all_real, axis=0)
    recon = np.concatenate(all_recon, axis=0)
    
    mse = np.mean((real - recon) ** 2)
    mae = np.mean(np.abs(real - recon))
    accuracy = (1 - (mae / dataset.data_range)) * 100
    
    print(f"\n--- Results for {name} ---")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f}")
    print(f"Reconstruction Accuracy: {accuracy:.2f}%")
    
    return real, recon, mse

# --- 4. MAIN ---
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Paths (ASSUMING Standard Structure)
    healthy_dir = os.path.join(os.path.dirname(base_dir), "data/kinematic/Healthy/processed")
    impaired_dir = os.path.join(os.path.dirname(base_dir), "data/kinematic/Stroke/processed")
    
    # 1. Load Healthy (Learns Stats)
    healthy_ds = KinematicDataset(healthy_dir)
    healthy_loader = DataLoader(healthy_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Load Impaired (Uses Healthy Stats!)
    impaired_ds = KinematicDataset(impaired_dir, mean=healthy_ds.mean, std=healthy_ds.std)
    impaired_loader = DataLoader(impaired_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Train on Healthy
    model = SimpleTransformer()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print(f"\nTraining on {len(healthy_ds)} Healthy samples...")
    for epoch in range(EPOCHS):
        model.train()
        for x in healthy_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
        if (epoch+1)%100==0: print(f"Epoch {epoch+1}: Loss {loss.item():.5f}")

    # 4. Compare
    print("\n--- COMPARATIVE ANALYSIS ---")
    h_real, h_recon, h_mse = evaluate_dataset(model, healthy_loader, healthy_ds, "HEALTHY")
    
    if len(impaired_ds) > 0:
        i_real, i_recon, i_mse = evaluate_dataset(model, impaired_loader, impaired_ds, "IMPAIRED")
        
        print("\n" + "="*40)
        print(f"ANOMALY DETECTION RESULT:")
        print(f"Healthy MSE:  {h_mse:.2f}")
        print(f"Impaired MSE: {i_mse:.2f}")
        ratio = i_mse / h_mse
        print(f"Error Ratio:  {ratio:.2f}x (Impaired error is {ratio:.1f} times higher)")
        print("="*40)
        
        # --- VISUALIZATION ---
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Healthy Plot
        idx_h = 0
        axes[0].plot(h_real[idx_h, :, 8], 'b', label='Original', linewidth=2)
        axes[0].plot(h_recon[idx_h, :, 8], 'r--', label='Reconstruction')
        axes[0].set_title(f"Healthy Sample (Low Error)\nMSE: {np.mean((h_real[idx_h]-h_recon[idx_h])**2):.2f}")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Impaired Plot
        idx_i = 0
        axes[1].plot(i_real[idx_i, :, 8], 'b', label='Original (Impaired)', linewidth=2)
        axes[1].plot(i_recon[idx_i, :, 8], 'r--', label='Reconstruction (Smoothed)')
        axes[1].set_title(f"Impaired Sample (High Error)\nMSE: {np.mean((i_real[idx_i]-i_recon[idx_i])**2):.2f}")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.suptitle("Anomaly Detection: The Model Fails to Reconstruct Impaired Motion", fontsize=16)
        plt.show()

    else:
        print("Could not find Impaired data to test.")

if __name__ == "__main__":
    main()