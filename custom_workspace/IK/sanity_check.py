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
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIG ---
SEQ_LEN = 100
INPUT_DIM = 12
LATENT_DIM = 32  # Simple bottleneck
BATCH_SIZE = 8
EPOCHS = 100
LR = 0.001

# --- 1. DATASET (HEALTHY ONLY) ---
class SimpleDataset(Dataset):
    def __init__(self, folder):
        self.data = []
        files = glob.glob(os.path.join(folder, "*_processed.csv"))
        
        if len(files) == 0:
            print(f"CRITICAL: No files found in {folder}")
            
        for fpath in files:
            try:
                df = pd.read_csv(fpath)
                data = df.values.astype(np.float32)
                if data.shape == (SEQ_LEN, INPUT_DIM):
                    self.data.append(data)
            except: pass
            
        self.data = np.array(self.data)
        print(f"Loaded {len(self.data)} samples. Shape: {self.data.shape}")
        
        # Simple Global Normalization
        self.mean = np.mean(self.data, axis=(0,1))
        self.std = np.std(self.data, axis=(0,1)) + 1e-6
        
        # Check for Motion
        z_range = self.data[:, :, 8].max() - self.data[:, :, 8].min()
        print(f"Max Vertical Motion (Z-Range): {z_range:.2f} (Should be > 50)")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        # Normalize on the fly
        raw = self.data[idx]
        norm = (raw - self.mean) / self.std
        return torch.FloatTensor(norm)

# --- 2. SIMPLE MODEL (No VAE, No Conditions) ---
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

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 64 # Small and efficient
        
        # Encoder
        self.input_proj = nn.Linear(INPUT_DIM, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, SEQ_LEN)
        enc_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        
        # Bottleneck (Compression)
        self.enc_to_lat = nn.Linear(SEQ_LEN * self.d_model, LATENT_DIM)
        self.lat_to_dec = nn.Linear(LATENT_DIM, SEQ_LEN * self.d_model)
        
        # Decoder
        dec_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=2)
        
        self.output_proj = nn.Linear(self.d_model, INPUT_DIM)

    def forward(self, x):
        # x: (Batch, 100, 12)
        
        # Encode
        x_emb = self.pos_encoder(self.input_proj(x))
        enc_out = self.encoder(x_emb)
        
        # Compress
        flat = enc_out.flatten(1)
        latent = self.enc_to_lat(flat) # The "Brain" representation
        
        # Expand
        expanded = self.lat_to_dec(latent).view(-1, SEQ_LEN, self.d_model)
        
        # Decode
        dec_out = self.pos_encoder(expanded)
        dec_out = self.decoder(dec_out)
        
        return self.output_proj(dec_out)

# --- 3. TRAIN & PLOT ---
def main():
    # Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data/kinematic/Healthy/processed")
    
    dataset = SimpleDataset(data_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleTransformer()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print("\n--- Starting Sanity Check Training ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x in loader:
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {total_loss / len(loader):.5f}")

    print("Training Done. Visualizing one sample...")
    
    # --- VISUALIZATION ---
    model.eval()
    
    # Take one real sample
    real_norm = dataset[0].unsqueeze(0) # (1, 100, 12)
    with torch.no_grad():
        recon_norm = model(real_norm)
    
    # Denormalize
    real = (real_norm.numpy() * dataset.std) + dataset.mean
    recon = (recon_norm.numpy() * dataset.std) + dataset.mean
    
    real = real.squeeze()
    recon = recon.squeeze()
    
    # Plot Z-Axis (Up/Down Motion) - This is the "drinking" part
    plt.figure(figsize=(10, 5))
    plt.plot(real[:, 8], label='Real Wrist Z (Height)', linewidth=2)
    plt.plot(recon[:, 8], label='Reconstructed Wrist Z', linestyle='--')
    plt.title("Did it learn the Drinking Motion?")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()