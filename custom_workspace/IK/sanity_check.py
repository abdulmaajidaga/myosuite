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
LATENT_DIM = 32
BATCH_SIZE = 8
EPOCHS = 1000  # Reduced for testing, increase for better results
LR = 0.001

# --- 1. DATASET ---
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
        if len(self.data) > 0:
            print(f"Loaded {len(self.data)} samples. Shape: {self.data.shape}")
            
            # Global Normalization
            self.mean = np.mean(self.data, axis=(0,1))
            self.std = np.std(self.data, axis=(0,1)) + 1e-6
            
            # Check for Motion
            z_range = self.data[:, :, 8].max() - self.data[:, :, 8].min()
            print(f"Max Vertical Motion (Z-Range): {z_range:.2f}")
            self.data_range = np.max(self.data) - np.min(self.data)
        else:
            print("No data loaded.")
            self.mean = 0
            self.std = 1
            self.data_range = 1

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        raw = self.data[idx]
        norm = (raw - self.mean) / self.std
        return torch.FloatTensor(norm)

# --- 2. MODEL ---
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
        self.d_model = 64
        
        # Encoder
        self.input_proj = nn.Linear(INPUT_DIM, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, SEQ_LEN)
        enc_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        
        # Bottleneck
        self.enc_to_lat = nn.Linear(SEQ_LEN * self.d_model, LATENT_DIM)
        self.lat_to_dec = nn.Linear(LATENT_DIM, SEQ_LEN * self.d_model)
        
        # Decoder
        dec_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=2)
        
        self.output_proj = nn.Linear(self.d_model, INPUT_DIM)

    def forward(self, x):
        x_emb = self.pos_encoder(self.input_proj(x))
        enc_out = self.encoder(x_emb)
        
        flat = enc_out.flatten(1)
        latent = self.enc_to_lat(flat)
        
        expanded = self.lat_to_dec(latent).view(-1, SEQ_LEN, self.d_model)
        
        dec_out = self.pos_encoder(expanded)
        dec_out = self.decoder(dec_out)
        
        return self.output_proj(dec_out)

# --- 3. MAIN (Training & Evaluation) ---
def main():
    # Setup Paths
    # NOTE: Update this path to where your actual CSV files are located
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_dir = os.path.join(os.path.dirname(base_dir), "data/kinematic/Healthy/processed")
    
    # Initialize Dataset
    dataset = SimpleDataset(data_dir)
    if len(dataset) == 0:
        return # Stop if no data
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    model = SimpleTransformer()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # --- TRAINING LOOP ---
    print("\n--- Starting Training ---")
    loss_history = []
    
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
            
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # --- EVALUATION & ERROR CHECKING ---
    print("\n--- Starting Evaluation & Error Checking ---")
    model.eval()
    
    all_real = []
    all_recon = []
    
    # 1. Collect all data
    with torch.no_grad():
        for x in loader:
            recon = model(x)
            
            # Denormalize batch
            x_np = (x.numpy() * dataset.std) + dataset.mean
            recon_np = (recon.numpy() * dataset.std) + dataset.mean
            
            all_real.append(x_np)
            all_recon.append(recon_np)
            
    # Concatenate all batches
    all_real = np.concatenate(all_real, axis=0)      # (N_Samples, 100, 12)
    all_recon = np.concatenate(all_recon, axis=0)    # (N_Samples, 100, 12)
    
    # 2. Calculate Global Errors
    # MSE: Mean Squared Error
    mse = np.mean((all_real - all_recon) ** 2)
    
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(all_real - all_recon))
    
    # Custom Accuracy Metric: (1 - MAE / Range) * 100
    # This gives a % score of how "close" the signal is relative to the data range
    accuracy = (1 - (mae / dataset.data_range)) * 100
    
    print("-" * 30)
    print(f"Global Results on Healthy Data ({len(all_real)} samples)")
    print("-" * 30)
    print(f"Mean Squared Error (MSE):   {mse:.5f}")
    print(f"Mean Absolute Error (MAE):  {mae:.5f}")
    print(f"Reconstruction Accuracy:    {accuracy:.2f}%")
    print("-" * 30)

    # --- VISUALIZATION ---
    # Pick the first sample from the evaluation set to visualize
    sample_idx = 0
    real_sample = all_real[sample_idx]
    recon_sample = all_recon[sample_idx]
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(14, 8))
    
    # Plot 1: 2D Signal Comparison (Z-Axis / Height)
    ax1 = fig.add_subplot(2, 2, 1)
    # Column 8 is usually Z (Height)
    ax1.plot(real_sample[:, 8], 'b-', label='Original', linewidth=2)
    ax1.plot(recon_sample[:, 8], 'r--', label='Generated', linewidth=2)
    ax1.set_title(f"2D Signal Reconstruction (Z-Axis)\nMAE: {np.mean(np.abs(real_sample[:,8]-recon_sample[:,8])):.4f}")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error over Time (Absolute Difference)
    ax2 = fig.add_subplot(2, 2, 2)
    error_signal = np.abs(real_sample[:, 8] - recon_sample[:, 8])
    ax2.fill_between(range(SEQ_LEN), error_signal, color='orange', alpha=0.5, label='Error Magnitude')
    ax2.set_title("Reconstruction Error (Z-Axis)")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Absolute Error")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 3D Trajectory (Assuming Cols 6,7,8 are X,Y,Z wrist position)
    ax3 = fig.add_subplot(2, 2, (3, 4), projection='3d')
    
    # Real path
    x_r, y_r, z_r = real_sample[:, 6], real_sample[:, 7], real_sample[:, 8]
    ax3.plot(x_r, y_r, z_r, 'b-', label='Original Path', linewidth=2)
    # Start/End points
    ax3.scatter(x_r[0], y_r[0], z_r[0], c='green', marker='o', s=50, label='Start')
    ax3.scatter(x_r[-1], y_r[-1], z_r[-1], c='black', marker='x', s=50, label='End')
    
    # Recon path
    x_g, y_g, z_g = recon_sample[:, 6], recon_sample[:, 7], recon_sample[:, 8]
    ax3.plot(x_g, y_g, z_g, 'r--', label='Generated Path', linewidth=1.5)
    
    ax3.set_title("3D Wrist Trajectory Comparison")
    ax3.set_xlabel("X Position")
    ax3.set_ylabel("Y Position")
    ax3.set_zlabel("Z Position")
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()