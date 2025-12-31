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
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), 'visual'))
import viz_utils as vu

# --- CONFIGURATION ---
SEQ_LEN = 100
PCA_COMPONENTS = 12
INPUT_CHANNELS = 25 # Updated to 25 (12 Pos + 12 Vel + 1 Time)
CONDITION_DIM = 1
LATENT_DIM = 64
BATCH_SIZE = 32
LR = 0.0005
EPOCHS = 1000
BETA = 0.005 
SMOOTH_WEIGHT = 5.0 
AUGMENT_FACTOR = 10 

# --- DATASET ---
class ConvDataset(Dataset):
    def __init__(self, stroke_dir, healthy_dir, scores_file, output_dir):
        self.data = []
        
        self.scores_map = {}
        if os.path.exists(scores_file):
            scores_df = pd.read_csv(scores_file)
            for _, row in scores_df.iterrows():
                base = os.path.splitext(row['filename'])[0]
                self.scores_map[base] = row['fma_score']
        
        all_raw = []
        meta_data = [] 
        
        print("Loading raw data...")
        for folder in [stroke_dir, healthy_dir]:
            files = glob.glob(os.path.join(folder, "*.csv"))
            for file_path in files:
                filename = os.path.basename(file_path)
                base = os.path.splitext(filename)[0]
                
                if "Healthy" in folder: fma = 66.0
                else:
                    fma = self.scores_map.get(base)
                    if fma is None: continue
                
                raw_df = pd.read_csv(file_path, skiprows=2, header=None)
                if raw_df.shape[1] < 63: continue
                raw_data = raw_df.iloc[:, :63].ffill().bfill().fillna(0).values
                
                resampled = signal.resample(raw_data, SEQ_LEN)
                
                # Base dataset (balancing included)
                repeats = 2 if "Stroke" in folder else 1
                for _ in range(repeats):
                    all_raw.append(resampled)
                    meta_data.append(fma)

        # Fit PCA on ORIGINAL data
        all_stacked = np.vstack(all_raw)
        self.pca = PCA(n_components=PCA_COMPONENTS)
        self.pca.fit(all_stacked)
        
        with open(os.path.join(output_dir, "pca_model.pkl"), 'wb') as f:
            pickle.dump(self.pca, f)
            
        # Transform base data
        print("Transforming and Augmenting...")
        base_pca_samples = []
        for traj in all_raw:
            p = self.pca.transform(traj)
            v = np.diff(p, axis=0, prepend=p[0].reshape(1, -1))
            base_pca_samples.append((p, v))
            
        # Fit Scaler on Position
        all_pos = np.vstack([x[0] for x in base_pca_samples])
        self.scaler = StandardScaler()
        self.scaler.fit(all_pos)
        
        with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Augment loop
        for i, (p, v) in enumerate(base_pca_samples):
            fma = meta_data[i]
            
            for _ in range(AUGMENT_FACTOR):
                # 1. Scale/Couple logic
                p_scaled = self.scaler.transform(p)
                v_scaled = v / self.scaler.scale_
                
                combined = np.concatenate([p_scaled, v_scaled], axis=1) # (100, 24)
                
                # 2. Augmentation (on standardized data)
                scale_factor = np.random.uniform(0.95, 1.05)
                noise = np.random.normal(0, 0.05, combined.shape)
                
                augmented_motion = (combined * scale_factor) + noise
                
                # 3. Add Time Channel (0 to 1)
                # This acts as a positional encoding (Clock Signal)
                time_seq = np.linspace(0, 1, SEQ_LEN).reshape(-1, 1) # (100, 1)
                augmented = np.concatenate([augmented_motion, time_seq], axis=1) # (100, 25)
                
                # Transpose for Conv1d: (25, 100)
                aug_t = augmented.T
                
                self.data.append({
                    'x': torch.FloatTensor(aug_t),
                    'c': torch.FloatTensor([fma / 66.0])
                })

        print(f"Dataset Ready: {len(self.data)} samples (Augmented with Time Channel)")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]['x'], self.data[idx]['c']

# --- CONV CVAE ---
class ConvCVAE(nn.Module):
    def __init__(self):
        super(ConvCVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 64, 4, 2, 1), # 50
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 1), # 25
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, 2, 1), # 12
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent
        flat_dim = 256 * 12 # 3072
        self.fc_mu = nn.Linear(flat_dim + CONDITION_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(flat_dim + CONDITION_DIM, LATENT_DIM)
        
        # Decoder Input
        self.decoder_input = nn.Linear(LATENT_DIM + CONDITION_DIM, flat_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1), # 24
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), # 48
            nn.ReLU(),
            nn.ConvTranspose1d(64, INPUT_CHANNELS, 4, 2, 1) # 96 -> Interpolated to 100
        )
        self.sizer = nn.AdaptiveAvgPool1d(100)

    def encode(self, x, c):
        h = self.encoder(x)
        h = torch.cat([h, c], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        h = self.decoder_input(z_c)
        h = h.view(-1, 256, 12) 
        h = self.decoder(h)
        return self.sizer(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # 1. Reconstruction (Weighted for Boundary Anchoring)
    # Start (0-10) and End (90-100) weighted 50x
    weights = torch.ones_like(x)
    weights[:, :, :10] = 50.0
    weights[:, :, -10:] = 50.0
    
    mse = torch.sum((recon_x - x).pow(2) * weights)
    
    # 2. KLD
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. Smoothness (Acceleration Penalty)
    vel = recon_x[:, :, 1:] - recon_x[:, :, :-1]
    acc = vel[:, :, 1:] - vel[:, :, :-1]
    
    smooth_loss = torch.sum(acc.pow(2))
    
    return mse + (BETA * kld) + (SMOOTH_WEIGHT * smooth_loss)

def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(base_dir)
    stroke = os.path.join(proj_dir, "data/kinematic/Stroke")
    healthy = os.path.join(proj_dir, "data/kinematic/Healthy")
    scores = os.path.join(base_dir, "output/scores.csv")
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = ConvDataset(stroke, healthy, scores, out_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = ConvCVAE()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Starting Augmented Conv1d Training (w/ Time Channel) on {len(dataset)} samples...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for x, c in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(x, c)
            loss = loss_function(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}: Avg Loss {epoch_loss / len(loader.dataset):.4f}")
            
    torch.save(model.state_dict(), os.path.join(out_dir, "conv_cvae.pth"))
    print("Model saved.")

if __name__ == "__main__":
    train()