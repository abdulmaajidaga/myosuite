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
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
SEQ_LEN = 100
PCA_COMPONENTS = 12
INPUT_CHANNELS = 24  # 12 Pos + 12 Vel (No explicit Time channel, used PosEncoding)
CONDITION_DIM = 1
LATENT_DIM = 64
BATCH_SIZE = 32
LR = 0.0001 # Slightly lower LR for Transformer
EPOCHS = 300
BETA = 0.005 
SMOOTH_WEIGHT = 5.0 
AUGMENT_FACTOR = 10 

# --- DATASET ---
class TransformerDataset(Dataset):
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
                
                repeats = 2 if "Stroke" in folder else 1
                for _ in range(repeats):
                    all_raw.append(resampled)
                    meta_data.append(fma)

        # Fit PCA on ORIGINAL data
        all_stacked = np.vstack(all_raw)
        self.pca = PCA(n_components=PCA_COMPONENTS)
        self.pca.fit(all_stacked)
        
        # We reuse the same PCA model names but maybe we should warn or separate? 
        # Overwriting is fine if it's consistent.
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
                
                # 2. Augmentation
                scale_factor = np.random.uniform(0.95, 1.05)
                noise = np.random.normal(0, 0.05, combined.shape)
                
                augmented = (combined * scale_factor) + noise
                
                # Transformer expects (Seq, Channels) -> (100, 24)
                # No Transpose here.
                
                self.data.append({
                    'x': torch.FloatTensor(augmented),
                    'c': torch.FloatTensor([fma / 66.0])
                })

        print(f"Dataset Ready: {len(self.data)} samples (Transformer Format)")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]['x'], self.data[idx]['c']

# --- POSITIONAL ENCODING ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        return x + self.pe[:, :x.size(1)]

# --- TRANSFORMER VAE ---
class TransformerVAE(nn.Module):
    def __init__(self):
        super(TransformerVAE, self).__init__()
        
        self.d_model = 128
        self.nhead = 4
        self.num_layers = 4
        
        # Feature Projector (Input -> d_model)
        # Input is (Batch, 100, 24 + 1(Condition)?) 
        # Actually condition is usually concatenated to latent or input.
        # Let's concatenate condition to input channels: 24 + 1 = 25
        self.input_proj = nn.Linear(INPUT_CHANNELS + CONDITION_DIM, self.d_model)
        
        self.pos_encoder = PositionalEncoding(self.d_model, SEQ_LEN)
        
        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Latent Projections
        # Flatten: (Batch, 100, d_model) -> (Batch, 100*d_model)
        self.fc_mu = nn.Linear(SEQ_LEN * self.d_model, LATENT_DIM)
        self.fc_logvar = nn.Linear(SEQ_LEN * self.d_model, LATENT_DIM)
        
        # Decoder Projections
        self.decoder_input = nn.Linear(LATENT_DIM + CONDITION_DIM, SEQ_LEN * self.d_model)
        
        # Decoder Transformer (Generator)
        # We reuse TransformerEncoder structure as a decoder since we are generating whole sequence at once from Latent
        decoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=self.num_layers)
        
        # Final Output
        self.output_proj = nn.Linear(self.d_model, INPUT_CHANNELS)

    def encode(self, x, c):
        # x: (B, 100, 24)
        # c: (B, 1) -> expand to (B, 100, 1)
        c_expanded = c.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        
        x_in = torch.cat([x, c_expanded], dim=2) # (B, 100, 25)
        x_emb = self.input_proj(x_in) # (B, 100, d_model)
        x_emb = self.pos_encoder(x_emb)
        
        h = self.transformer_encoder(x_emb) # (B, 100, d_model)
        h_flat = h.flatten(1) # (B, 100*d_model)
        
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        # z: (B, 64)
        z_c = torch.cat([z, c], dim=1)
        
        h_flat = self.decoder_input(z_c)
        h = h_flat.view(-1, SEQ_LEN, self.d_model) # (B, 100, d_model)
        
        # Add Positional Encoding to the Latent Projection
        # This tells the transformer which part of the sequence it is generating
        h = self.pos_encoder(h)
        
        out_emb = self.transformer_decoder(h) # (B, 100, d_model)
        return self.output_proj(out_emb) # (B, 100, 24)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # recon_x, x: (B, 100, 24)
    
    # 1. Reconstruction with Boundary Anchoring
    # Weights shape: (1, 100, 1) broadcastable
    weights = torch.ones_like(x)
    weights[:, :10, :] = 50.0 # Start
    weights[:, -10:, :] = 50.0 # End
    
    mse = torch.sum((recon_x - x).pow(2) * weights)
    
    # 2. KLD
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. Smoothness (Acceleration Penalty)
    # Derivative along Time (dim 1)
    vel = recon_x[:, 1:, :] - recon_x[:, :-1, :]
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    
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
    
    dataset = TransformerDataset(stroke, healthy, scores, out_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = TransformerVAE()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Starting Transformer VAE Training on {len(dataset)} samples...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for x, c in loader:
            if torch.cuda.is_available():
                x, c = x.cuda(), c.cuda()
                
            optimizer.zero_grad()
            recon, mu, logvar = model(x, c)
            loss = loss_function(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}: Avg Loss {epoch_loss / len(loader.dataset):.4f}")
            
    # Save Model
    save_path = os.path.join(out_dir, "transformer_vae.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Transformer Model saved to {save_path}")

if __name__ == "__main__":
    train()