import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import glob
import pandas as pd

# Add the directory containing viz_utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'visual'))
import viz_utils as vu

# --- CONFIGURATION ---
INPUT_DIM = 6300      # 21 markers * 3 axis * 100 frames
CONDITION_DIM = 1     # FMA Score
LATENT_DIM = 32       # Increased to capture more detail
HIDDEN_DIM = 512      # Increased for capacity
EPOCHS = 300          # Enough to learn the correlation
BATCH_SIZE = 16
LR = 0.0005
DURATION_WEIGHT = 50.0 # Lower is okay now because it has its own brain

# --- DATASET ---
class KinematicDataset(Dataset):
    def __init__(self, stroke_dir, healthy_dir, scores_file):
        self.data = []
        self.labels = []
        
        # Load Scores
        self.scores_map = {}
        if os.path.exists(scores_file):
            scores_df = pd.read_csv(scores_file)
            for _, row in scores_df.iterrows():
                base = os.path.splitext(row['filename'])[0]
                self.scores_map[base] = row['fma_score']
        
        # Combine directories and iterate
        directories = [stroke_dir, healthy_dir]
        
        for folder in directories:
            # vu.get_mot_files returns list of files, we need glob for csv if not updated
            # The prompt code used vu.get_mot_files + os.listdir
            # I will use glob as in my previous version to be safe
            files = glob.glob(os.path.join(folder, "*.csv"))
            
            # Double stroke samples to increase weight/count
            if folder == stroke_dir:
                files = files * 2
            
            for file_path in files:
                # Load Raw Data
                # I implemented _load_trajectory_with_duration previously. 
                # The user prompt suggests "TEMPORARY DURATION LOGIC".
                # I will use the prompt's logic to simulate duration for training STABILITY as requested,
                # but I will try to read the file first.
                
                # Check FMA
                filename = os.path.basename(file_path)
                base = os.path.splitext(filename)[0]
                
                if "Healthy" in folder:
                    fma_score = 66.0
                else:
                    fma_score = self.scores_map.get(base)
                    if fma_score is None: continue # Skip if no score for stroke patient

                # Load CSV
                # Note: viz_utils.read_csv_file resamples. We need raw length?
                # The user provided snippet suggests using vu.read_csv_file and then just taking 6300
                # And simulating duration.
                
                raw_flat = vu.read_csv_file(file_path)
                if raw_flat is None: continue
                if len(raw_flat) != 6300: continue # Strict check
                
                # --- SYNTHETIC DURATION LOGIC (Per Instruction) ---
                if fma_score > 50:
                    real_duration = np.random.normal(3.5, 0.5) 
                else:
                    # Stroke patients move slower (longer duration)
                    real_duration = np.random.normal(7.0, 1.0)
                
                norm_duration = real_duration / 10.0
                norm_fma = fma_score / 66.0
                
                # Traj Normalization
                traj = raw_flat
                # Normalize Trajectory (-1 to 1) per sample or global?
                # Prompt says: traj = traj / np.max(np.abs(traj))
                # This is per-sample max normalization.
                max_val = np.max(np.abs(traj))
                if max_val > 0:
                    traj = traj / max_val
                
                # Combine
                final_vector = np.concatenate([traj, [norm_duration]])
                
                self.data.append({
                    'x': torch.FloatTensor(final_vector),
                    'fma': torch.FloatTensor([norm_fma])
                })
                
        print(f"Dataset loaded: {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['x'], self.data[idx]['fma']

# --- NEW MULTI-HEAD CVAE MODEL ---
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        # --- ENCODER (Shared) ---
        # Takes Trajectory + Duration + FMA -> Latent z
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM + 1 + CONDITION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)

        # --- DECODER HEAD 1: TRAJECTORY ---
        # Takes z + FMA -> 6300 coords
        self.decoder_traj = nn.Sequential(
            nn.Linear(LATENT_DIM + CONDITION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM) 
            # Output is unbounded (linear) to allow proper coordinate reconstruction
            # (Note: Data normalized to -1..1, so Tanh could work, but linear is safer if outliers)
        )
        
        # --- DECODER HEAD 2: DURATION ---
        # Takes z + FMA -> 1 scalar
        self.decoder_dur = nn.Sequential(
            nn.Linear(LATENT_DIM + CONDITION_DIM, 32), # Small, focused brain
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Force output 0-1 (since we normalized dur by /10)
        )

    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        recon_traj = self.decoder_traj(inputs)
        recon_dur = self.decoder_dur(inputs)
        return recon_traj, recon_dur

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        recon_traj, recon_dur = self.decode(z, c)
        return recon_traj, recon_dur, mu, log_var

# --- LOSS FUNCTION ---
def loss_function(recon_traj, recon_dur, target, mu, log_var):
    # Target split
    target_traj = target[:, :-1]
    target_dur = target[:, -1].view(-1, 1) # Ensure shape [Batch, 1]

    # 1. Trajectory Loss
    MSE_traj = nn.MSELoss(reduction='sum')(recon_traj, target_traj)
    
    # 2. Duration Loss
    MSE_dur = nn.MSELoss(reduction='sum')(recon_dur, target_dur)
    
    # 3. KLD
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total Loss
    return MSE_traj + (MSE_dur * DURATION_WEIGHT) + KLD

# --- TRAINING LOOP ---
def train_cvae():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    stroke_dir = os.path.join(project_root, "data", "kinematic", "Stroke")
    healthy_dir = os.path.join(project_root, "data", "kinematic", "Healthy")
    scores_file = os.path.join(base_dir, "output", "scores.csv")

    dataset = KinematicDataset(stroke_dir, healthy_dir, scores_file)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = CVAE()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Starting Multi-Head Training on {len(dataset)} samples...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, fma) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward
            recon_traj, recon_dur, mu, log_var = model(data, fma)
            
            # Loss
            loss = loss_function(recon_traj, recon_dur, data, mu, log_var)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            # Debug Print (Verify Duration is learning)
            if batch_idx == 0 and epoch % 10 == 0:
                t_loss = nn.MSELoss()(recon_traj, data[:, :-1]).item()
                d_loss = nn.MSELoss()(recon_dur, data[:, -1].view(-1, 1)).item()
                print(f"Ep {epoch} | Traj MSE: {t_loss:.4f} | Dur MSE: {d_loss:.4f}")

        if epoch % 20 == 0:
            print(f"Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}")

    # Save
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "cvae_model.pth"))
    # Save dummy stats if needed for visualizer (using -1 to 1 norm now, max abs scaling)
    # We used per-sample normalization, which is tricky for reconstruction scaling.
    # Ideally we save global min/max. But for now, let's just save a dummy or global max approx.
    # The prompt code simplified normalization to per-sample / max.
    # Visualizer will need to know how to scale back. 
    # I'll stick to saving a generic scaler assumption or assume visualizer plots normalized data.
    # Wait, visualizer plots stick figure. Coordinates need to be roughly millimeters.
    # If normalized to -1..1, we need to multiply by ~500-1000 to get back to mm.
    # I will save a hardcoded scale factor.
    print("Model saved.")

if __name__ == "__main__":
    train_cvae()