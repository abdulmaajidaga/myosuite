import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import random
import copy
import pickle
from scipy.signal import savgol_filter

# --- IMPORTS ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cvae

# --- CONFIG ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REAL_STROKE_JERK = 76000.0
N_TRIALS = 20
EPOCHS_PER_TRIAL = 50
BATCH_SIZE = 32

# Defined Ranges for CVAE
PARAM_GRID = {
    'LATENT_DIM': [32, 64, 128],
    'LR': [1e-3, 5e-4, 1e-4],
    'BETA': [0.001, 0.005, 0.01, 0.1],
    'SMOOTH_WEIGHT': [1.0, 5.0, 10.0], # Not used directly in cvae_loss but kept for consistency
    'W_BOUNDARY': [50.0, 100.0, 200.0] # Not used directly in cvae_loss
}

def load_dataset():
    print("Loading CVAE Dataset...")
    base = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(base)
    stroke = os.path.join(proj, "data", "kinematic", "Stroke")
    healthy = os.path.join(proj, "data", "kinematic", "Healthy")
    scores = os.path.join(base, "output", "scores.csv")
    return cvae.KinematicDataset(stroke, healthy, scores)

def cvae_loss(recon_traj, recon_dur, target, mu, logvar, params):
    target_traj = target[:, :-1]
    target_dur = target[:, -1].view(-1, 1)
    
    mse = nn.MSELoss(reduction='sum')(recon_traj, target_traj)
    dur_loss = nn.MSELoss(reduction='sum')(recon_dur, target_dur)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return mse + (dur_loss * 50.0) + (params['BETA'] * kld)

def evaluate_physics(model):
    model.eval()
    n_samples = 50
    fma = 20.0
    
    drifts = []
    jerks = []
    
    for _ in range(n_samples):
        z = torch.randn(1, model.fc_mu.out_features)
        c = torch.tensor([[fma / 66.0]])
        with torch.no_grad():
            recon, _ = model.decode(z, c)
        traj = recon.detach().numpy().reshape(100, 63) * 1000.0 
            
        drift = np.linalg.norm(traj[0, :3] - traj[-1, :3])
        drifts.append(drift)
        
        w = traj[:, :3]
        vel = np.diff(w, axis=0)
        acc = np.diff(vel, axis=0)
        j = np.diff(acc, axis=0)
        jerk_val = np.mean(np.sum(j**2, axis=1)) * 1000
        jerks.append(jerk_val)
        
    avg_drift = np.mean(drifts)
    avg_jerk = np.mean(jerks)
    jerk_dev = abs(avg_jerk - REAL_STROKE_JERK)
    score = avg_drift + (jerk_dev * 0.0001)
    
    return score, avg_drift, avg_jerk

def run_trial(dataset, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   -> Training on {device}")

    cvae.LATENT_DIM = params['LATENT_DIM']
    model = cvae.CVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['LR'])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model.train()
    for epoch in range(EPOCHS_PER_TRIAL):
        for batch in loader:
            x, c = batch
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            recon_traj, recon_dur, mu, logvar = model(x, c)
            loss = cvae_loss(recon_traj, recon_dur, x, mu, logvar, params)
            loss.backward()
            optimizer.step()
            
    model.cpu()
    score, drift, jerk = evaluate_physics(model)
    return model, score, drift, jerk

def main():
    ds_cvae = load_dataset()
    best_score = float('inf')
    
    print(f"\nStarting Auto-Tuning CVAE ({N_TRIALS} trials)...")
    
    for i in range(N_TRIALS):
        params = {k: random.choice(v) for k, v in PARAM_GRID.items()}
        print(f"Trial {i+1}/{N_TRIALS}: {params}")
        
        try:
            model, score, drift, jerk = run_trial(ds_cvae, params)
            print(f"  -> Score: {score:.4f} | Drift: {drift:.2f} | Jerk: {jerk:.0f}")
            
            if score < best_score:
                best_score = score
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_CVAE.pth"))
                with open(os.path.join(OUTPUT_DIR, "best_CVAE_config.json"), 'w') as f:
                    json.dump(params, f, indent=4)
                print("  -> New Best!")
        except Exception as e:
            print(f"  -> Trial Failed: {e}")

if __name__ == "__main__":
    main()
