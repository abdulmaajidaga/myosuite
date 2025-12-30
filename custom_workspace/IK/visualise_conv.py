import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
import os
import sys
import pickle

# --- CONFIG ---
PCA_COMPONENTS = 12
INPUT_CHANNELS = 24 
CONDITION_DIM = 1
LATENT_DIM = 64

# --- MODEL (Must match) ---
class ConvCVAE(nn.Module):
    def __init__(self):
        super(ConvCVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(3072 + CONDITION_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(3072 + CONDITION_DIM, LATENT_DIM)
        self.decoder_input = nn.Linear(LATENT_DIM + CONDITION_DIM, 3072)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(64, INPUT_CHANNELS, 4, 2, 1)
        )
        self.sizer = nn.AdaptiveAvgPool1d(100)

    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        h = self.decoder_input(z_c).view(-1, 256, 12)
        return self.sizer(self.decoder(h))

def generate(model, fma_score):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM)
        c = torch.tensor([[fma_score / 66.0]])
        recon = model.decode(z, c) # (1, 24, 100)
        return recon.squeeze().T.numpy() # (100, 24)

def animate(traj, fma, pca, scaler):
    # Inverse Coupled Scaling
    # Split Pos/Vel
    scaled_pos = traj[:, :12]
    scaled_vel = traj[:, 12:]
    
    # Inverse Scale Position
    pca_pos = scaler.inverse_transform(scaled_pos)
    
    # Inverse PCA -> 63
    raw_traj = pca.inverse_transform(pca_pos)
    
    # Filter
    smooth_traj = np.zeros_like(raw_traj)
    for i in range(raw_traj.shape[1]):
        smooth_traj[:, i] = savgol_filter(raw_traj[:, i], 11, 3)
        
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Conv1d VAE Generation (FMA {fma})")
    
    frames = smooth_traj.reshape(100, -1)
    wra = frames[:, 0:3]; elb = frames[:, 21:24]; sho = frames[:, 48:51]
    
    all_p = np.vstack([wra, elb, sho])
    ax.set_xlim([all_p[:,0].min(), all_p[:,0].max()])
    ax.set_ylim([all_p[:,1].min(), all_p[:,1].max()])
    ax.set_zlim([all_p[:,2].min(), all_p[:,2].max()])
    
    line, = ax.plot([],[],[], 'b-', lw=4)
    pts, = ax.plot([],[],[], 'ko')
    
    def update(i):
        x = [sho[i,0], elb[i,0], wra[i,0]]
        y = [sho[i,1], elb[i,1], wra[i,1]]
        z = [sho[i,2], elb[i,2], wra[i,2]]
        line.set_data_3d(x, y, z)
        pts.set_data_3d(x, y, z)
        return line, pts
        
    ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "output/conv_cvae.pth")
    pca_path = os.path.join(base, "output/pca_model.pkl")
    scaler_path = os.path.join(base, "output/scaler.pkl")
    
    if not os.path.exists(model_path):
        print("Model not found.")
    else:
        model = ConvCVAE()
        model.load_state_dict(torch.load(model_path))
        with open(pca_path, 'rb') as f: pca = pickle.load(f)
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
            
        t = generate(model, 66)
        animate(t, 66, pca, scaler)
