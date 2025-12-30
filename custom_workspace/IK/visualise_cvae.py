import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import sys

# Redefine CVAE class to match training script (needed for loading state_dict)
class CVAE(nn.Module):
    def __init__(self, input_dim=6301, cond_dim=1, latent_dim=32):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Tanh()
        )

    def encode(self, x, c):
        x_c = torch.cat([x, c], dim=1)
        h = self.encoder(x_c)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        return self.decoder(z_c)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

def load_model(model_path, stats_path):
    # Load Stats
    stats = np.load(stats_path)
    data_min = stats['data_min']
    data_max = stats['data_max']
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0
    
    # Load Model
    model = CVAE(input_dim=6301, cond_dim=1, latent_dim=32)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, data_min, data_range

def generate_sample(model, fma_score, data_min, data_range):
    with torch.no_grad():
        # Condition
        cond = torch.tensor([[fma_score / 66.0]], dtype=torch.float32)
        # Latent
        z = torch.randn(1, 32)
        # Decode
        out_norm = model.decode(z, cond).numpy()
        # Denormalize: [-1, 1] -> [0, 1] -> Original
        zero_one = (out_norm + 1) / 2.0
        out_raw = zero_one * data_range + data_min
        
        # Split Trajectory and Duration
        # out_raw is (1, 6301)
        full_vec = out_raw.flatten()
        traj = full_vec[:-1] # 6300
        dur_norm = full_vec[-1] # 1
        
        # Denormalize Duration (It was divided by 10.0)
        duration = dur_norm * 10.0
        
    return traj, duration

def animate_stick_figure(traj_flat, duration, fma_score):
    # Reshape: (100 frames, 21 markers * 3 coords)
    # The columns are [X1, Y1, Z1, X2, Y2, Z2, ...]
    frames = traj_flat.reshape(100, -1)
    
    # Extract specific markers
    # Indices: WRA (0-2), ELB_L (21-23), SA_3 (48-50)
    wra = frames[:, 0:3]
    elb = frames[:, 21:24]
    sho = frames[:, 48:51]
    
    # Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Synthetic Motion (FMA {fma_score}) | Duration: {duration:.2f}s")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Fix limits
    all_pts = np.vstack([wra, elb, sho])
    min_limit = np.min(all_pts) - 50
    max_limit = np.max(all_pts) + 50
    ax.set_xlim([min_limit, max_limit])
    ax.set_ylim([min_limit, max_limit])
    ax.set_zlim([min_limit, max_limit])
    
    # Lines
    line_upper, = ax.plot([], [], [], 'b-', linewidth=4, label='Upper Arm')
    line_fore, = ax.plot([], [], [], 'r-', linewidth=4, label='Forearm')
    
    # Points
    pt_sho, = ax.plot([], [], [], 'ko', markersize=8)
    pt_elb, = ax.plot([], [], [], 'ko', markersize=6)
    pt_wra, = ax.plot([], [], [], 'go', markersize=8)
    
    ax.legend()

    # Calculate interval to match predicted duration
    # 100 frames total. Duration = T seconds. 
    # Interval per frame = T / 100 * 1000 (ms)
    # e.g. 4s -> 40ms/frame
    interval_ms = (duration / 100.0) * 1000
    if interval_ms < 1: interval_ms = 1

    def update(frame):
        s = sho[frame]
        e = elb[frame]
        w = wra[frame]
        
        # Update Upper Arm (Shoulder -> Elbow)
        line_upper.set_data_3d([s[0], e[0]], [s[1], e[1]], [s[2], e[2]])
        
        # Update Forearm (Elbow -> Wrist)
        line_fore.set_data_3d([e[0], w[0]], [e[1], w[1]], [e[2], w[2]])
        
        # Update Joints
        pt_sho.set_data_3d([s[0]], [s[1]], [s[2]])
        pt_elb.set_data_3d([e[0]], [e[1]], [e[2]])
        pt_wra.set_data_3d([w[0]], [w[1]], [w[2]])
        
        return line_upper, line_fore, pt_sho, pt_elb, pt_wra

    ani = FuncAnimation(fig, update, frames=100, interval=interval_ms, blit=False)
    plt.show()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "output", "cvae_model_full.pth")
    stats_path = os.path.join(base_dir, "output", "cvae_stats.npz")
    
    if not os.path.exists(model_path):
        print("Model not found. Run vae.py first.")
        return

    # Load
    model, d_min, d_range = load_model(model_path, stats_path)
    
    # Generate for FMA 45 (Mid-range)
    fma = 20
    print(f"Generating Synthetic Arm Motion for FMA Score: {fma}")
    traj, duration = generate_sample(model, fma, d_min, d_range)
    print(f"Predicted Duration: {duration:.2f} seconds")
    
    # Animate
    animate_stick_figure(traj, duration, fma)

if __name__ == "__main__":
    main()