import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import sys
import cvae
# --- CONFIGURATION MATCHING VAE.PY ---
INPUT_DIM = cvae.INPUT_DIM
CONDITION_DIM = cvae.CONDITION_DIM
LATENT_DIM = cvae.LATENT_DIM
HIDDEN_DIM = cvae.HIDDEN_DIM


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
        )
        
        # --- DECODER HEAD 2: DURATION ---
        # Takes z + FMA -> 1 scalar
        self.decoder_dur = nn.Sequential(
            nn.Linear(LATENT_DIM + CONDITION_DIM, 32), # Small, focused brain
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
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

def load_model(model_path, stats_path):
    # Load Stats (Optional now, but keeping for compatibility)
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        data_min = stats['data_min']
        data_max = stats['data_max']
        data_range = data_max - data_min
    else:
        data_min, data_range = None, None
    
    # Load Model (Instantiate without arguments)
    model = CVAE()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, data_min, data_range

def generate_sample(model, target_fma, data_min, data_range):
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM)
        norm_fma = target_fma / 66.0
        c = torch.tensor([[norm_fma]]).float()
        
        # Get separate outputs
        recon_traj, recon_dur = model.decode(z, c)
        
        # Process Trajectory
        # Denormalize Trajectory (Assuming approx range -1 to 1 based on training)
        # Training used: traj = raw / max_val. 
        # We don't have per-sample max_val here, so we must assume a standard scale.
        # Assuming typical max value approx 600-1000mm.
        traj = recon_traj.numpy().flatten() * 600.0
        
        # Process Duration
        # Training used: norm_dur = real_dur / 10.0
        duration = recon_dur.item() * 10.0
        
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
    model_path = os.path.join(base_dir, "output", "cvae_model.pth")
    stats_path = os.path.join(base_dir, "output", "cvae_stats.npz")
    
    if not os.path.exists(model_path):
        print("Model not found. Run vae.py first.")
        return

    # Load
    model, d_min, d_range = load_model(model_path, stats_path)
    
    # Generate for FMA 45 (Mid-range)
    fma = 66
    print(f"Generating Synthetic Arm Motion for FMA Score: {fma}")
    traj, duration = generate_sample(model, fma, d_min, d_range)
    print(f"Predicted Duration: {duration:.2f} seconds")
    
    # Animate
    animate_stick_figure(traj, duration, fma)

if __name__ == "__main__":
    main()