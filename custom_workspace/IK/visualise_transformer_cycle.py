import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import sys
from scipy.signal import savgol_filter

# --- IMPORT TRANSFORMER MODEL ---
# Ensure we can import from the IK directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from transformer_vae import TransformerVAE
except ImportError:
    print("Error: Could not import TransformerVAE from transformer_vae.py")
    sys.exit(1)

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "transformer_vae.pth")
PCA_PATH = os.path.join(OUTPUT_DIR, "pca_model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
LATENT_DIM = 64 # Matches transformer_vae.py

def load_components():
    # Load Transforms
    if not os.path.exists(PCA_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: PCA or Scaler models not found in output/ directory.")
        sys.exit(1)
        
    with open(PCA_PATH, 'rb') as f: pca = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    
    # Load Model
    model = TransformerVAE()
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    return model, pca, scaler

def generate_cycle(model, pca, scaler, fma_score):
    # 1. Prepare Latent Input
    # TransformerVAE.decode expects (z, c)
    z = torch.randn(1, LATENT_DIM)
    c = torch.tensor([[fma_score / 66.0]]) # Normalized FMA
    
    with torch.no_grad():
        # Decode
        recon = model.decode(z, c) # Returns (1, 100, 24)
    
    # 2. Inverse Transform
    data_np = recon.squeeze().numpy() # (100, 24)
    
    # Split Pos/Vel
    scaled_pos = data_np[:, :12]
    
    # Inverse Scale
    pca_pos = scaler.inverse_transform(scaled_pos)
    
    # Inverse PCA to get back to original joint coordinates (63 channels)
    markers = pca.inverse_transform(pca_pos) # (100, 63)
    
    # Smooth markers for better visualization
    for i in range(63):
        markers[:, i] = savgol_filter(markers[:, i], window_length=11, polyorder=3)
        
    return markers

def plot_cycle_check(traj_healthy, traj_stroke):
    fig = plt.figure(figsize=(14, 6))
    
    # --- PLOT 1: Healthy Cycle ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Full Path (Using first 3 columns as representative wrist marker)
    ax1.plot(traj_healthy[:, 0], traj_healthy[:, 1], traj_healthy[:, 2], 
             c='green', alpha=0.6, label='Trajectory')
    
    # Highlight START (Green Dot) and END (Red X)
    ax1.scatter(traj_healthy[0,0], traj_healthy[0,1], traj_healthy[0,2], 
                c='lime', s=100, label='Start (Fr 0)')
    ax1.scatter(traj_healthy[-1,0], traj_healthy[-1,1], traj_healthy[-1,2], 
                c='red', marker='x', s=100, label='End (Fr 100)')
    
    dist_h = np.linalg.norm(traj_healthy[0,:3]-traj_healthy[-1,:3])
    ax1.set_title(f"Healthy Cycle (FMA 66)\nDist(Start-End): {dist_h:.2f}mm")
    ax1.legend()
    
    # --- PLOT 2: Stroke Cycle ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(traj_stroke[:, 0], traj_stroke[:, 1], traj_stroke[:, 2], 
             c='red', alpha=0.6, label='Trajectory')
    
    ax2.scatter(traj_stroke[0,0], traj_stroke[0,1], traj_stroke[0,2], 
                c='lime', s=100, label='Start')
    ax2.scatter(traj_stroke[-1,0], traj_stroke[-1,1], traj_stroke[-1,2], 
                c='darkred', marker='x', s=100, label='End')
    
    dist_s = np.linalg.norm(traj_stroke[0,:3]-traj_stroke[-1,:3])
    ax2.set_title(f"Stroke Cycle (FMA 20)\nDist(Start-End): {dist_s:.2f}mm")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model, pca, scaler = load_components()
    print("Generating trajectories for cycle check...")
    
    # Generate for healthy (66) and stroke (20)
    h_traj = generate_cycle(model, pca, scaler, 66)
    s_traj = generate_cycle(model, pca, scaler, 20)
    
    plot_cycle_check(h_traj, s_traj)
