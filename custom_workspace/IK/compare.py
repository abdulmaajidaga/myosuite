import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import sys
import os
from scipy.signal import savgol_filter

# --- IMPORTS ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cvae import CVAE
    CVAE_AVAILABLE = True
except ImportError:
    CVAE_AVAILABLE = False

try:
    from conv_vae import ConvCVAE
    CONV_AVAILABLE = True
except ImportError:
    CONV_AVAILABLE = False

try:
    from transformer_vae import TransformerVAE
    TRANS_AVAILABLE = True
except ImportError:
    TRANS_AVAILABLE = False

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
device = "cpu"

def load_pca_models():
    pca_path = os.path.join(OUTPUT_DIR, "pca_model.pkl")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    if not os.path.exists(pca_path) or not os.path.exists(scaler_path):
        return None, None
    with open(pca_path, 'rb') as f: pca = pickle.load(f)
    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    return pca, scaler

def get_cvae_trajectory(fma):
    if not CVAE_AVAILABLE: return None, "CVAE (Missing)"
    model = CVAE().to(device)
    path = os.path.join(OUTPUT_DIR, "cvae_model.pth")
    if not os.path.exists(path): return None, "CVAE (No Weights)"
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    z = torch.randn(1, 32).to(device)
    c = torch.tensor([[fma / 66.0]]).to(device)
    with torch.no_grad():
        recon_traj, _ = model.decode(z, c)
    traj = recon_traj.detach().cpu().squeeze().numpy().reshape(100, 63)
    return traj * 1000.0, "CVAE (Dense)"

def get_conv_trajectory(fma, pca, scaler):
    if not CONV_AVAILABLE or pca is None: return None, "ConvVAE (Missing)"
    model = ConvCVAE().to(device)
    path = os.path.join(OUTPUT_DIR, "conv_cvae.pth")
    if not os.path.exists(path): return None, "ConvVAE (No Weights)"
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    z = torch.randn(1, 64).to(device)
    c = torch.tensor([[fma / 66.0]]).to(device)
    with torch.no_grad():
        recon = model.decode(z, c)
    data = recon.detach().cpu().squeeze().numpy().T
    pca_pos = scaler.inverse_transform(data[:, :12]) 
    return pca.inverse_transform(pca_pos), "Conv1D VAE"

def get_trans_trajectory(fma, pca, scaler):
    if not TRANS_AVAILABLE or pca is None: return None, "TransVAE (Missing)"
    model = TransformerVAE().to(device)
    path = os.path.join(OUTPUT_DIR, "transformer_vae.pth")
    if not os.path.exists(path): return None, "TransVAE (No Weights)"
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    z = torch.randn(1, 64).to(device)
    c = torch.tensor([[fma / 66.0]]).to(device)
    with torch.no_grad():
        recon = model.decode(z, c)
    data = recon.detach().cpu().squeeze().numpy() 
    pca_pos = scaler.inverse_transform(data[:, :12])
    return pca.inverse_transform(pca_pos), "Transformer VAE"

def calculate_metrics(traj):
    if traj is None: return 0, 0
    start, end = traj[0, :3], traj[-1, :3]
    drift = np.linalg.norm(start - end)
    wrist = traj[:, :3]
    jerk = np.mean(np.sum(np.diff(wrist, n=3, axis=0)**2, axis=1)) * 1000 
    return drift, jerk

def main():
    print("Comparing Models in World Space (mm)...")
    pca, scaler = load_pca_models()
    fma = 20 # Compare on Stroke patient movement
    
    results = [
        get_cvae_trajectory(fma),
        get_conv_trajectory(fma, pca, scaler),
        get_trans_trajectory(fma, pca, scaler)
    ]
    
    fig = plt.figure(figsize=(18, 6))
    for i, (traj, name) in enumerate(results):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        if traj is None:
            ax.text(0.5, 0.5, 0.5, name, ha='center'); continue
        
        # Smooth and Calculate
        for k in range(traj.shape[1]):
            traj[:, k] = savgol_filter(traj[:, k], 11, 3)
        drift, jerk = calculate_metrics(traj)
        
        # Plot Wrist
        w = traj[:, :3]
        ax.plot(w[:,0], w[:,1], w[:,2], lw=2, label=f'Jerk: {jerk:.2f}')
        ax.scatter(w[0,0], w[0,1], w[0,2], c='g', s=80, label='Start')
        ax.scatter(w[-1,0], w[-1,1], w[-1,2], c='r', marker='x', s=80, label='End')
        
        color = 'green' if drift < 50 else 'red'
        ax.set_title(f"{name}\nDrift: {drift:.1f}mm", color=color, fontweight='bold')
        ax.legend(); ax.view_init(20, 45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()