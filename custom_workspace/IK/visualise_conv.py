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

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from conv_vae import ConvCVAE
except ImportError:
    print("Error: Could not import ConvCVAE from conv_vae.py")
    sys.exit(1)

# --- CONFIG ---
# These are used for generation/animation logic, not model definition anymore
PCA_COMPONENTS = 12
CONDITION_DIM = 1
LATENT_DIM = 64

def generate(model, fma_score):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM)
        c = torch.tensor([[fma_score / 66.0]])
        recon = model.decode(z, c) # (1, 25, 100)
        return recon.squeeze().T.numpy() # (100, 25)

def animate(traj, fma, pca, scaler):
    # Inverse Coupled Scaling
    # Split Pos/Vel/Time
    # traj is (100, 25)
    # 0-11: Position (12)
    # 12-23: Velocity (12)
    # 24: Time (1)
    
    scaled_pos = traj[:, :12]
    # scaled_vel = traj[:, 12:24] 
    
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
    # Assuming standard marker mapping (starts with wrist, etc)
    # This depends on the original CSV structure.
    # Usually: Wrist, Elbow, Shoulder are in specific columns.
    # Indices 0:3, 21:24, 48:51 were used in previous version.
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
        print("Model not found. Run conv_vae.py to train first.")
    else:
        model = ConvCVAE()
        # Ensure model is compatible with the saved state dict
        try:
            model.load_state_dict(torch.load(model_path))
            print("Model loaded successfully.")
            
            with open(pca_path, 'rb') as f: pca = pickle.load(f)
            with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
                
            t = generate(model, 66)
            animate(t, 66, pca, scaler)
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            print("The model definition likely doesn't match the checkpoint.")