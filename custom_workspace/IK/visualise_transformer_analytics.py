import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import pickle
import os
import sys

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from transformer_vae import TransformerVAE, TransformerDataset
except ImportError:
    print("Error: Could not import TransformerVAE/TransformerDataset from transformer_vae.py")
    sys.exit(1)

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # IK/
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "transformer_vae.pth")
PCA_PATH = os.path.join(OUTPUT_DIR, "pca_model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")

# Data Paths
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Root of repo
STROKE_DIR = os.path.join(PROJECT_ROOT, "data", "kinematic", "Stroke")
HEALTHY_DIR = os.path.join(PROJECT_ROOT, "data", "kinematic", "Healthy")
SCORES_FILE = os.path.join(OUTPUT_DIR, "scores.csv")

def load_data_and_model():
    # Load Model
    model = TransformerVAE()
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Model not found at {MODEL_PATH}. Skipping model visualization.")
        model = None

    # Load Data
    print("Loading Dataset (this re-processes raw data)...")
    dataset = TransformerDataset(STROKE_DIR, HEALTHY_DIR, SCORES_FILE, OUTPUT_DIR)
    
    return model, dataset

def plot_latent_tsne(model, dataset):
    print("Generating Latent Space t-SNE...")
    
    latent_vectors = []
    labels = [] # 0 for Stroke, 1 for Healthy
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for x, c in loader:
            # Transformer dataset returns x: (B, 100, 24)
            # Encode
            mu, logvar = model.encode(x, c)
            z = model.reparameterize(mu, logvar)
            
            latent_vectors.append(z.numpy())
            
            # c is normalized FMA (0-1). Healthy is ~1.0
            is_healthy = (c > 0.9).numpy().flatten()
            labels.append(is_healthy)

    if not latent_vectors: return

    X = np.concatenate(latent_vectors, axis=0)
    y = np.concatenate(labels, axis=0)
    
    # Run t-SNE
    n_samples = X.shape[0]
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000, init='random')
    X_embedded = tsne.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_embedded[y==0, 0], X_embedded[y==0, 1], c='red', label='Stroke', alpha=0.6)
    plt.scatter(X_embedded[y==1, 0], X_embedded[y==1, 1], c='green', label='Healthy', alpha=0.6)
    plt.title("Latent Space t-SNE (Cluster Separation)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_trajectory_dynamics(dataset):
    print("Generating 3D Trajectory Dynamics (PC1 vs PC2 vs Time)...")
    
    # Dataset samples: (x, c)
    # x shape: (100, 24) -> (Time, Channels)
    
    n = len(dataset)
    
    stroke_samples = []
    healthy_samples = []
    
    for i in range(n):
        x, c = dataset[i]
        if c.item() > 0.9:
            if len(healthy_samples) < 10: healthy_samples.append(x)
        else:
            if len(stroke_samples) < 10: stroke_samples.append(x)
            
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    time = np.arange(100)
    
    # Plot Healthy (Green)
    for x in healthy_samples:
        # x is (100, 24)
        # PC1 is column 0, PC2 is column 1
        pc1 = x[:, 0].numpy()
        pc2 = x[:, 1].numpy()
        ax.plot(pc1, pc2, time, c='green', alpha=0.4, linewidth=1)

    # Plot Stroke (Red)
    for x in stroke_samples:
        pc1 = x[:, 0].numpy()
        pc2 = x[:, 1].numpy()
        ax.plot(pc1, pc2, time, c='red', alpha=0.8, linewidth=1.5)
        
    ax.plot([], [], [], c='green', label='Healthy')
    ax.plot([], [], [], c='red', label='Stroke')
    
    ax.set_xlabel("PC 1 (Primary Movement)")
    ax.set_ylabel("PC 2 (Secondary Deviation)")
    ax.set_zlabel("Time (Frames)")
    ax.set_title("Temporal Dynamics in Latent PCA Space")
    ax.legend()
    plt.show()

def plot_reconstruction_comparison(model, dataset):
    if model is None: return
    print("Generating Reconstruction Comparison...")
    
    # Pick one Stroke sample
    target_idx = -1
    for i in range(len(dataset)):
        if dataset[i][1].item() < 0.5: # Low score
            target_idx = i
            break
            
    if target_idx == -1: return

    x, c = dataset[target_idx]
    
    model.eval()
    with torch.no_grad():
        # Add batch dim
        x_in = x.unsqueeze(0) # (1, 100, 24)
        c_in = c.unsqueeze(0) # (1, 1)
        recon, _, _ = model(x_in, c_in)
        
    x_np = x.numpy()
    recon_np = recon.squeeze().numpy()
    
    # Plot PC1 (Main motion)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_np[:, 0], label="Original PC1", color='black')
    plt.plot(recon_np[:, 0], label="Reconstructed PC1", color='blue', linestyle='--')
    plt.title("PC 1 Reconstruction (Position)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_np[:, 12], label="Original PC1 Velocity", color='black')
    plt.plot(recon_np[:, 12], label="Reconstructed Velocity", color='red', linestyle='--')
    plt.title("PC 1 Velocity Reconstruction")
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    model, dataset = load_data_and_model()
    
    if model:
        plot_latent_tsne(model, dataset)
        plot_reconstruction_comparison(model, dataset)
    
    plot_trajectory_dynamics(dataset)
