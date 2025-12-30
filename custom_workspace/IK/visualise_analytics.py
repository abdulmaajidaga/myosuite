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
    from conv_vae import ConvCVAE, ConvDataset
except ImportError:
    print("Error: Could not import ConvCVAE/ConvDataset from conv_vae.py")
    sys.exit(1)

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # IK/
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "conv_cvae.pth")
PCA_PATH = os.path.join(OUTPUT_DIR, "pca_model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")

# Data Paths
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Root of repo
STROKE_DIR = os.path.join(PROJECT_ROOT, "data", "kinematic", "Stroke")
HEALTHY_DIR = os.path.join(PROJECT_ROOT, "data", "kinematic", "Healthy")
SCORES_FILE = os.path.join(OUTPUT_DIR, "scores.csv")

def load_data_and_model():
    # Load Model
    model = ConvCVAE()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}. Skipping model visualization.")
        model = None

    # Load Data
    # ConvDataset expects (stroke_dir, healthy_dir, scores_file, output_dir)
    # It will re-fit/re-save PCA/Scaler, which is redundant but safe if data hasn't changed.
    # ideally we'd modify ConvDataset to support 'eval' mode, but this works for now.
    print("Loading Dataset (this re-processes raw data)...")
    dataset = ConvDataset(STROKE_DIR, HEALTHY_DIR, SCORES_FILE, OUTPUT_DIR)
    
    return model, dataset

def plot_latent_tsne(model, dataset):
    print("Generating Latent Space t-SNE...")
    
    latent_vectors = []
    labels = [] # 0 for Stroke, 1 for Healthy
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for x, c in loader:
            # Encode
            # x shape: (B, 24, 100)
            # c shape: (B, 1)
            mu, logvar = model.encode(x, c)
            z = model.reparameterize(mu, logvar)
            
            latent_vectors.append(z.numpy())
            
            # Create labels based on Condition 'c' (FMA score)
            # c is normalized FMA (0-1). Healthy is ~1.0, Stroke is < 1.0
            # Stroke range: 16-20 -> norm 0.24-0.30
            # Healthy: 66 -> norm 1.0
            is_healthy = (c > 0.9).numpy().flatten()
            labels.append(is_healthy)

    if not latent_vectors: return

    X = np.concatenate(latent_vectors, axis=0)
    y = np.concatenate(labels, axis=0)
    
    # Run t-SNE
    # Perplexity must be < number of samples.
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
    
    # Access dataset directly. 
    # dataset[i] returns (x, c). x is (24, 100).
    
    # Collect indices
    n = len(dataset)
    indices = np.arange(n)
    
    # Separate by condition
    # Accessing item one by one is slow if huge, but fine for small N.
    # We can just iterate once.
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
        # x is (24, 100). Position PCs are 0-11.
        pc1 = x[0, :].numpy()
        pc2 = x[1, :].numpy()
        ax.plot(pc1, pc2, time, c='green', alpha=0.4, linewidth=1)

    # Plot Stroke (Red)
    for x in stroke_samples:
        pc1 = x[0, :].numpy()
        pc2 = x[1, :].numpy()
        ax.plot(pc1, pc2, time, c='red', alpha=0.8, linewidth=1.5)
        
    # Legend
    ax.plot([], [], [], c='green', label='Healthy')
    ax.plot([], [], [], c='red', label='Stroke')
    
    ax.set_xlabel("PC 1 (Primary Movement)")
    ax.set_ylabel("PC 2 (Secondary Deviation)")
    ax.set_zlabel("Time (Frames)")
    ax.set_title("Temporal Dynamics in Latent PCA Space")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    model, dataset = load_data_and_model()
    
    if model:
        plot_latent_tsne(model, dataset)
    
    plot_trajectory_dynamics(dataset)
