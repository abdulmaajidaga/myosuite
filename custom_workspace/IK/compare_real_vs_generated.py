import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import sys
import os
import glob
from scipy.signal import savgol_filter, resample

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STROKE_DIR = os.path.join(PROJECT_ROOT, "data", "kinematic", "Stroke")
HEALTHY_DIR = os.path.join(PROJECT_ROOT, "data", "kinematic", "Healthy")
SEQ_LEN = 100
N_SAMPLES = 20 # Number of samples to average over

sys.path.append(BASE_DIR)

# --- IMPORTS ---
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

# --- UTILS ---
def load_pca_models():
    pca_path = os.path.join(OUTPUT_DIR, "pca_model.pkl")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    if not os.path.exists(pca_path) or not os.path.exists(scaler_path): return None, None
    with open(pca_path, 'rb') as f: pca = pickle.load(f)
    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    return pca, scaler

def process_raw_file(filepath):
    """Reads CSV, resamples to 100 frames, returns (100, 63) array."""
    try:
        # Skip header rows based on project convention (usually 2)
        df = pd.read_csv(filepath, skiprows=2, header=None)
        if df.shape[1] < 63: return None
        
        raw_data = df.iloc[:, :63].ffill().bfill().fillna(0).values
        # Resample to 100 frames
        resampled = resample(raw_data, SEQ_LEN)
        return resampled
    except Exception:
        return None

def get_real_data_stats(directory, n=20):
    files = glob.glob(os.path.join(directory, "*.csv"))
    np.random.shuffle(files)
    files = files[:n]
    
    trajectories = []
    for f in files:
        traj = process_raw_file(f)
        if traj is not None:
            trajectories.append(traj)
            
    return trajectories

# --- GENERATORS (From compare.py) ---
def gen_cvae(model, fma):
    z = torch.randn(1, 32)
    c = torch.tensor([[fma / 66.0]])
    with torch.no_grad():
        recon, _ = model.decode(z, c)
    traj = recon.detach().numpy().reshape(100, 63)
    return traj * 1000.0 # Scale to mm

def gen_conv(model, pca, scaler, fma):
    z = torch.randn(1, 64)
    c = torch.tensor([[fma / 66.0]])
    with torch.no_grad():
        recon = model.decode(z, c) # (1, 25, 100)
    data = recon.detach().numpy().squeeze().T # (100, 25)
    pca_pos = scaler.inverse_transform(data[:, :12])
    return pca.inverse_transform(pca_pos)

def gen_trans(model, pca, scaler, fma):
    z = torch.randn(1, 64)
    c = torch.tensor([[fma / 66.0]])
    with torch.no_grad():
        recon = model.decode(z, c) # (1, 100, 24)
    data = recon.detach().numpy().squeeze()
    pca_pos = scaler.inverse_transform(data[:, :12])
    return pca.inverse_transform(pca_pos)

# --- METRICS ---
def calculate_metrics(traj):
    # Traj: (100, 63)
    
    # 1. Drift (mm)
    start = traj[0, :3]
    end = traj[-1, :3]
    drift = np.linalg.norm(start - end)
    
    # 2. Path Length (mm) - Arc length of wrist
    wrist = traj[:, :3]
    dist = np.diff(wrist, axis=0)
    path_len = np.sum(np.linalg.norm(dist, axis=1))
    
    # 3. Smoothness (Jerk)
    vel = np.diff(wrist, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    # Mean squared jerk, scaled
    smoothness = np.mean(np.sum(jerk**2, axis=1)) * 1000
    
    return drift, path_len, smoothness

def print_stats(name, trajectories):
    drifts, paths, smooths = [], [], []
    for t in trajectories:
        d, p, s = calculate_metrics(t)
        drifts.append(d)
        paths.append(p)
        smooths.append(s)
        
    print(f"{name:20} | Drift: {np.mean(drifts):6.2f} ±{np.std(drifts):5.2f} | Path: {np.mean(paths):6.2f} | Jerk: {np.mean(smooths):6.2f}")
    return np.mean(drifts), np.mean(paths), np.mean(smooths)

# --- PLOTTING ---
def plot_comparison(real_traj, gen_trajs, names, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Real (Black)
    # Filter for nice visual
    real_smooth = savgol_filter(real_traj, 11, 3, axis=0)
    r = real_smooth[:, :3]
    ax.plot(r[:,0], r[:,1], r[:,2], 'k-', lw=3, label='Real Data', alpha=0.8)
    
    colors = ['r', 'b', 'g', 'm']
    
    for i, t in enumerate(gen_trajs):
        if t is None: continue
        t_smooth = savgol_filter(t, 11, 3, axis=0)
        w = t_smooth[:, :3]
        ax.plot(w[:,0], w[:,1], w[:,2], color=colors[i%len(colors)], lw=2, label=names[i], alpha=0.7)
        
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()

def main():
    print("Loading Models and Data...")
    pca, scaler = load_pca_models()
    
    # Load Models
    models = {}
    if CVAE_AVAILABLE:
        m = CVAE()
        p = os.path.join(OUTPUT_DIR, "cvae_model.pth")
        if os.path.exists(p):
            m.load_state_dict(torch.load(p, map_location='cpu'))
            m.eval()
            models['CVAE'] = m
            
    if CONV_AVAILABLE and pca is not None:
        m = ConvCVAE()
        p = os.path.join(OUTPUT_DIR, "conv_cvae.pth")
        if os.path.exists(p):
            m.load_state_dict(torch.load(p, map_location='cpu'))
            m.eval()
            models['Conv'] = m
            
    if TRANS_AVAILABLE and pca is not None:
        m = TransformerVAE()
        p = os.path.join(OUTPUT_DIR, "transformer_vae.pth")
        if os.path.exists(p):
            m.load_state_dict(torch.load(p, map_location='cpu'))
            m.eval()
            models['Trans'] = m

    # --- 1. STROKE COMPARISON ---
    print("\n" + "="*60)
    print("   STROKE PATIENT COMPARISON (Target FMA: 20)")
    print("="*60)
    
    real_stroke = get_real_data_stats(STROKE_DIR, n=N_SAMPLES)
    if not real_stroke:
        print("Error: No real stroke data found.")
        return

    print_stats("Real Data (Stroke)", real_stroke)
    
    gen_stroke_trajs = []
    gen_names = []
    
    # Generate Batch
    for name, model in models.items():
        batch = []
        for _ in range(N_SAMPLES):
            if name == 'CVAE': t = gen_cvae(model, 20)
            elif name == 'Conv': t = gen_conv(model, pca, scaler, 20)
            elif name == 'Trans': t = gen_trans(model, pca, scaler, 20)
            batch.append(t)
        
        print_stats(f"Generated ({name})", batch)
        gen_stroke_trajs.append(batch[0]) # Save one for plot
        gen_names.append(name)

    # --- 2. HEALTHY COMPARISON ---
    print("\n" + "="*60)
    print("   HEALTHY COMPARISON (Target FMA: 66)")
    print("="*60)
    
    real_healthy = get_real_data_stats(HEALTHY_DIR, n=N_SAMPLES)
    
    print_stats("Real Data (Healthy)", real_healthy)
    
    gen_healthy_trajs = []
    
    for name, model in models.items():
        batch = []
        for _ in range(N_SAMPLES):
            if name == 'CVAE': t = gen_cvae(model, 66)
            elif name == 'Conv': t = gen_conv(model, pca, scaler, 66)
            elif name == 'Trans': t = gen_trans(model, pca, scaler, 66)
            batch.append(t)
        
        print_stats(f"Generated ({name})", batch)
        gen_healthy_trajs.append(batch[0])

    # Plotting
    print("\nDisplaying Stroke Comparison Plot...")
    plot_comparison(real_stroke[0], gen_stroke_trajs, gen_names, "Stroke Comparison (FMA 20)")
    
    print("Displaying Healthy Comparison Plot...")
    plot_comparison(real_healthy[0], gen_healthy_trajs, gen_names, "Healthy Comparison (FMA 66)")

if __name__ == "__main__":
    main()
