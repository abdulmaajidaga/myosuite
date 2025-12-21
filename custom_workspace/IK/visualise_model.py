"""
===============================================================================
FILE: visualize_model.py
===============================================================================
Visualizes the PCA Latent Space and the learned Regression Path.
Helps you see if the model is interpolating correctly between Sick and Healthy.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# --- CONFIG ---
DATA_DIR = "/home/abdul/Desktop/myosuite/custom_workspace/IK/output"
SCORES_FILE = os.path.join(DATA_DIR, "scores.csv")
MODEL_PATH = os.path.join(DATA_DIR, "motion_generator_model.pkl")
FIXED_FRAMES = 100 # Must match training script

def read_mot_file(filepath):
    """Helper to load and normalize motion data exactly like the trainer."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        header_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == 'endheader':
                header_idx = i
                break
        data = np.loadtxt(lines[header_idx+2:])
        joints = data[:, 1:] 
        return signal.resample(joints, FIXED_FRAMES).flatten()
    except Exception as e:
        return None

def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Run train_motion_model.py first.")
        return

    print("Loading Model and Data...")
    model = joblib.load(MODEL_PATH)
    pca = model['pca']
    regressors = model['regressors']
    scaler = model['score_scaler']
    
    # 1. RE-LOAD REAL DATA FOR PLOTTING
    # We need to re-process the files to project them into PCA space
    df = pd.read_csv(SCORES_FILE)
    real_pca_points = []
    real_scores = []
    
    for idx, row in df.iterrows():
        path = os.path.join(DATA_DIR, row['filename'])
        if os.path.exists(path):
            flat_motion = read_mot_file(path)
            if flat_motion is not None:
                # Project this patient into the latent space
                latent = pca.transform([flat_motion])[0]
                real_pca_points.append(latent)
                real_scores.append(row['fma_score'])
    
    X_real = np.array(real_pca_points) # Shape: (N_samples, N_components)
    y_real = np.array(real_scores)

    # 2. GENERATE THE "LEARNED PATH" (Regression Line)
    # Generate scores from 0 to 70 to visualize the full trajectory
    synth_scores = np.linspace(0, 70, 100).reshape(-1, 1)
    synth_scores_norm = scaler.transform(synth_scores)
    
    # Predict latent vector for each synthetic score
    synth_pca_points = np.zeros((100, len(regressors)))
    for i, regr in enumerate(regressors):
        synth_pca_points[:, i] = regr.predict(synth_scores_norm)

    # --- PLOTTING ---
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Model Visualization (Variance Explained: {np.sum(pca.explained_variance_ratio_)*100:.1f}%)", fontsize=16)

    # A. THE MAP (PC1 vs PC2)
    # This shows the "shape" of the movement space.
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot Real Data
    sc = ax1.scatter(X_real[:, 0], X_real[:, 1], c=y_real, cmap='viridis', s=100, edgecolors='k', zorder=5)
    plt.colorbar(sc, ax=ax1, label='FMA Score')
    
    # Plot Learned Path
    ax1.plot(synth_pca_points[:, 0], synth_pca_points[:, 1], 'r--', linewidth=2, label='Learned Path', zorder=4)
    ax1.scatter(synth_pca_points[-1, 0], synth_pca_points[-1, 1], c='red', marker='*', s=200, label='Projected Healthy')
    
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title("The Motion Space (Map)")
    ax1.legend()

    # B. COMPONENT TRENDS
    # How does each component change as Score increases?
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_real, X_real[:, 0], c='blue', label='Real Data PC1')
    ax2.plot(synth_scores, synth_pca_points[:, 0], 'r--', label='Model Prediction')
    ax2.set_xlabel("FMA Score")
    ax2.set_ylabel("Principal Component 1")
    ax2.set_title("PC1 Trend (Usually Range of Motion)")
    ax2.legend()

    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(y_real, X_real[:, 1], c='green', label='Real Data PC2')
    ax3.plot(synth_scores, synth_pca_points[:, 1], 'r--', label='Model Prediction')
    ax3.set_xlabel("FMA Score")
    ax3.set_ylabel("Principal Component 2")
    ax3.set_title("PC2 Trend (Usually Coordination)")

    # C. VARIANCE BAR CHART
    ax4 = plt.subplot(2, 1, 2)
    indices = range(1, len(pca.explained_variance_ratio_) + 1)
    ax4.bar(indices, pca.explained_variance_ratio_ * 100, color='skyblue', edgecolor='black')
    ax4.plot(indices, np.cumsum(pca.explained_variance_ratio_ * 100), 'r-o', label='Cumulative Variance')
    ax4.set_xlabel("Principal Component Index")
    ax4.set_ylabel("Variance Explained (%)")
    ax4.set_title("How much info does the model capture?")
    ax4.legend()

    plt.tight_layout()
    
    # Save
    out_file = os.path.join(DATA_DIR, "model_visualization.png")
    plt.savefig(out_file)
    print(f"\n📊 Visualization saved to: {out_file}")
    print("Open this image to see how your patients cluster!")

if __name__ == "__main__":
    main()