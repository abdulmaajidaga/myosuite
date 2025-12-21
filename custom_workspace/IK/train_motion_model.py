"""
===============================================================================
FILE: train_motion_model.py
===============================================================================
Trains a statistical model to generate arm motion based on FMA-UE scores.
Uses PCA to compress motion and SVR (Support Vector Regression) to map scores.
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ========================================
# CONFIGURATION
# ========================================
DATA_DIR = "/home/abdul/Desktop/myosuite/custom_workspace/IK/output"
SCORES_FILE = os.path.join(DATA_DIR, "scores.csv")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "motion_generator_model.pkl")

# We must normalize all motions to a fixed number of frames to compare them
FIXED_FRAMES = 100 
# How many dimensions to keep (3-5 is usually enough for 25 samples)
N_COMPONENTS = 5 
# ========================================

def read_mot_file(filepath):
    """Reads a MOT file, extracts joint data, and resamples to fixed length."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find header
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == 'endheader':
            header_idx = i
            break
            
    # Load data
    col_names = lines[header_idx+1].strip().split('\t')
    data = np.loadtxt(lines[header_idx+2:])
    
    # Separate Time and Joints
    # Assuming column 0 is time
    joints = data[:, 1:] 
    joint_names = col_names[1:]
    
    # Resample to FIXED_FRAMES
    # This stretches/shrinks the motion so they align in time
    resampled_joints = signal.resample(joints, FIXED_FRAMES)
    
    return resampled_joints, joint_names

def flatten_motion(motion_data):
    """Flattens (Frames, Joints) into a 1D vector."""
    return motion_data.flatten()

def unflatten_motion(flat_vector, n_joints):
    """Restores 1D vector back to (Frames, Joints)."""
    return flat_vector.reshape(FIXED_FRAMES, n_joints)

def train_model():
    print(f"--- Training Motion Generator (N={N_COMPONENTS} PCs) ---")
    
    # 1. Load Scores
    df_scores = pd.read_csv(SCORES_FILE)
    
    X_motions = []
    y_scores = []
    joint_names = None
    
    print(f"Loading {len(df_scores)} files...")
    
    for idx, row in df_scores.iterrows():
        mot_path = os.path.join(DATA_DIR, row['filename'])
        if not os.path.exists(mot_path):
            print(f"  ⚠️ Missing: {mot_path}")
            continue
            
        motion, names = read_mot_file(mot_path)
        if joint_names is None: joint_names = names
        
        # Flatten structure for PCA
        X_motions.append(flatten_motion(motion))
        y_scores.append(row['fma_score'])

    X = np.array(X_motions)
    y = np.array(y_scores).reshape(-1, 1) # 2D array for scaler

    # 2. Dimensionality Reduction (PCA)
    # Compresses (25 samples x 3000 features) -> (25 samples x 5 features)
    pca = PCA(n_components=N_COMPONENTS)
    X_latent = pca.fit_transform(X)
    
    print(f"  > PCA Explained Variance: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

    # 3. Regression (Map Score -> Latent Space)
    # We normalize scores to 0-1 range for better regression performance
    score_scaler = MinMaxScaler()
    y_norm = score_scaler.fit_transform(y)
    
    # We train one regressor per PC component
    regressors = []
    for i in range(N_COMPONENTS):
        # SVR is robust for small datasets
        # C=1.0, epsilon=0.1 are standard defaults
        regr = SVR(kernel='rbf', C=10, gamma='scale') 
        regr.fit(y_norm, X_latent[:, i])
        regressors.append(regr)
        
    print("  > Regressors trained.")

    # 4. Save Everything
    model_data = {
        'pca': pca,
        'regressors': regressors,
        'score_scaler': score_scaler,
        'joint_names': joint_names,
        'n_joints': len(joint_names),
        'X_shape': X.shape
    }
    
    joblib.dump(model_data, MODEL_SAVE_PATH)
    print(f"✓ Model saved to: {MODEL_SAVE_PATH}")

def generate_motion(target_score, output_filename="generated.mot"):
    """
    Generates a new MOT file for a specific FMA score.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        print("Error: Model not found. Train first.")
        return

    # Load Model
    model = joblib.load(MODEL_SAVE_PATH)
    pca = model['pca']
    regressors = model['regressors']
    scaler = model['score_scaler']
    n_joints = model['n_joints']
    joint_names = model['joint_names']

    print(f"\n--- Generating Motion for FMA Score: {target_score} ---")

    # 1. Normalize Input Score
    score_in = np.array([[target_score]])
    score_norm = scaler.transform(score_in)

    # 2. Predict Latent Vector (PCA weights)
    latent_vector = np.zeros((1, N_COMPONENTS))
    for i, regr in enumerate(regressors):
        latent_vector[0, i] = regr.predict(score_norm)[0]

    # 3. Reconstruct Full Motion (Inverse PCA)
    flat_motion = pca.inverse_transform(latent_vector)[0]
    motion_data = unflatten_motion(flat_motion, n_joints)

    # 4. Save to MOT
    save_path = os.path.join(DATA_DIR, output_filename)
    
    # Create fake time array (0 to 1 second normalized)
    # You can scale this by duration if you have duration data
    time_arr = np.linspace(0, 2.0, FIXED_FRAMES) 
    
    with open(save_path, 'w') as f:
        f.write(f"dataset\nversion=1\nnRows={FIXED_FRAMES}\nnColumns={n_joints+1}\ninDegrees=no\nendheader\n")
        f.write("time\t" + "\t".join(joint_names) + "\n")
        for i in range(FIXED_FRAMES):
            row_str = "\t".join([f"{x:.6f}" for x in motion_data[i]])
            f.write(f"{time_arr[i]:.6f}\t{row_str}\n")
            
    print(f"✓ Generated file saved: {save_path}")

# ========================================
# MAIN ENTRY
# ========================================
if __name__ == "__main__":
    # 1. Train (Run this once or whenever data changes)
    if not os.path.exists(MODEL_SAVE_PATH):
        train_model()
    else:
        print("Model already exists. Loading...")

    # 2. Example Generation
    # Generate a motion for a low score (poor movement)
    generate_motion(target_score=15, output_filename="gen_score_15.mot")
    
    # Generate a motion for a high score (good movement)
    generate_motion(target_score=60, output_filename="gen_score_60.mot")