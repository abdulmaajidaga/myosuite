import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import re

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Paths
AUGMENTED_DIR = os.path.join(PROJECT_ROOT, "data/kinematic/Augmented")
REAL_HEALTHY_DIR = os.path.join(PROJECT_ROOT, "data/kinematic/Healthy/processed")
REAL_STROKE_DIR = os.path.join(PROJECT_ROOT, "data/kinematic/Stroke/processed")
SCORES_FILE = os.path.join(BASE_DIR, "output/scores.csv")

SEQ_LEN = 100
EXPECTED_COLS = 12 

def load_score_map():
    try:
        df = pd.read_csv(SCORES_FILE)
        id_col = df.columns[0]
        score_col = df.columns[1]
        df[id_col] = df[id_col].astype(str).str.replace('.mot', '').str.strip()
        return dict(zip(df[id_col], df[score_col]))
    except:
        return {}

def load_from_dir(directory, label_type, default_score=None, score_map=None):
    data_list = []
    scores = []
    types = []
    
    if not os.path.exists(directory):
        print(f"Skipping {label_type} (Dir not found): {directory}")
        return [], [], []

    files = []
    # robust scan
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".csv"):
                files.append(os.path.join(root, filename))

    print(f"Scanning {label_type}: Found {len(files)} files.")

    for f in files:
        try:
            name = os.path.basename(f).replace('_processed.csv', '').replace('.csv', '')
            final_score = default_score
            
            if label_type == "Augmented":
                match = re.search(r'FMA_(\d+)', name)
                if match: final_score = int(match.group(1))
            elif label_type == "Real Stroke" and score_map:
                final_score = score_map.get(name)
                # Try fallback lookup
                if final_score is None:
                    final_score = score_map.get(name.split('_')[0])
                if final_score is None: final_score = 30 
            elif label_type == "Real Healthy":
                final_score = 66

            if final_score is None: continue 

            # Load
            df = pd.read_csv(f)
            data = df.select_dtypes(include=[np.number]).values
            
            if len(data) == 0: continue

            # [REMOVED] Unit Correction Block was here.
            # Data is now kept in its original units (likely Millimeters).

            if len(data) != SEQ_LEN:
                data = resample(data, SEQ_LEN)
            
            flat_vector = data.flatten()
            
            if len(flat_vector) == SEQ_LEN * EXPECTED_COLS:
                data_list.append(flat_vector)
                scores.append(final_score)
                types.append(label_type)

        except Exception as e:
            continue
            
    return data_list, scores, types

def main():
    score_map = load_score_map()
    
    h_X, h_y, h_t = load_from_dir(REAL_HEALTHY_DIR, "Real Healthy", default_score=66)
    s_X, s_y, s_t = load_from_dir(REAL_STROKE_DIR, "Real Stroke", score_map=score_map)
    a_X, a_y, a_t = load_from_dir(AUGMENTED_DIR, "Augmented")
    
    # if len(a_X) > 600:
    #     indices = np.random.choice(len(a_X), 600, replace=False)
    #     a_X = [a_X[i] for i in indices]
    #     a_y = [a_y[i] for i in indices]
    #     a_t = [a_t[i] for i in indices]

    X = np.array(h_X + s_X + a_X)
    y = h_y + s_y + a_y
    labels = h_t + s_t + a_t

    if len(X) == 0:
        print("No data found!")
        return

    # --- Standardize ---
    print("Standardizing Data (centering at 0)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # --------------------------------

    print(f"Running PCA on {len(X)} samples...")
    pca = PCA(n_components=2)
    # Use SCALED data for PCA
    X_pca = pca.fit_transform(X_scaled) 
    
    print(f"Explained Variance: {pca.explained_variance_ratio_}")

    # Plot
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'FMA Score': y,
        'Dataset': labels
    })

    # --- INTERACTIVE PLOT ---
    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(left=0.2) # Make room for checkboxes on the left

    markers = {"Augmented": "o", "Real Healthy": "P", "Real Stroke": "X"}
    datasets = ["Augmented", "Real Healthy", "Real Stroke"]
    
    # Color mapping
    norm = plt.Normalize(plot_df['FMA Score'].min(), plot_df['FMA Score'].max())
    cmap = plt.cm.viridis
    
    scatters = {}
    
    for ds in datasets:
        subset = plot_df[plot_df['Dataset'] == ds]
        if subset.empty: continue
        
        # Lower opacity for Augmented data to avoid clutter
        alpha_val = 0.2 if ds == "Augmented" else 0.9
        
        sc = ax.scatter(
            subset['PC1'], subset['PC2'],
            c=subset['FMA Score'], cmap=cmap, norm=norm,
            marker=markers.get(ds, 'o'),
            s=80, alpha=alpha_val, edgecolors='black',
            label=ds
        )
        scatters[ds] = sc

    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('FMA Score')

    ax.set_title('Manifold of Recovery (Standardized)')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Add Checkboxes
    from matplotlib.widgets import CheckButtons
    
    # Position: [left, bottom, width, height]
    rax = plt.axes([0.02, 0.5, 0.12, 0.15])
    labels_list = list(scatters.keys())
    visibility = [True] * len(labels_list)
    check = CheckButtons(rax, labels_list, visibility)

    def func(label):
        scatters[label].set_visible(not scatters[label].get_visible())
        plt.draw()

    check.on_clicked(func)
    
    save_path = os.path.join(BASE_DIR, "output/pca_manifold_comparison.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()