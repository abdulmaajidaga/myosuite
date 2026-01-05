import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import resample
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap # pip install umap-learn

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
        return [], [], []

    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".csv"):
                files.append(os.path.join(root, filename))

    # Downsample augmented slightly to speed up t-SNE/UMAP
    # if label_type == "Augmented" and len(files) > 800:
    #     files = np.random.choice(files, 800, replace=False)

    print(f"Loading {len(files)} from {label_type}...")

    for f in files:
        try:
            name = os.path.basename(f).replace('_processed.csv', '').replace('.csv', '')
            final_score = default_score
            
            if label_type == "Augmented":
                match = re.search(r'FMA_(\d+)', name)
                if match: final_score = int(match.group(1))
            elif label_type == "Real Stroke" and score_map:
                final_score = score_map.get(name)
                if final_score is None:
                    final_score = score_map.get(name.split('_')[0])
                if final_score is None: final_score = 30 
            elif label_type == "Real Healthy":
                final_score = 66

            if final_score is None: continue 

            df = pd.read_csv(f)
            data = df.select_dtypes(include=[np.number]).values
            
            if len(data) == 0: continue
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

def plot_projection(ax, X_proj, y, labels, title, markers):
    df_plot = pd.DataFrame({
        'D1': X_proj[:, 0],
        'D2': X_proj[:, 1],
        'Score': y,
        'Type': labels
    })
    
    sns.scatterplot(
        data=df_plot, x='D1', y='D2', hue='Score', style='Type',
        markers=markers, palette='viridis', s=60, alpha=0.7, edgecolor='black', ax=ax
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')

def main():
    score_map = load_score_map()
    
    h_X, h_y, h_t = load_from_dir(REAL_HEALTHY_DIR, "Real Healthy", default_score=66)
    s_X, s_y, s_t = load_from_dir(REAL_STROKE_DIR, "Real Stroke", score_map=score_map)
    a_X, a_y, a_t = load_from_dir(AUGMENTED_DIR, "Augmented")

    X = np.array(h_X + s_X + a_X)
    y = np.array(h_y + s_y + a_y)
    labels = np.array(h_t + s_t + a_t)

    if len(X) == 0: return

    # 1. Standardize (CRITICAL for t-SNE/UMAP)
    print("Standardizing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Setup Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(left=0.15) # Make room for checkboxes
    
    markers = {"Augmented": "o", "Real Healthy": "P", "Real Stroke": "X"}
    datasets = ["Augmented", "Real Healthy", "Real Stroke"]
    
    # Store scatter objects for toggling
    # Structure: { 'Augmented': [sc_pca, sc_tsne, sc_umap], ... }
    scatter_groups = {ds: [] for ds in datasets}

    # Helper to plot and store handles
    def plot_projection_interactive(ax, X_proj, y, labels, title):
        df_plot = pd.DataFrame({
            'D1': X_proj[:, 0],
            'D2': X_proj[:, 1],
            'Score': y,
            'Type': labels
        })
        
        norm = plt.Normalize(y.min(), y.max())
        cmap = plt.cm.viridis
        
        for ds in datasets:
            subset = df_plot[df_plot['Type'] == ds]
            if subset.empty: continue
            
            alpha_val = 0.2 if ds == "Augmented" else 0.9
            
            sc = ax.scatter(
                subset['D1'], subset['D2'],
                c=subset['Score'], cmap=cmap, norm=norm,
                marker=markers.get(ds, 'o'),
                s=60, alpha=alpha_val, edgecolors='black',
                label=ds
            )
            scatter_groups[ds].append(sc)
            
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # 2. PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plot_projection_interactive(axes[0], X_pca, y, labels, "PCA (Linear Global)")

    # 3. t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)
    plot_projection_interactive(axes[1], X_tsne, y, labels, "t-SNE (Non-Linear Local)")

    # 4. UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    plot_projection_interactive(axes[2], X_umap, y, labels, "UMAP (Balanced Manifold)")

    # --- Checkboxes ---
    from matplotlib.widgets import CheckButtons
    rax = plt.axes([0.02, 0.4, 0.10, 0.15]) # Left side
    visibility = [True] * len(datasets)
    check = CheckButtons(rax, datasets, visibility)

    def func(label):
        # Toggle visibility for this dataset across ALL 3 plots
        for sc in scatter_groups[label]:
            sc.set_visible(not sc.get_visible())
        plt.draw()

    check.on_clicked(func)

    plt.tight_layout(rect=[0.12, 0, 1, 1]) # Adjust layout to not overlap checkboxes
    save_path = os.path.join(BASE_DIR, "output/manifold_comparison_full.png")
    plt.savefig(save_path, dpi=300)
    print(f"Comparison saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()