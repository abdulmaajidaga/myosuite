"""
===============================================================================
FILE: viz_plots.py
===============================================================================
Plotting and Analysis logic for the MyoSuite Dashboard.
Refactored for robustness and better visualization.
"""

import os
import io
import base64
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import umap
from scipy import stats
import mujoco

# Import config and utils
from viz_utils import *

# --- HELPER: Save Matplotlib to Base64 ---
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{data}"

# --- CORE: Manifold Visualization (Rewritten) ---
def plot_manifolds(X, y, filenames, title):
    """
    Generates a 1x3 Plotly subplot comparing PCA, t-SNE, and UMAP. 
    
    Args:
        X (np.array): Feature matrix (n_samples, n_features).
        y (list or np.array): Labels (categorical strings or continuous floats).
        filenames (list): Tooltip labels.
        title (str): Main chart title.
        
    Returns:
        plotly.graph_objects.Figure: The interactive figure.
    """
    # 1. Safety Checks & Data Prep
    if X is None or len(X) < 3:
        print(f"⚠️ Not enough data to plot manifolds for {title} (n={len(X) if X is not None else 0})")
        return None

    # Ensure numeric
    X = np.array(X, dtype=float)

    # Handle NaNs (Impute with 0)
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_clean = imputer.fit_transform(X)
    
    # Standard Scaling (Crucial for PCA/t-SNE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # 2. Dimensionality Reduction
    print(f"   Running Reductions for '{title}' (n={len(X_scaled)})...")
    
    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    # t-SNE
    # Perplexity should be < n_samples. Default 30 is too high for small N.
    perp = min(30, max(2, len(X_scaled) // 3))
    tsne = TSNE(n_components=3, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)
    
    # UMAP
    # n_neighbors should be small for local structure
    n_neigh = min(15, max(2, len(X_scaled) // 2))
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neigh, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # 3. Visualization Setup
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            f"PCA (Exp. Var: {sum(var_exp)*100:.1f}%)", 
            f"t-SNE (Perp: {perp})", 
            f"UMAP (Neighbors: {n_neigh})"
        ),
        horizontal_spacing=0.02
    )
    
    # 4. Color Logic
    is_categorical = isinstance(y[0], str)
    
    if is_categorical:
        # Categorical: Stroke vs Healthy
        # We manually split traces to have a proper legend
        unique_labels = sorted(list(set(y)))
        # Define palette
        palette = {'Healthy': '#3498db', 'Stroke': '#e74c3c'}
        
        for label in unique_labels:
            indices = [i for i, val in enumerate(y) if val == label]
            color = palette.get(label, '#95a5a6') # Fallback gray
            
            # Common marker props
            marker_props = dict(size=5, color=color, opacity=0.9, line=dict(width=0.5, color='white'))
            
            # Add Traces
            # PCA
            fig.add_trace(go.Scatter3d(
                x=X_pca[indices, 0], y=X_pca[indices, 1], z=X_pca[indices, 2],
                mode='markers', marker=marker_props,
                name=f"PCA - {label}", legendgroup=label,
                text=[filenames[i] for i in indices], hoverinfo='text+name'
            ), row=1, col=1)
            
            # t-SNE
            fig.add_trace(go.Scatter3d(
                x=X_tsne[indices, 0], y=X_tsne[indices, 1], z=X_tsne[indices, 2],
                mode='markers', marker=marker_props,
                name=f"t-SNE - {label}", legendgroup=label, showlegend=False,
                text=[filenames[i] for i in indices], hoverinfo='text+name'
            ), row=1, col=2)
            
            # UMAP
            fig.add_trace(go.Scatter3d(
                x=X_umap[indices, 0], y=X_umap[indices, 1], z=X_umap[indices, 2],
                mode='markers', marker=marker_props,
                name=f"UMAP - {label}", legendgroup=label, showlegend=False,
                text=[filenames[i] for i in indices], hoverinfo='text+name'
            ), row=1, col=3)
            
    else:
        # Continuous: FMA Scores
        # Single trace with colorscale
        marker_props = dict(
            size=6, 
            color=y, 
            colorscale='Viridis', 
            showscale=True,
            colorbar=dict(title="FMA Score", x=1.02, len=0.5),
            opacity=0.9
        )
        
        hover_text = [f"File: {f}<br>FMA: {s}" for f, s in zip(filenames, y)]
        
        fig.add_trace(go.Scatter3d(
            x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
            mode='markers', marker=marker_props,
            name='PCA', text=hover_text, hoverinfo='text'
        ), row=1, col=1)
        
        # Hide colorbar for others to avoid duplicates
        marker_props_nc = marker_props.copy()
        marker_props_nc['showscale'] = False
        
        fig.add_trace(go.Scatter3d(
            x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2],
            mode='markers', marker=marker_props_nc,
            name='t-SNE', text=hover_text, hoverinfo='text'
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter3d(
            x=X_umap[:, 0], y=X_umap[:, 1], z=X_umap[:, 2],
            mode='markers', marker=marker_props_nc,
            name='UMAP', text=hover_text, hoverinfo='text'
        ), row=1, col=3)

    # 5. Final Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.95),
        height=600,
        margin=dict(l=10, r=10, b=10, t=60),
        legend=dict(x=0.01, y=0.9),
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        scene2=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3'),
        scene3=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3'),
        template='plotly_white'
    )
    
    return fig

# --- 1. MODEL QUALITY (Matplotlib) ---
def plot_model_quality(X_raw, y_scores):
    if not os.path.exists(MODEL_PATH): return None
    
    # Impute before model use
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_clean = imputer.fit_transform(X_raw)
    
    model = joblib.load(MODEL_PATH)
    pca = model['pca']
    regressors = model['regressors']
    scaler = model['score_scaler']
    
    X_projected = pca.transform(X_clean)
    
    # Synth Data for Trend Lines
    synth_scores = np.linspace(0, 70, 100).reshape(-1, 1)
    synth_scores_norm = scaler.transform(synth_scores)
    synth_pca = np.zeros((100, len(regressors)))
    for i, regr in enumerate(regressors):
        synth_pca[:, i] = regr.predict(synth_scores_norm)

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(18, 10))
    
    # PC1 vs PC2
    ax1 = plt.subplot(2, 3, 1)
    sc = ax1.scatter(X_projected[:, 0], X_projected[:, 1], c=y_scores, cmap='viridis', s=80, edgecolors='k')
    plt.colorbar(sc, ax=ax1, label='FMA Score')
    ax1.plot(synth_pca[:, 0], synth_pca[:, 1], 'r--', lw=2, label='Learned Path')
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.legend()
    ax1.set_title("Latent Space (PC1 vs PC2)")

    # Trends
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_scores, X_projected[:, 0], alpha=0.6)
    ax2.plot(synth_scores, synth_pca[:, 0], 'r--')
    ax2.set_xlabel("FMA"); ax2.set_ylabel("PC1"); ax2.set_title("PC1 Trend")

    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(y_scores, X_projected[:, 1], alpha=0.6)
    ax3.plot(synth_scores, synth_pca[:, 1], 'r--')
    ax3.set_xlabel("FMA"); ax3.set_ylabel("PC2"); ax3.set_title("PC2 Trend")

    # Variance
    ax4 = plt.subplot(2, 1, 2)
    indices = range(1, len(pca.explained_variance_ratio_) + 1)
    ax4.bar(indices, pca.explained_variance_ratio_ * 100, color='skyblue')
    ax4.plot(indices, np.cumsum(pca.explained_variance_ratio_ * 100), 'r-o', label='Cumulative')
    ax4.set_xlabel("Component"); ax4.set_ylabel("Variance (%)"); ax4.legend()
    ax4.set_title("PCA Variance Explained")

    plt.tight_layout()
    return fig_to_base64(fig)

# --- 2. TRAJECTORIES (Matplotlib) ---
def plot_trajectories(df_scores, all_mot_dfs):
    # Setup MuJoCo for FK
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_XML)
        data = mujoco.MjData(model)
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'V_Wrist')
    except: return None, None

    groups = {'Severe (<30)': (0, 30), 'Moderate (30-50)': (30, 50), 'Mild (>50)': (50, 100)}
    colors = {'Severe (<30)': 'red', 'Moderate (30-50)': 'orange', 'Mild (>50)': 'green'}
    
    # Compute 3D Paths
    paths = []
    scores = []
    
    # Pre-calc 3D paths
    for idx, row in df_scores.iterrows():
        fname = row['filename']
        if fname not in all_mot_dfs: continue
        mot_df = all_mot_dfs[fname] # Should be DataFrame (not resampled array) 
        
        # Resample DF for FK
        indices = np.linspace(0, len(mot_df)-1, FIXED_FRAMES).astype(int)
        df_res = mot_df.iloc[indices].reset_index(drop=True)
        
        path_3d = []
        for i in range(len(df_res)):
            for col in df_res.columns:
                if col == 'time': continue
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, col)
                if jid != -1:
                    data.qpos[model.jnt_qposadr[jid]] = df_res.loc[i, col]
            mujoco.mj_forward(model, data)
            if wrist_id != -1:
                path_3d.append(data.site_xpos[wrist_id].copy())
        
        paths.append(np.array(path_3d))
        scores.append(row['fma_score'])

    # Plot 1: Joint Profiles (Mean +/- Std)
    fig1, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.flatten()
    joints = ['shoulder_elv', 'elbow_flexion', 'pro_sup', 'shoulder_rot']
    
    for i, joint in enumerate(joints):
        ax = axes[i]
        for gname, (low, high) in groups.items():
            # Gather trajs
            trajs = []
            for j, row in df_scores.iterrows():
                if low <= row['fma_score'] < high and row['filename'] in all_mot_dfs:
                    df = all_mot_dfs[row['filename']]
                    if joint in df:
                        trajs.append(signal.resample(df[joint].values, FIXED_FRAMES))
            
            if trajs:
                arr = np.array(trajs)
                mean = np.mean(arr, axis=0)
                std = np.std(arr, axis=0)
                x = np.linspace(0, 100, FIXED_FRAMES)
                ax.plot(x, mean, label=gname, color=colors[gname])
                ax.fill_between(x, mean-std, mean+std, color=colors[gname], alpha=0.1)
        
        ax.set_title(joint); ax.legend(fontsize='small')

    plt.tight_layout()
    img_joints = fig_to_base64(fig1)

    # Plot 2: 3D Paths
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection='3d')
    for p, s in zip(paths, scores):
        c = 'gray'
        for gname, (low, high) in groups.items():
            if low <= s < high: c = colors[gname]
        ax.plot(p[:,0], p[:,1], p[:,2], color=c, alpha=0.4)
    
    ax.set_title("3D Wrist Trajectories")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    img_3d = fig_to_base64(fig2)
    
    return img_joints, img_3d

# --- 3. METRICS (Matplotlib) ---
def plot_metrics(df_scores, all_mot_dfs):
    # Compute metrics
    metrics_list = []
    
    # We need MuJoCo for path efficiency
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_XML)
        data = mujoco.MjData(model)
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'V_Wrist')
    except: return None

    for idx, row in df_scores.iterrows():
        fname = row['filename']
        if fname not in all_mot_dfs: continue
        mot_df = all_mot_dfs[fname]
        
        m = {'fma': row['fma_score']}
        
        # ROM
        for j in ['shoulder_elv', 'elbow_flexion', 'pro_sup']:
            if j in mot_df: m[f'ROM_{j}'] = mot_df[j].max() - mot_df[j].min()
            
        # Duration
        m['Duration'] = mot_df['time'].iloc[-1] - mot_df['time'].iloc[0]
        
        # Velocity/Efficiency (Requires FK loop)
        positions = []
        for i in range(len(mot_df)):
            for col in mot_df.columns:
                if col=='time': continue
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, col)
                if jid!=-1: data.qpos[model.jnt_qposadr[jid]] = mot_df.iloc[i][col]
            mujoco.mj_forward(model, data)
            if wrist_id!=-1: positions.append(data.site_xpos[wrist_id].copy())
        
        pos = np.array(positions)
        seg_lens = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        path_len = np.sum(seg_lens)
        disp = np.linalg.norm(pos[-1] - pos[0])
        m['Path_Eff'] = path_len / (disp + 1e-6)
        
        dt = np.diff(mot_df['time'].values)
        vels = seg_lens / (dt + 1e-6)
        m['Peak_Vel'] = np.max(vels)
        
        metrics_list.append(m)
        
    df_m = pd.DataFrame(metrics_list)
    cols = [c for c in df_m.columns if c != 'fma']
    
    n_metrics = len(cols)
    rows = (n_metrics + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4*rows))
    axes = axes.flatten()
    
    for i, metric in enumerate(cols):
        sns.regplot(data=df_m, x='fma', y=metric, ax=axes[i], line_kws={'color':'red'})
        r, p = stats.pearsonr(df_m['fma'], df_m[metric])
        axes[i].set_title(f"{metric}\nR={r:.2f}")
    
    plt.tight_layout()
    return fig_to_base64(fig)

# --- 4. SIMILARITY (Seaborn) ---
def plot_similarity(X_flat, filenames, scores):
    # Impute first
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_clean = imputer.fit_transform(X_flat)

    # Compute Distance Matrix
    n = len(X_clean)
    dist_mat = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            d = np.linalg.norm(X_clean[i] - X_clean[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            
    labels = [f"{f} ({s})" for f, s in zip(filenames, scores)]
    df_dist = pd.DataFrame(dist_mat, index=labels, columns=labels)
    
    fig = plt.figure(figsize=(12, 12))
    sns.clustermap(df_dist, method='average', cmap="viridis_r", figsize=(12, 12),
                   cbar_kws={'label': 'Distance'})
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close('all')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{data}"