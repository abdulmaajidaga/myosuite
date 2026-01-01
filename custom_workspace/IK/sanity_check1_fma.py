import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- CONFIG ---
SEQ_LEN = 100
INPUT_DIM = 12
LATENT_DIM = 32
BATCH_SIZE = 8
EPOCHS = 600
LR = 0.001

# --- 1. HELPER: LOAD SCORES ---
def load_scores_map(score_csv_path):
    """
    Reads score.csv and returns a dict: {'SB01': 45, 'SB02': 22, ...}
    Assumes CSV format: PatientID, FMA_Score (or similar)
    """
    score_map = {}
    if not os.path.exists(score_csv_path):
        print(f"WARNING: Score file not found at {score_csv_path}")
        return score_map
        
    try:
        df = pd.read_csv(score_csv_path)
        # Try to intelligently find the ID and Score columns
        # Taking the first column as ID and usually the last or specific 'FMA' as score
        id_col = df.columns[0] 
        # Look for a column that likely contains the score
        score_col = [c for c in df.columns if 'score' in c.lower() or 'fma' in c.lower()]
        score_col = score_col[0] if score_col else df.columns[1]
        
        print(f"Loading scores using columns: ID='{id_col}', Score='{score_col}'")
        
        for _, row in df.iterrows():
            # Clean ID: remove spaces, ensure string
            pid = str(row[id_col]).strip().replace('.mot', '')
            score = float(row[score_col])
            score_map[pid] = score
            
        print(f"Loaded {len(score_map)} clinical scores.")
    except Exception as e:
        print(f"Error reading score CSV: {e}")
        
    return score_map

def get_id_from_filename(filename):
    # Assumes filename format like "S5_12_1_processed.csv"
    # Returns "S5_12_1"
    base = os.path.basename(filename)
    return base.replace('_processed.csv', '')

# --- 2. DATASET ---
class KinematicDataset(Dataset):
    def __init__(self, folder, score_map=None, is_healthy=False, mean=None, std=None):
        self.data = []
        self.scores = []
        self.filenames = []
        
        files = glob.glob(os.path.join(folder, "*_processed.csv"))
        
        valid_samples = 0
        missing_scores = 0
        
        for fpath in files:
            try:
                df = pd.read_csv(fpath)
                data_np = df.values.astype(np.float32)
                
                if data_np.shape == (SEQ_LEN, INPUT_DIM):
                    # GET SCORE
                    if is_healthy:
                        score = 66.0 # Max score for healthy
                    else:
                        pid = get_id_from_filename(fpath)
                        if pid in score_map:
                            score = score_map[pid]
                        else:
                            # If we can't find a score, skip this file or assign NaN? 
                            # Let's skip to keep data clean for correlation.
                            missing_scores += 1
                            continue 
                    
                    self.data.append(data_np)
                    self.scores.append(score)
                    self.filenames.append(fpath)
                    valid_samples += 1
            except: pass
            
        self.data = np.array(self.data)
        self.scores = np.array(self.scores)
        
        # Stats handling
        if valid_samples > 0:
            if mean is None:
                self.mean = np.mean(self.data, axis=(0,1))
                self.std = np.std(self.data, axis=(0,1)) + 1e-6
                print(f"[{'HEALTHY' if is_healthy else 'STROKE'}] Loaded {valid_samples} samples.")
            else:
                self.mean = mean
                self.std = std
                print(f"[{'HEALTHY' if is_healthy else 'STROKE'}] Loaded {valid_samples} samples. (Using Healthy Stats)")
                
            if missing_scores > 0:
                print(f"   -> Skipped {missing_scores} files due to missing FMA scores in CSV.")
        else:
            print(f"CRITICAL: No valid data found in {folder}")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        raw = self.data[idx]
        norm = (raw - self.mean) / self.std
        score = self.scores[idx]
        return torch.FloatTensor(norm), torch.FloatTensor([score])

# --- 3. MODEL (Standard AE) ---
class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 64
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True), num_layers=2)
        self.dec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True), num_layers=2)
        self.in_proj = nn.Linear(INPUT_DIM, 64)
        self.out_proj = nn.Linear(64, INPUT_DIM)
        self.to_lat = nn.Linear(SEQ_LEN*64, LATENT_DIM)
        self.from_lat = nn.Linear(LATENT_DIM, SEQ_LEN*64)
        
    def forward(self, x):
        # Simplify for brevity (PosEncoding omitted for brevity, usually helps but AE works without too)
        x_emb = self.in_proj(x)
        lat = self.to_lat(self.enc(x_emb).flatten(1))
        rec = self.dec(self.from_lat(lat).view(-1, SEQ_LEN, 64))
        return self.out_proj(rec)

# --- 4. MAIN ---
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_data = os.path.dirname(base_dir) # Go up one level to 'custom_workspace'
    
    # 1. PATHS
    healthy_dir = os.path.join(root_data, "data/kinematic/Healthy/processed")
    impaired_dir = os.path.join(root_data, "data/kinematic/Stroke/processed")
    score_path = os.path.join(base_dir, "output/scores.csv")
    
    # 2. Load Scores
    score_map = load_scores_map(score_path)
    
    # 3. Load Datasets
    # Healthy (Score = 66 auto-assigned)
    healthy_ds = KinematicDataset(healthy_dir, is_healthy=True)
    
    # Stroke (Scores matched from CSV)
    # Note: We pass healthy stats to normalize stroke data
    stroke_ds = KinematicDataset(impaired_dir, score_map=score_map, is_healthy=False, 
                                 mean=healthy_ds.mean, std=healthy_ds.std)
    
    # Combine for evaluation later
    full_eval_loader = DataLoader(torch.utils.data.ConcatDataset([healthy_ds, stroke_ds]), 
                                  batch_size=1, shuffle=False)
    
    # 4. Train (Only on Healthy)
    loader = DataLoader(healthy_ds, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleTransformer()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print("\n--- Training on Healthy Data ---")
    for epoch in range(EPOCHS):
        model.train()
        for x, _ in loader: # Ignore score during training
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
        if (epoch+1)%100==0: print(f"Epoch {epoch+1}: Loss {loss.item():.5f}")

    # 5. EVALUATION: CORRELATION ANALYSIS
    print("\n--- Clinical Correlation Analysis ---")
    model.eval()
    
    mse_list = []
    fma_list = []
    colors = [] # 0 for healthy, 1 for stroke
    
    with torch.no_grad():
        for x, score in full_eval_loader:
            # Reconstruct
            recon = model(x)
            
            # Calculate MSE (Unnormalized for better scale)
            # We de-normalize to get real millimeter error
            real_x = (x.numpy() * healthy_ds.std) + healthy_ds.mean
            real_rec = (recon.numpy() * healthy_ds.std) + healthy_ds.mean
            
            mse = np.mean((real_x - real_rec) ** 2)
            fma = score.item()
            
            mse_list.append(mse)
            fma_list.append(fma)
            
            # Color code: If FMA is 66, it's likely healthy
            colors.append('green' if fma >= 65.9 else 'red')

    mse_arr = np.array(mse_list)
    fma_arr = np.array(fma_list)
    
    # 6. Statistics
    # Filter out the healthy 66s for correlation if you only want to see correlation within Stroke
    # But usually, we want the whole spectrum.
    
    # Calculate Correlation (Pearson)
    corr, _ = pearsonr(mse_arr, fma_arr)
    print(f"\nCorrelation (MSE vs FMA): {corr:.4f}")
    print("Interpretation: Closer to -1.0 is better (High Error = Low Score)")

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(mse_arr, fma_arr, c=colors, alpha=0.7, s=80, edgecolors='k')
    
    # Regression Line (Trend)
    z = np.polyfit(mse_arr, fma_arr, 1)
    p = np.poly1d(z)
    plt.plot(mse_arr, p(mse_arr), "b--", alpha=0.5, label=f"Trend Line (corr={corr:.2f})")
    
    plt.title(f"AI Biomarker Validation: Model Error vs Clinical Score\nCorrelation: {corr:.3f}")
    plt.xlabel("Reconstruction Error (MSE) - Lower is Better")
    plt.ylabel("FMA-UE Score (Max 66) - Higher is Better")
    plt.legend(["Trend", "Healthy Data", "Stroke Data"])
    
    # Create a dummy legend for points
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Healthy (66)'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Stroke (<66)')]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()