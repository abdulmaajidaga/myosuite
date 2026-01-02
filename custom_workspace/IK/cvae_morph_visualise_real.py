import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import os
import sys

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output/cvae_fma_model.pth")
OUTPUT_CSV = os.path.join(BASE_DIR, "output/generated_realistic.csv")

# Model Params (Must match training)
INPUT_DIM = 12       
CONDITION_DIM = 1    
HIDDEN_DIM = 128     
LATENT_DIM = 16      
SEQ_LEN = 100        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Model Definitions (Required for Loading) ---
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(INPUT_DIM + CONDITION_DIM, HIDDEN_DIM, batch_first=True)
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x, c):
        c_expanded = c.unsqueeze(1).repeat(1, x.size(1), 1)
        inputs = torch.cat([x, c_expanded], dim=2)
        _, (hidden, _) = self.lstm(inputs)
        last_hidden = hidden[-1]
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc_start = nn.Linear(LATENT_DIM + CONDITION_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.fc_out = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, z, c, seq_len):
        latent_input = torch.cat([z, c], dim=1)
        hidden_start = self.fc_start(latent_input)
        lstm_input = hidden_start.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.lstm(lstm_input)
        recon_motion = self.fc_out(output)
        return recon_motion

class MotionCVAE(nn.Module):
    def __init__(self):
        super(MotionCVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def inference(self, c, seq_len=SEQ_LEN):
        with torch.no_grad():
            z = torch.randn(c.size(0), LATENT_DIM).to(c.device)
            generated = self.decoder.forward(z, c, seq_len)
            return generated

# --- 2. Biomechanical Post-Processing ---

def apply_temporal_scaling(df, fma_score):
    """
    Adjusts duration based on FMA.
    Healthy (66) = Fast (~2 seconds / 100 frames)
    Severe (20) = Slow (~6 seconds / 300 frames)
    """
    # Calculate target frames (Linear mapping)
    # Score 66 -> 100 frames
    # Score 20 -> 300 frames
    # Formula: 100 + (66 - score) * 4.3
    target_len = int(100 + (66 - fma_score) * 4.5)
    
    print(f"   -> Time Scaling: Resampling 100 frames to {target_len} frames (Slower).")
    
    new_data = {}
    for col in df.columns:
        new_data[col] = signal.resample(df[col], target_len)
        
    return pd.DataFrame(new_data, columns=df.columns)

def apply_biological_tremor(df, fma_score):
    """
    Injects 3-6Hz oscillation (tremor) inversely proportional to FMA score.
    """
    if fma_score > 60: return df # Healthy enough, no tremor needed
    
    n_frames = len(df)
    
    # 1. Magnitude: Lower score = Higher Tremor
    # FMA 20 -> 0.015 magnitude
    # FMA 60 -> 0.000 magnitude
    tremor_mag = max(0, (60 - fma_score) * 0.0001)
    
    print(f"   -> Tremor Injection: Adding {tremor_mag:.4f} magnitude noise.")

    # We simulate this with sine waves + random noise
    t = np.linspace(0, n_frames / 50.0, n_frames) # Assume 50Hz playback
    
    # Construct tremor signal (Sum of two sine waves for organic feel)
    tremor_x = np.sin(2 * np.pi * 3 * t) * tremor_mag + np.random.normal(0, tremor_mag/2, n_frames)
    tremor_y = np.sin(2 * np.pi * 4 * t) * tremor_mag + np.random.normal(0, tremor_mag/2, n_frames)
    tremor_z = np.sin(2 * np.pi * 5 * t) * tremor_mag + np.random.normal(0, tremor_mag/2, n_frames)
    
    df_noisy = df.copy()
    # Apply to Wrist and Elbow (Shoulder is usually more stable)
    for part in ['El', 'Wr']:
        df_noisy[f'{part}_x'] += tremor_x
        df_noisy[f'{part}_y'] += tremor_y
        df_noisy[f'{part}_z'] += tremor_z
        
    return df_noisy

# --- 3. Generation Logic ---

def generate_realistic_motion(target_score):
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return None

    print(f"--- Generating Realistic Motion for FMA {target_score} ---")
    
    # 1. AI Generation (The "Base" Shape)
    model = MotionCVAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    norm_score = target_score / 66.0
    c = torch.FloatTensor([[norm_score]]).to(DEVICE)
    output = model.inference(c, SEQ_LEN).squeeze(0).cpu().numpy()
    
    cols = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
    df_base = pd.DataFrame(output, columns=cols)
    
    # 2. Apply Time Scaling (Slow down low scores)
    df_timed = apply_temporal_scaling(df_base, target_score)
    
    # 3. Apply Tremor (Add shake to low scores)
    df_final = apply_biological_tremor(df_timed, target_score)
    
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")
    return df_final

# --- 4. Animation ---

def animate(df, score):
    print("--- Starting Animation ---")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    all_x = pd.concat([df['Sh_x'], df['El_x'], df['Wr_x']])
    all_y = pd.concat([df['Sh_y'], df['El_y'], df['Wr_y']])
    all_z = pd.concat([df['Sh_z'], df['El_z'], df['Wr_z']])
    pad = 0.1
    xlim = (all_x.min()-pad, all_x.max()+pad)
    ylim = (all_y.min()-pad, all_y.max()+pad)
    zlim = (all_z.min()-pad, all_z.max()+pad)

    line_arm, = ax.plot([], [], [], 'o-', lw=4, color='blue', label=f'FMA {score} (Realistic)')
    line_trace, = ax.plot([], [], [], '-', lw=1, color='orange', alpha=0.5)
    
    ax.set_title(f"Simulation: FMA {score}\n(Note Speed & Jitter)")
    ax.legend()

    def update(frame):
        row = df.iloc[frame]
        xs = [row['Sh_x'], row['El_x'], row['Wr_x']]
        ys = [row['Sh_y'], row['El_y'], row['Wr_y']]
        zs = [row['Sh_z'], row['El_z'], row['Wr_z']]
        
        line_arm.set_data(xs, ys)
        line_arm.set_3d_properties(zs)
        
        hist = df.iloc[:frame+1]
        line_trace.set_data(hist['Wr_x'], hist['Wr_y'])
        line_trace.set_3d_properties(hist['Wr_z'])
        
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        return line_arm, line_trace

    # Interval=20ms is standard for 50fps playback
    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=20, blit=False)
    plt.show()

if __name__ == "__main__":
    target = 25 # Default to low score to see effects
    if len(sys.argv) > 1: target = int(sys.argv[1])
        
    df_out = generate_realistic_motion(target)
    if df_out is not None:
        animate(df_out, target)