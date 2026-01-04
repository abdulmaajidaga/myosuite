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
import joblib  # <--- REQUIRED for loading the scaler

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output/cvae/cvae.pth")
SCALER_PATH = os.path.join(BASE_DIR, "output/cvae/scaler.pkl") 
OUTPUT_CSV = os.path.join(BASE_DIR, "output/cvae/generated_motion.csv")

# Model Params (Must match training)
INPUT_DIM = 12       
CONDITION_DIM = 1    
HIDDEN_DIM = 128     
LATENT_DIM = 16      
SEQ_LEN = 100        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Model Definitions ---
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

def lock_shoulder(df):
    """
    Forces the shoulder to stay fixed at its starting position (First Frame).
    """
    fixed_sh_x = df['Sh_x'].iloc[0]
    fixed_sh_y = df['Sh_y'].iloc[0]
    fixed_sh_z = df['Sh_z'].iloc[0]
    
    df['Sh_x'] = fixed_sh_x
    df['Sh_y'] = fixed_sh_y
    df['Sh_z'] = fixed_sh_z
    return df

def apply_temporal_scaling(df, fma_score):
    # Score 66 -> 100 frames, Score 20 -> 300 frames
    target_len = int(100 + (66 - fma_score) * 4.5)
    print(f"   -> Time Scaling: Resampling to {target_len} frames.")
    
    new_data = {}
    for col in df.columns:
        new_data[col] = signal.resample(df[col], target_len)
    return pd.DataFrame(new_data, columns=df.columns)

def apply_biological_tremor(df, fma_score):
    """
    Injects tremor noise. Lower FMA = Higher Magnitude.
    """
    if fma_score > 60: return df 
    n_frames = len(df)
    
    # Magnitude
    tremor_mag = max(0, (60 - fma_score) * 0.5) 
    print(f"   -> Tremor: Injecting magnitude {tremor_mag:.2f}")

    t = np.linspace(0, n_frames / 50.0, n_frames)
    
    # 3-6Hz Oscillation + Random Noise
    noise = lambda: np.sin(2 * np.pi * 4 * t) * tremor_mag + np.random.normal(0, tremor_mag/2, n_frames)
    
    df_noisy = df.copy()
    for part in ['El', 'Wr']:
        df_noisy[f'{part}_x'] += noise()
        df_noisy[f'{part}_y'] += noise()
        df_noisy[f'{part}_z'] += noise()
        
    return df_noisy

# --- 3. Generation Logic ---

def generate_realistic_motion(target_score):
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run training first.")
        return None
    
    if not os.path.exists(SCALER_PATH):
        print("Scaler not found. Run training first.")
        return None

    print(f"--- Generating Realistic Motion for FMA {target_score} ---")
    
    # 1. AI Generation
    model = MotionCVAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    norm_score = target_score / 66.0
    c = torch.FloatTensor([[norm_score]]).to(DEVICE)
    
    # Raw Output (Normalized -1 to 1)
    output_norm = model.inference(c, SEQ_LEN).squeeze(0).cpu().numpy()
    
    # 2. UN-NORMALIZE (The Fix)
    scaler = joblib.load(SCALER_PATH)
    output_mm = scaler.inverse_transform(output_norm) 
    
    cols = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
    df_base = pd.DataFrame(output_mm, columns=cols)
    
    # 3. Post-Processing Pipeline
    df_locked = lock_shoulder(df_base)          
    # df_timed = apply_temporal_scaling(df_locked, target_score)
    # df_final = apply_biological_tremor(df_timed, target_score)
    df_final = df_locked  # Temporal scaling and tremor disabled for cleaner output
    # df_final = df_timed  # Tremor disabled for cleaner output
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")
    return df_final

# --- 4. Animation ---

def animate(df, score):
    print("--- Starting Animation ---")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine plot limits based on data
    all_vals = df[['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z']].values.flatten()
    min_v, max_v = all_vals.min(), all_vals.max()
    
    # The Arm (Blue Line)
    line_arm, = ax.plot([], [], [], 'o-', lw=4, color='blue', label=f'FMA {score}')
    
    # The Trace (Yellow Path) - RESTORED
    line_trace, = ax.plot([], [], [], '-', lw=1, color='orange', alpha=0.6)
    
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_zlim(min_v, max_v)
    ax.set_title(f"Simulation: FMA {score}")
    ax.legend()

    def update(frame):
        row = df.iloc[frame]
        xs = [row['Sh_x'], row['El_x'], row['Wr_x']]
        ys = [row['Sh_y'], row['El_y'], row['Wr_y']]
        zs = [row['Sh_z'], row['El_z'], row['Wr_z']]
        
        # Update Arm
        line_arm.set_data(xs, ys)
        line_arm.set_3d_properties(zs)
        
        # Update Trace (Yellow line following Wrist)
        hist = df.iloc[:frame+1]
        line_trace.set_data(hist['Wr_x'], hist['Wr_y'])
        line_trace.set_3d_properties(hist['Wr_z'])
        
        return line_arm, line_trace

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=20, blit=False)
    plt.show()

if __name__ == "__main__":
    target = 25
    if len(sys.argv) > 1: target = int(sys.argv[1])
        
    df_out = generate_realistic_motion(target)
    if df_out is not None:
        animate(df_out, target)