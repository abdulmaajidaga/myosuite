import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output/cvae_fma_model.pth")
OUTPUT_CSV = os.path.join(BASE_DIR, "output/generated_motion.csv")

# Model Hyperparameters (MUST MATCH TRAINING EXACTLY)
INPUT_DIM = 12       
CONDITION_DIM = 1    
HIDDEN_DIM = 128     
LATENT_DIM = 16      
SEQ_LEN = 100        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Define Model Architecture (Required to load weights) ---
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
            generated = self.decoder.forward(z, c, seq_len) # Fixed call
            return generated

# --- 2. Generation Logic ---

def generate(target_score):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return None

    print(f"--- Generating Motion for FMA {target_score} ---")
    
    # Init Model
    model = MotionCVAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Prepare Input (Normalize Score 0-66 -> 0-1)
    norm_score = target_score / 66.0
    c = torch.FloatTensor([[norm_score]]).to(DEVICE)
    
    # Generate
    output_tensor = model.inference(c, SEQ_LEN)
    data = output_tensor.squeeze(0).cpu().numpy()
    
    # Save to CSV
    cols = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")
    
    return df

# --- 3. Animation Logic ---

def animate(df, score):
    print("--- Starting Animation ---")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calc limits for camera
    all_x = pd.concat([df['Sh_x'], df['El_x'], df['Wr_x']])
    all_y = pd.concat([df['Sh_y'], df['El_y'], df['Wr_y']])
    all_z = pd.concat([df['Sh_z'], df['El_z'], df['Wr_z']])
    pad = 0.1
    xlim = (all_x.min()-pad, all_x.max()+pad)
    ylim = (all_y.min()-pad, all_y.max()+pad)
    zlim = (all_z.min()-pad, all_z.max()+pad)

    # Elements
    line_arm, = ax.plot([], [], [], 'o-', lw=4, color='blue', label=f'AI Gen (FMA {score})')
    line_trace, = ax.plot([], [], [], '-', lw=1, color='orange', alpha=0.5)
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"Generated Motion: FMA {score}")
    ax.legend()

    def update(frame):
        row = df.iloc[frame]
        
        # Skeleton Points
        xs = [row['Sh_x'], row['El_x'], row['Wr_x']]
        ys = [row['Sh_y'], row['El_y'], row['Wr_y']]
        zs = [row['Sh_z'], row['El_z'], row['Wr_z']]
        
        line_arm.set_data(xs, ys)
        line_arm.set_3d_properties(zs)
        
        # Trace
        hist = df.iloc[:frame+1]
        line_trace.set_data(hist['Wr_x'], hist['Wr_y'])
        line_trace.set_3d_properties(hist['Wr_z'])
        
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        return line_arm, line_trace

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=30, blit=False)
    plt.show()

# --- Main ---

if __name__ == "__main__":
    # Default score 45, or take from command line
    target = 45
    if len(sys.argv) > 1:
        target = int(sys.argv[1])
        
    df_gen = generate(target)
    if df_gen is not None:
        animate(df_gen, target)