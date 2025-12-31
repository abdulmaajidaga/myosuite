import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pickle
import os
import sys
import math

# --- CONFIG ---
INPUT_DIM = 12
SEQ_LEN = 100
LATENT_DIM = 64
CONDITION_DIM = 1

# --- MODEL DEFINITION (MUST MATCH TRAINED MODEL EXACTLY) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        
        self.input_proj = nn.Linear(INPUT_DIM + 1, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, SEQ_LEN)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.fc_mu = nn.Linear(SEQ_LEN * self.d_model, LATENT_DIM)
        self.fc_logvar = nn.Linear(SEQ_LEN * self.d_model, LATENT_DIM)
        
        self.decoder_input = nn.Linear(LATENT_DIM + 1, SEQ_LEN * self.d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=4)
        
        self.output_proj = nn.Linear(self.d_model, INPUT_DIM)
        
        # --- NEW LAYER (This was missing) ---
        self.condition_bias = nn.Linear(1, INPUT_DIM)

    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        h_flat = self.decoder_input(z_c)
        h = h_flat.view(-1, SEQ_LEN, self.d_model)
        h = self.pos_encoder(h)
        out_emb = self.transformer_decoder(h)
        
        out = self.output_proj(out_emb)
        
        # --- NEW LOGIC (This was missing) ---
        c_bias = self.condition_bias(c).unsqueeze(1) 
        return out + c_bias

# --- UTILS ---
def load_model_and_scaler(model_path, scaler_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler not found at {scaler_path}")
        sys.exit(1)

    model = TransformerVAE()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except RuntimeError as e:
        print(f"CRITICAL ERROR: {e}")
        print("Model definition mismatch. Ensure this script matches training script exactly.")
        sys.exit(1)
        
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
    return model, scaler_data

def generate_motion(model, scaler_data, fma_score):
    z = torch.randn(1, LATENT_DIM)
    c = torch.tensor([[fma_score / 66.0]])
    
    with torch.no_grad():
        recon = model.decode(z, c)
    
    norm_traj = recon.squeeze().numpy()
    traj = (norm_traj * scaler_data['std']) + scaler_data['mean']
    return traj

def animate_comparison(healthy_traj, stroke_traj):
    fig = plt.figure(figsize=(14, 8))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Generated Healthy (FMA 66)")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Generated Stroke (FMA 20)")
    
    axes = [ax1, ax2]
    trajs = [healthy_traj, stroke_traj]
    lines = []
    
    for ax in axes:
        # Auto-center view
        ax.set_xlim([-400, 400])
        ax.set_ylim([-400, 400])
        ax.set_zlim([-400, 400])
        ax.set_xlabel('X')
        ax.scatter([0], [0], [0], c='k', s=50, label='Chest')
        
        line, = ax.plot([], [], [], 'bo-', lw=3)
        vec_line, = ax.plot([], [], [], 'r-', lw=2)
        lines.append((line, vec_line))

    def update(i):
        for idx, (traj, (line, vec_line)) in enumerate(zip(trajs, lines)):
            row = traj[i]
            sh, el, wr, vec = row[0:3], row[3:6], row[6:9], row[9:12]
            
            # Skeleton
            line.set_data_3d([0, sh[0], el[0], wr[0]], 
                             [0, sh[1], el[1], wr[1]], 
                             [0, sh[2], el[2], wr[2]])
            
            # Vector (Fixed Scale 30)
            vec_end = wr + (vec * 30) 
            vec_line.set_data_3d([wr[0], vec_end[0]], 
                                 [wr[1], vec_end[1]], 
                                 [wr[2], vec_end[2]])
            
        return [l for pair in lines for l in pair]

    ani = FuncAnimation(fig, update, frames=100, interval=30, blit=False)
    plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "output")
    model_path = os.path.join(out_dir, "transformer_skeleton.pth")
    scaler_path = os.path.join(out_dir, "skeleton_scaler.pkl")
    
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    h_traj = generate_motion(model, scaler, 66)
    s_traj = generate_motion(model, scaler, 20)
    animate_comparison(h_traj, s_traj)