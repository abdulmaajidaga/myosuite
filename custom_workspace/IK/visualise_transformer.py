import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
import os
import sys
import pickle

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from transformer_vae import TransformerVAE
except ImportError:
    print("Error: Could not import TransformerVAE from transformer_vae.py")
    sys.exit(1)

# --- CONFIG ---
LATENT_DIM = 64
SEQ_LEN = 100
INPUT_CHANNELS = 24

def generate(model, fma_score):
    model.eval()
    with torch.no_grad():
        # 1. Sample Latent Z
        z = torch.randn(1, LATENT_DIM)
        
        # 2. Prepare Condition
        c = torch.tensor([[fma_score / 66.0]])
        
        # 3. Decode
        # Output shape is (1, 100, 24)
        recon = model.decode(z, c) 
        
        return recon.squeeze().numpy() # (100, 24)

def animate(traj, fma, pca, scaler):
    # Inverse Coupled Scaling
    # traj is (100, 24)
    # 0-11: Position (12)
    # 12-23: Velocity (12)
    
    scaled_pos = traj[:, :12]
    
    # Inverse Scale Position
    pca_pos = scaler.inverse_transform(scaled_pos)
    
    # Inverse PCA -> 63 (Original Joint Data)
    raw_traj = pca.inverse_transform(pca_pos)
    
    # Filter for smoother visualization
    smooth_traj = np.zeros_like(raw_traj)
    for i in range(raw_traj.shape[1]):
        smooth_traj[:, i] = savgol_filter(raw_traj[:, i], 11, 3)
        
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Transformer VAE Generation (FMA {fma})")
    
    frames = smooth_traj.reshape(100, -1)
    
    # Indices for Wrist, Elbow, Shoulder
    # Based on standard column mapping (Wra, Elb, Sho)
    wra = frames[:, 0:3]
    elb = frames[:, 21:24]
    sho = frames[:, 48:51]
    
    all_p = np.vstack([wra, elb, sho])
    ax.set_xlim([all_p[:,0].min(), all_p[:,0].max()])
    ax.set_ylim([all_p[:,1].min(), all_p[:,1].max()])
    ax.set_zlim([all_p[:,2].min(), all_p[:,2].max()])
    
    line, = ax.plot([],[],[], 'b-', lw=4)
    pts, = ax.plot([],[],[], 'ko')
    
    # Text annotation for phases
    phase_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    
    def update(i):
        x = [sho[i,0], elb[i,0], wra[i,0]]
        y = [sho[i,1], elb[i,1], wra[i,1]]
        z = [sho[i,2], elb[i,2], wra[i,2]]
        line.set_data_3d(x, y, z)
        pts.set_data_3d(x, y, z)
        
        # Simple Phase Annotation
        if i < 15: phase = "Start"
        elif i < 45: phase = "Reach"
        elif i < 65: phase = "Drink"
        elif i < 90: phase = "Return"
        else: phase = "End"
        
        phase_text.set_text(f"Frame: {i}/100\nPhase: {phase}")
        
        return line, pts, phase_text
        
    ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    base = os.path.dirname(__file__)
    output_dir = os.path.join(base, "output")
    
    model_path = os.path.join(output_dir, "transformer_vae.pth")
    pca_path = os.path.join(output_dir, "pca_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run 'python IK/transformer_vae.py' to train the model first.")
    else:
        # Load Model
        model = TransformerVAE()
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print("Transformer Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
            
        # Load Preprocessors
        if not os.path.exists(pca_path) or not os.path.exists(scaler_path):
             print("Error: PCA or Scaler model missing. Train the model first.")
             sys.exit(1)
             
        with open(pca_path, 'rb') as f: pca = pickle.load(f)
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        
        # User Choice
        print("\nSelect FMA Score to visualize:")
        print("1. Low Function (FMA 20)")
        print("2. Moderate Function (FMA 45)")
        print("3. Healthy/High Function (FMA 66)")
        
        choice = input("Enter 1, 2, or 3 (default 3): ").strip()
        
        if choice == '1': score = 20
        elif choice == '2': score = 45
        else: score = 66
            
        print(f"Generating motion for FMA {score}...")
        traj = generate(model, score)
        animate(traj, score, pca, scaler)
