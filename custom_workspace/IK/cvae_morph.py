import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
import glob
import re

# --- Configuration ---
# Hyperparameters
INPUT_DIM = 12       # Sh(3) + El(3) + Wr(3) + WrVec(3)
CONDITION_DIM = 1    # The FMA Score
HIDDEN_DIM = 128     # Neurons in LSTM layers
LATENT_DIM = 16      # Size of the "style" vector (z)
SEQ_LEN = 100        # Standardized length (we must resample all inputs to this)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(root_data, "data/kinematic/augmented_smooth") # Training on your SMOOTHED data
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "output/cvae_fma_model.pth")

# --- 1. Dataset Class ---

class MotionDataset(Dataset):
    def __init__(self, root_dir, seq_len=100):
        self.files = []
        self.seq_len = seq_len
        
        # Recursively find all FMA_*.csv files in subfolders
        # We look for "FMA_XX.csv" pattern
        pattern = os.path.join(root_dir, "**", "FMA_*.csv")
        all_csvs = glob.glob(pattern, recursive=True)
        
        # Also include original Stroke/Healthy if they are in a standard folder?
        # For now, let's focus on the augmented dataset which contains everything
        
        print(f"Scanning {len(all_csvs)} files...")
        
        for f in all_csvs:
            try:
                # Extract score from filename
                score_match = re.search(r'FMA_(\d+)', os.path.basename(f))
                if score_match:
                    score = int(score_match.group(1))
                    # Normalize score to 0-1 range for better training stability
                    norm_score = (score - 0) / 66.0 
                    self.files.append((f, norm_score))
            except:
                continue
                
        print(f"Loaded {len(self.files)} valid motion sequences.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, score = self.files[idx]
        
        # Load Data
        df = pd.read_csv(path)
        
        # Ensure we only have the 12 numeric columns
        # (Assuming your augmented data is clean, but let's be safe)
        data = df.select_dtypes(include=[np.number]).values
        
        # Resample to fixed SEQ_LEN
        if len(data) != self.seq_len:
            from scipy.signal import resample
            data = resample(data, self.seq_len)
            
        # Convert to Tensor
        motion_seq = torch.FloatTensor(data) # Shape: [100, 12]
        score_val = torch.FloatTensor([score]) # Shape: [1]
        
        return motion_seq, score_val

# --- 2. The CVAE Model ---

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Input size is Motion Features + Condition (broadcasted)
        self.lstm = nn.LSTM(INPUT_DIM + CONDITION_DIM, HIDDEN_DIM, batch_first=True)
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x, c):
        # x: [Batch, Seq, Feat], c: [Batch, 1]
        
        # Expand condition to match sequence length
        # c_expanded: [Batch, Seq, 1]
        c_expanded = c.unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Concatenate motion + condition
        inputs = torch.cat([x, c_expanded], dim=2)
        
        # LSTM encoding
        _, (hidden, _) = self.lstm(inputs)
        
        # We take the last hidden state
        last_hidden = hidden[-1]
        
        # Predict distribution parameters
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
        # z: [Batch, Latent], c: [Batch, 1]
        
        # Combine Latent + Condition
        latent_input = torch.cat([z, c], dim=1) # [Batch, Latent+Cond]
        
        # Map to hidden state size
        hidden_start = self.fc_start(latent_input)
        
        # Prepare inputs for LSTM
        # We repeat the hidden vector for every timestep as input
        # Shape: [Batch, Seq, Hidden]
        lstm_input = hidden_start.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        output, _ = self.lstm(lstm_input)
        
        # Map back to motion space (12 dims)
        recon_motion = self.fc_out(output)
        
        return recon_motion

class MotionCVAE(nn.Module):
    def __init__(self):
        super(MotionCVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        # Encode
        mu, logvar = self.encoder(x, c)
        # Sample
        z = self.reparameterize(mu, logvar)
        # Decode
        recon_x = self.decoder(z, c, x.size(1))
        return recon_x, mu, logvar
    
    def inference(self, c, seq_len=SEQ_LEN):
        # Used for generation ONLY (no input motion needed)
        with torch.no_grad():
            # Sample random z from normal distribution
            z = torch.randn(c.size(0), LATENT_DIM).to(c.device)
            # Decode
            generated = self.decoder(z, c, seq_len)
            return generated

# --- 3. Training Logic ---

def loss_function(recon_x, x, mu, logvar):
    # 1. Reconstruction Loss (MSE) - Did the motion look right?
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # 2. KL Divergence - Did we learn a clean latent space?
    # formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def train():
    print(f"--- Initializing CVAE Training on {DEVICE} ---")
    
    # Load Data
    full_dataset = MotionDataset(DATA_DIR, SEQ_LEN)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    # Init Model
    model = MotionCVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (motion, score) in enumerate(train_loader):
            motion = motion.to(DEVICE)
            score = score.to(DEVICE)
            
            optimizer.zero_grad()
            
            recon_motion, mu, logvar = model(motion, score)
            
            loss = loss_function(recon_motion, motion, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save
    if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH))
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# --- 4. Generation Helper ---

def generate_motion(model_path, target_fma_score):
    """
    Stand-alone function to generate a CSV from a trained model.
    """
    # Load Model
    model = MotionCVAE().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Prepare Condition
    # Remember we normalized score (0-66 -> 0-1) during training!
    norm_score = target_fma_score / 66.0
    c = torch.FloatTensor([[norm_score]]).to(DEVICE)
    
    # Generate
    generated_tensor = model.inference(c, seq_len=SEQ_LEN)
    
    # Convert to DataFrame
    data_np = generated_tensor.squeeze(0).cpu().numpy()
    
    # Define Columns (Must match your training data order!)
    cols = [
        'Sh_x','Sh_y','Sh_z',
        'El_x','El_y','El_z',
        'Wr_x','Wr_y','Wr_z',
        'WrVec_x','WrVec_y','WrVec_z'
    ]
    
    df = pd.DataFrame(data_np, columns=cols)
    return df

if __name__ == "__main__":
    # If run directly, train the model
    train()
    
    # Test Generation
    print("\nTesting Generation for FMA 66...")
    df_gen = generate_motion(MODEL_SAVE_PATH, 66)
    print(df_gen.head())
    df_gen.to_csv("test_cvae_generation.csv", index=False)