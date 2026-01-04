import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
import glob
import re
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler # <--- ADDED
import joblib # <--- ADDED

# --- Configuration ---
# Hyperparameters
INPUT_DIM = 12       
CONDITION_DIM = 1    
HIDDEN_DIM = 128     
LATENT_DIM = 16      
SEQ_LEN = 100        
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(root_data, "data/kinematic/Augmented") 
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "output/cvae/cvae.pth")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "output/cvae/scaler.pkl") # <--- ADDED

# --- 1. Dataset Class ---

class MotionDataset(Dataset):
    def __init__(self, root_dir, scaler, seq_len=100): # <--- Added scaler arg
        self.files = []
        self.seq_len = seq_len
        self.scaler = scaler # <--- Store scaler
        
        pattern = os.path.join(root_dir, "**", "FMA_*.csv")
        all_csvs = glob.glob(pattern, recursive=True)
        
        print(f"Scanning {len(all_csvs)} files...")
        
        for f in all_csvs:
            try:
                score_match = re.search(r'FMA_(\d+)', os.path.basename(f))
                if score_match:
                    score = int(score_match.group(1))
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
        data = df.select_dtypes(include=[np.number]).values
        
        # 1. Resample FIRST (to fix shape)
        if len(data) != self.seq_len:
            data = resample(data, self.seq_len)
            
        # 2. Normalize SECOND (to fix scale)
        # This converts ~500.0 (mm) down to ~1.0 (std dev)
        data = self.scaler.transform(data) # <--- THE FIX
            
        motion_seq = torch.FloatTensor(data) 
        score_val = torch.FloatTensor([score]) 
        
        return motion_seq, score_val

# --- 2. The CVAE Model (Unchanged) ---

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, c, x.size(1))
        return recon_x, mu, logvar
    
    def inference(self, c, seq_len=SEQ_LEN):
        with torch.no_grad():
            z = torch.randn(c.size(0), LATENT_DIM).to(c.device)
            generated = self.decoder(z, c, seq_len)
            return generated

# --- 3. Training Logic ---

def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train():
    print(f"--- Initializing CVAE Training on {DEVICE} ---")
    
    # --- STEP 1: Fit Scaler on ALL data first ---
    print("Fitting Scaler (Normalization)...")
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "FMA_*.csv"), recursive=True)
    all_data_list = []
    
    # We load a sample (or all) to fit the scaler
    # Since dataset is small (~2000 files), loading all is fine and safest
    for f in all_files:
        try:
            df = pd.read_csv(f)
            data = df.select_dtypes(include=[np.number]).values
            if len(data) > 0: all_data_list.append(data)
        except: pass
        
    full_array = np.vstack(all_data_list)
    
    scaler = StandardScaler()
    scaler.fit(full_array)
    
    # Save Scaler for later generation!
    if not os.path.exists(os.path.dirname(SCALER_SAVE_PATH)):
        os.makedirs(os.path.dirname(SCALER_SAVE_PATH))
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")
    # ---------------------------------------------
    
    # Load Data (Passing the scaler)
    full_dataset = MotionDataset(DATA_DIR, scaler, SEQ_LEN)
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

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# --- 4. Generation Helper ---

def generate_motion(model_path, target_fma_score):
    """
    Generates CSV, automatically un-normalizing back to mm.
    """
    model = MotionCVAE().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    norm_score = target_fma_score / 66.0
    c = torch.FloatTensor([[norm_score]]).to(DEVICE)
    
    # Generate (This is in Normalized Space, e.g., -1.5 to 1.5)
    generated_tensor = model.inference(c, seq_len=SEQ_LEN)
    data_np = generated_tensor.squeeze(0).cpu().numpy()
    
    # --- STEP 2: Un-Normalize back to Millimeters ---
    scaler = joblib.load(SCALER_SAVE_PATH)
    data_denorm = scaler.inverse_transform(data_np)
    # ------------------------------------------------
    
    cols = [
        'Sh_x','Sh_y','Sh_z',
        'El_x','El_y','El_z',
        'Wr_x','Wr_y','Wr_z',
        'WrVec_x','WrVec_y','WrVec_z'
    ]
    
    df = pd.DataFrame(data_denorm, columns=cols)
    return df

if __name__ == "__main__":
    train()
    
    print("\nTesting Generation for FMA 66...")
    df_gen = generate_motion(MODEL_SAVE_PATH, 66)
    print(df_gen.head())
    df_gen.to_csv("test_cvae_generation.csv", index=False)