import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Constants ---
INPUT_DIM = 15
CONDITION_DIM = 1
HIDDEN_DIM = 256
LATENT_DIM = 32
SEQ_LEN = 100

# =============================================================================
# STAGE 0: BASELINE (Standard LSTM CVAE + Concatenation)
# =============================================================================

class EncoderBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM + CONDITION_DIM, HIDDEN_DIM // 2,
                           num_layers=2, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x, c):
        # Concatenate condition at every time step
        c_expanded = c.unsqueeze(1).repeat(1, x.size(1), 1)
        inputs = torch.cat([x, c_expanded], dim=2)
        _, (hidden, _) = self.lstm(inputs)
        # Combine bidirectional hidden states
        pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

class DecoderBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_start = nn.Linear(LATENT_DIM + CONDITION_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=2, batch_first=True)
        self.fc_out = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, z, c, seq_len):
        latent_input = torch.cat([z, c], dim=1)
        hidden_start = F.gelu(self.fc_start(latent_input))
        lstm_input = hidden_start.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.lstm(lstm_input)
        return self.fc_out(output)

class MotionCVAE_Stage0(nn.Module):
    """Standard LSTM CVAE with simple concatenation."""
    def __init__(self, cond_drop_prob=0.0): # cond_drop_prob ignored for Stage 0
        super().__init__()
        self.encoder = EncoderBase()
        self.decoder = DecoderBase()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, c, x.size(1))
        return recon_x, mu, logvar

    def inference(self, c, seq_len=SEQ_LEN, guidance_scale=1.0):
        # No CFG in Stage 0
        with torch.no_grad():
            z = torch.randn(c.size(0), LATENT_DIM).to(c.device)
            return self.decoder(z, c, seq_len)

# =============================================================================
# STAGE 1: CFG (Stage 0 + Classifier-Free Guidance support)
# =============================================================================

class MotionCVAE_Stage1(MotionCVAE_Stage0):
    """Baseline + Classifier-Free Guidance (CFG)."""
    def __init__(self, cond_drop_prob=0.1):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob

    def forward(self, x, c):
        if self.training:
            drop_mask = torch.rand(c.size(0), 1, device=c.device) < self.cond_drop_prob
            c = c * (~drop_mask).float()
        
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, c, x.size(1))
        return recon_x, mu, logvar

    def inference(self, c, seq_len=SEQ_LEN, guidance_scale=2.0):
        with torch.no_grad():
            z = torch.randn(c.size(0), LATENT_DIM).to(c.device)
            if guidance_scale == 1.0:
                return self.decoder(z, c, seq_len)
            
            c_null = torch.zeros_like(c)
            out_cond = self.decoder(z, c, seq_len)
            out_uncond = self.decoder(z, c_null, seq_len)
            return out_uncond + guidance_scale * (out_cond - out_uncond)

# =============================================================================
# STAGE 2: FiLM (Stage 1 + Feature-wise Linear Modulation)
# =============================================================================

class FiLMLayer(nn.Module):
    def __init__(self, condition_dim, hidden_dim):
        super().__init__()
        self.scale = nn.Linear(condition_dim, hidden_dim)
        self.shift = nn.Linear(condition_dim, hidden_dim)

    def forward(self, x, c):
        # x: (B, L, H) or (B, H), c: (B, C)
        s = self.scale(c)
        sh = self.shift(c)
        if x.dim() == 3:
            s = s.unsqueeze(1)
            sh = sh.unsqueeze(1)
        return x * s + sh

class EncoderFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM // 2,
                           num_layers=2, batch_first=True, bidirectional=True)
        self.film = FiLMLayer(CONDITION_DIM, HIDDEN_DIM)
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x, c):
        lstm_out, (hidden, _) = self.lstm(x) # No concatenation here
        pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)
        modulated = self.film(pooled, c) # Modulate after LSTM
        return self.fc_mu(modulated), self.fc_logvar(modulated)

class DecoderFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_start = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.film1 = FiLMLayer(CONDITION_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=2, batch_first=True)
        self.film2 = FiLMLayer(CONDITION_DIM, HIDDEN_DIM)
        self.fc_out = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, z, c, seq_len):
        h = F.gelu(self.fc_start(z))
        h = self.film1(h, c) # Modulate latent projection
        lstm_input = h.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.lstm(lstm_input)
        output = self.film2(output, c) # Modulate LSTM output
        return self.fc_out(output)

class MotionCVAE_Stage2(MotionCVAE_Stage1):
    """Stage 1 + FiLM Conditioning (instead of concatenation)."""
    def __init__(self, cond_drop_prob=0.1):
        super().__init__(cond_drop_prob)
        self.encoder = EncoderFiLM()
        self.decoder = DecoderFiLM()

# =============================================================================
# STAGE 3: OPTIMIZED (Stage 2 + Residual Skip Connection)
# =============================================================================

class DecoderOptimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_start = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.film1 = FiLMLayer(CONDITION_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=2, batch_first=True)
        self.film2 = FiLMLayer(CONDITION_DIM, HIDDEN_DIM)
        # Residual projection
        self.res_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc_out = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, z, c, seq_len):
        h = F.gelu(self.fc_start(z))
        h = self.film1(h, c)
        lstm_input = h.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.lstm(lstm_input)
        output = self.film2(output, c)
        # Simple residual connection
        output = F.gelu(self.res_proj(output) + lstm_input)
        return self.fc_out(output)

class MotionCVAE_Stage3(MotionCVAE_Stage2):
    """Stage 2 + Residual Skip Connection in Decoder."""
    def __init__(self, cond_drop_prob=0.1):
        super().__init__(cond_drop_prob)
        self.decoder = DecoderOptimized()

# Mapping for the training script
STAGE_MODELS = {
    0: MotionCVAE_Stage0,
    1: MotionCVAE_Stage1,
    2: MotionCVAE_Stage2,
    3: MotionCVAE_Stage3
}
