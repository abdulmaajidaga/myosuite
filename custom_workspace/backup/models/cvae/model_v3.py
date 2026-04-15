"""
CVAE v3: TimeVAE-style temporal convolution decoder + spectral smoothness.

Changes from V2:
- Decoder replaced: BiLSTM → dual-branch temporal convolution (TrendConvStack + DetailConvStack)
  Eliminates boundary artifacts — Conv1d with symmetric padding treats all frames identically,
  no hidden state warm-up/cool-down.
- Trend branch (large kernels 15/9/5) captures low-frequency movement shape
- Detail branch (small kernels 3/3/3) captures high-frequency local patterns
- Dual FiLM preserved: pre-branch (spatial shape) + post-fusion (temporal magnitude)

Preserved from V2:
- Encoder: BiLSTM + FiLM + Bahdanau attention pooling (identical)
- Sinusoidal PE, deterministic eval, no CFG
- API: MotionCVAE(cond_drop_prob=...), .inference(c, seq_len, guidance_scale)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Default hyperparameters (must match training) ---
INPUT_DIM = 15       # 12 arm + 3 trunk
CONDITION_DIM = 1
HIDDEN_DIM = 256
LATENT_DIM = 32
NUM_HEADS = 4        # Unused in v3, kept for import compat
SEQ_LEN = 100


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: condition -> (gamma, beta) that scale+shift features."""

    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # gamma and beta
        )

    def forward(self, x, c):
        # x: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        # c: (batch, cond_dim)
        gamma_beta = self.net(c)  # (batch, hidden_dim * 2)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (batch, hidden_dim)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)  # (batch, 1, hidden_dim)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


def sinusoidal_encoding(seq_len, dim, device):
    """Generate sinusoidal positional encoding."""
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (seq_len, dim)


class Encoder(nn.Module):
    """BiLSTM encoder with FiLM conditioning and attention-weighted pooling.

    Identical to V2 encoder — FiLM applied BEFORE pooling so attention
    weights are FMA-dependent.
    """

    def __init__(self, input_dim=INPUT_DIM, cond_dim=CONDITION_DIM,
                 hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=2,
                            batch_first=True, dropout=0.1, bidirectional=True)
        self.film = FiLM(cond_dim, hidden_dim)

        # Bahdanau-style attention for pooling
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        # x: (batch, seq_len, input_dim), c: (batch, cond_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # FiLM conditioning on full sequence BEFORE pooling
        conditioned = self.film(lstm_out, c)  # (batch, seq_len, hidden_dim)

        # Attention-weighted pooling (Bahdanau-style)
        energy = torch.tanh(self.attn_proj(conditioned))  # (batch, seq_len, hidden_dim)
        scores = self.attn_score(energy).squeeze(-1)       # (batch, seq_len)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        pooled = (conditioned * weights).sum(dim=1)         # (batch, hidden_dim)

        return self.fc_mu(pooled), self.fc_logvar(pooled)


class TrendConvStack(nn.Module):
    """Large-kernel convolution stack for low-frequency movement shape.

    Conv1d stack: 256→128(k=15) → 128(k=9) → 256(k=5)
    Symmetric (same) padding, GELU activation, LayerNorm after each conv.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, 128, kernel_size=15, padding=7)
        self.norm1 = nn.LayerNorm(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=9, padding=4)
        self.norm2 = nn.LayerNorm(128)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=5, padding=2)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        h = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)

        h = self.conv1(h)                    # (batch, 128, seq_len)
        h = F.gelu(self.norm1(h.transpose(1, 2))).transpose(1, 2)

        h = self.conv2(h)                    # (batch, 128, seq_len)
        h = F.gelu(self.norm2(h.transpose(1, 2))).transpose(1, 2)

        h = self.conv3(h)                    # (batch, hidden_dim, seq_len)
        h = F.gelu(self.norm3(h.transpose(1, 2)))  # (batch, seq_len, hidden_dim)

        return h


class DetailConvStack(nn.Module):
    """Small-kernel bottleneck convolution stack for high-frequency local patterns.

    Conv1d stack: 256→64(k=3) → 64(k=3) → 256(k=3)
    Bottleneck design keeps parameter count low.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(64)
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        h = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)

        h = self.conv1(h)                    # (batch, 64, seq_len)
        h = F.gelu(self.norm1(h.transpose(1, 2))).transpose(1, 2)

        h = self.conv2(h)                    # (batch, 64, seq_len)
        h = F.gelu(self.norm2(h.transpose(1, 2))).transpose(1, 2)

        h = self.conv3(h)                    # (batch, hidden_dim, seq_len)
        h = F.gelu(self.norm3(h.transpose(1, 2)))  # (batch, seq_len, hidden_dim)

        return h


class TimeVAEDecoder(nn.Module):
    """TimeVAE-style decoder with dual-branch temporal convolution.

    Architecture:
      z(32) → FC(256) → expand to (B, seq_len, 256) → +sinusoidal PE → FiLM_pre(c)
           ├─ TrendConvStack: large kernels for low-freq shape
           └─ DetailConvStack: small kernels for high-freq detail
           → cat(512) → Fusion(512→256) → FiLM_post(c) → FC(15)
    """

    def __init__(self, input_dim=INPUT_DIM, cond_dim=CONDITION_DIM,
                 hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc_expand = nn.Linear(latent_dim, hidden_dim)
        self.film_pre = FiLM(cond_dim, hidden_dim)   # before branches: spatial shape

        self.trend = TrendConvStack(hidden_dim)
        self.detail = DetailConvStack(hidden_dim)

        # Fusion: concatenated trend+detail (2*hidden_dim) → hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.film_post = FiLM(cond_dim, hidden_dim)  # after fusion: temporal magnitude
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, z, c, seq_len):
        # z: (batch, latent_dim), c: (batch, cond_dim)
        h = F.gelu(self.fc_expand(z))  # (batch, hidden_dim)
        h = h.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_dim)

        # Add sinusoidal positional encoding
        pe = sinusoidal_encoding(seq_len, self.hidden_dim, h.device)
        h = h + pe.unsqueeze(0)  # broadcast over batch

        # Pre-branch FiLM: controls spatial shape (range, trunk pattern)
        h = self.film_pre(h, c)

        # Dual-branch temporal convolution
        trend_out = self.trend(h)   # (batch, seq_len, hidden_dim)
        detail_out = self.detail(h)  # (batch, seq_len, hidden_dim)

        # Fuse branches
        fused = torch.cat([trend_out, detail_out], dim=-1)  # (batch, seq_len, 2*hidden_dim)
        fused = self.fusion(fused)  # (batch, seq_len, hidden_dim)

        # Post-fusion FiLM: sets temporal magnitude
        out = self.film_post(fused, c)

        return self.fc_out(out)


class MotionCVAE(nn.Module):
    """Conditional VAE with TimeVAE-style decoder for motion generation."""

    def __init__(self, input_dim=INPUT_DIM, cond_dim=CONDITION_DIM,
                 hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
                 cond_drop_prob=0.0):
        super().__init__()
        # cond_drop_prob accepted for backward compat but unused
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, cond_dim, hidden_dim, latent_dim)
        self.decoder = TimeVAEDecoder(input_dim, cond_dim, hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic at eval time

    def forward(self, x, c, drop_cond=None):
        # drop_cond accepted for backward compat but unused
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, c, x.size(1))
        return recon_x, mu, logvar

    def inference(self, c, seq_len=SEQ_LEN, guidance_scale=1.0):
        """Generate motion for given condition. guidance_scale kept for compat but unused."""
        with torch.no_grad():
            z = torch.randn(c.size(0), self.latent_dim, device=c.device)
            return self.decoder(z, c, seq_len)
