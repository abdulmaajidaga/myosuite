"""
CVAE v2: FiLM conditioning + temporal dynamics restoration.

Changes from v2-initial (over-smoothing fix):
- Encoder: attention-weighted pooling replaces final-hidden pooling
  (learns FMA-dependent temporal patterns instead of discarding mid-sequence)
- FiLM applied BEFORE pooling so attention is condition-aware
- Decoder: ResidualBlock + TemporalConvBlock after BiLSTM
  (per-frame sharpening + cross-timestep pattern detection)
- Decoder: dual FiLM — pre-LSTM (spatial shape) + post-LSTM/pre-refinement (temporal
  magnitude). Refinement blocks see FMA-conditioned features, so TemporalConvBlock
  (kernel=5) learns FMA-dependent smoothing: wide bells for healthy, jerky for stroke.

Preserved from v2-initial:
- Sinusoidal PE, deterministic eval, no CFG
- All hyperparameters configurable via constructor kwargs
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
NUM_HEADS = 4        # Unused in v2, kept for import compat
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


class ResidualBlock(nn.Module):
    """Per-frame pointwise MLP with residual connection + LayerNorm.

    Gives the decoder capacity to amplify/dampen features at individual
    timesteps independently (e.g., sharp velocity spike at frame 37
    without affecting frame 38).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.layer_norm(x + residual)


class TemporalConvBlock(nn.Module):
    """Bottleneck 1D convolution with residual connection.

    Operates across timesteps to detect/amplify local temporal patterns
    like rapid direction changes that characterize jerky stroke motion.
    Architecture: hidden_dim -> bottleneck -> bottleneck -> hidden_dim (kernel=5).
    """

    def __init__(self, hidden_dim, bottleneck=64):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, bottleneck, kernel_size=1)
        self.conv2 = nn.Conv1d(bottleneck, bottleneck, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(bottleneck, hidden_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        residual = x
        # Conv1d expects (batch, channels, seq_len)
        h = x.transpose(1, 2)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = self.dropout(h)
        h = self.conv3(h)
        # Back to (batch, seq_len, hidden_dim)
        h = h.transpose(1, 2)
        return self.layer_norm(h + residual)


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

    FiLM is applied BEFORE pooling (on 3D sequence), so the learned attention
    weights are FMA-dependent — e.g., attend to jerk frames for stroke,
    smooth peaks for healthy.
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


class Decoder(nn.Module):
    """BiLSTM decoder with sinusoidal PE, dual FiLM, and temporal refinement.

    Dual FiLM: pre-LSTM (spatial shape) + post-LSTM/pre-refinement (temporal
    magnitude). Refinement blocks then operate on FMA-conditioned features —
    TemporalConvBlock (last before output) learns FMA-dependent smoothing:
    wide smooth bells for healthy, jerky peaks for stroke.
    """

    def __init__(self, input_dim=INPUT_DIM, cond_dim=CONDITION_DIM,
                 hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc_expand = nn.Linear(latent_dim, hidden_dim)
        self.film_pre = FiLM(cond_dim, hidden_dim)   # before LSTM: spatial shape
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=2,
                            batch_first=True, dropout=0.1, bidirectional=True)
        self.film_post = FiLM(cond_dim, hidden_dim)  # after LSTM: temporal magnitude

        # Refinement blocks operate on FMA-conditioned features
        self.res_block = ResidualBlock(hidden_dim)
        self.temporal_conv = TemporalConvBlock(hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, z, c, seq_len):
        # z: (batch, latent_dim), c: (batch, cond_dim)
        h = F.gelu(self.fc_expand(z))  # (batch, hidden_dim)
        h = h.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_dim)

        # Add sinusoidal positional encoding
        pe = sinusoidal_encoding(seq_len, self.hidden_dim, h.device)
        h = h + pe.unsqueeze(0)  # broadcast over batch

        # Pre-LSTM FiLM: controls spatial shape (range, trunk pattern)
        h = self.film_pre(h, c)

        # BiLSTM decoding
        out, _ = self.lstm(h)

        # Post-LSTM FiLM: sets temporal magnitude before refinement
        out = self.film_post(out, c)

        # Refinement on FMA-conditioned features: TempConv learns
        # smooth bells for high FMA, jerky peaks for low FMA
        out = self.res_block(out)
        out = self.temporal_conv(out)

        return self.fc_out(out)


class MotionCVAE(nn.Module):
    """Conditional VAE with FiLM conditioning for motion generation."""

    def __init__(self, input_dim=INPUT_DIM, cond_dim=CONDITION_DIM,
                 hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
                 cond_drop_prob=0.0):
        super().__init__()
        # cond_drop_prob accepted for backward compat but unused
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, cond_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, cond_dim, hidden_dim, latent_dim)

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
