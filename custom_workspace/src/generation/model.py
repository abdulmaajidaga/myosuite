"""
Shared CVAE model architecture for motion generation.

Contains: SelfAttention, Encoder, Decoder, ResidualBlock, MotionCVAE.

All consumer scripts should import from here instead of defining their own copy:
    from src.generation.model import MotionCVAE
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
NUM_HEADS = 4
SEQ_LEN = 100


class SelfAttention(nn.Module):
    """Multi-head self-attention for temporal dependencies."""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(context)

        # Residual connection + layer norm
        return self.layer_norm(x + output)


class Encoder(nn.Module):
    """Bidirectional LSTM encoder with self-attention."""
    def __init__(self):
        super().__init__()
        # Bidirectional LSTM
        self.lstm = nn.LSTM(INPUT_DIM + CONDITION_DIM, HIDDEN_DIM // 2,
                           num_layers=2, batch_first=True, dropout=0.1,
                           bidirectional=True)

        # Self-attention layer
        self.attention = SelfAttention(HIDDEN_DIM, NUM_HEADS)

        # Output projection
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x, c):
        # Expand condition to sequence length
        c_expanded = c.unsqueeze(1).repeat(1, x.size(1), 1)
        inputs = torch.cat([x, c_expanded], dim=2)

        # BiLSTM encoding
        lstm_out, (hidden, _) = self.lstm(inputs)

        # Apply self-attention
        attn_out = self.attention(lstm_out)

        # Use mean pooling over sequence for final representation
        pooled = attn_out.mean(dim=1)

        return self.fc_mu(pooled), self.fc_logvar(pooled)


class ResidualBlock(nn.Module):
    """Residual block for decoder."""
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


class Decoder(nn.Module):
    """LSTM decoder with residual connections and attention."""
    def __init__(self):
        super().__init__()
        # Initial projection from latent space
        self.fc_start = nn.Linear(LATENT_DIM + CONDITION_DIM, HIDDEN_DIM)

        # Bidirectional LSTM for decoding
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM // 2,
                           num_layers=2, batch_first=True, dropout=0.1,
                           bidirectional=True)

        # Residual refinement blocks
        self.res_block1 = ResidualBlock(HIDDEN_DIM)
        self.res_block2 = ResidualBlock(HIDDEN_DIM)

        # Output projection
        self.fc_out = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, z, c, seq_len):
        # Project latent to hidden
        latent_input = torch.cat([z, c], dim=1)
        hidden_start = F.gelu(self.fc_start(latent_input))

        # Expand to sequence
        lstm_input = hidden_start.unsqueeze(1).repeat(1, seq_len, 1)

        # LSTM decoding
        output, _ = self.lstm(lstm_input)

        # Residual refinement
        output = self.res_block1(output)
        output = self.res_block2(output)

        # Project to output dimensions
        return self.fc_out(output)


class MotionCVAE(nn.Module):
    """Conditional VAE with attention, residual connections, and classifier-free guidance."""
    def __init__(self, cond_drop_prob=0.1):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.cond_drop_prob = cond_drop_prob  # Probability of dropping condition during training

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c, drop_cond=None):
        # Classifier-free guidance: randomly drop condition during training
        if self.training and drop_cond is None:
            drop_mask = torch.rand(c.size(0), 1, device=c.device) < self.cond_drop_prob
            c = c * (~drop_mask).float()  # Zero out condition for dropped samples
        elif drop_cond is True:
            c = torch.zeros_like(c)

        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, c, x.size(1))
        return recon_x, mu, logvar

    def inference(self, c, seq_len=SEQ_LEN, guidance_scale=2.0):
        """Generate with classifier-free guidance for stronger conditioning."""
        with torch.no_grad():
            z = torch.randn(c.size(0), LATENT_DIM).to(c.device)

            if guidance_scale == 1.0:
                # No guidance, just conditional generation
                return self.decoder(z, c, seq_len)

            # Classifier-free guidance: interpolate between conditional and unconditional
            c_null = torch.zeros_like(c)

            # Generate conditional output
            out_cond = self.decoder(z, c, seq_len)

            # Generate unconditional output (with same z for consistency)
            out_uncond = self.decoder(z, c_null, seq_len)

            # Guided output = unconditional + scale * (conditional - unconditional)
            out_guided = out_uncond + guidance_scale * (out_cond - out_uncond)

            return out_guided
