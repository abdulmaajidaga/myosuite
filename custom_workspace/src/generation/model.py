"""
model.py — Configurable CVAE for stroke motion generation.

Graduated from test/ sandbox after systematic ablation (Phases A–D).
Best config (D_base): use_film=True, use_cfg=True, use_residual=False

Architecture map:
  Stage 1 equivalent: use_film=False, use_cfg=True,  use_residual=False
  Stage 2 equivalent: use_film=True,  use_cfg=True,  use_residual=False  ← D_base (best)
  Stage 3 equivalent: use_film=True,  use_cfg=True,  use_residual=True
  Stage 3 + TCB:      use_film=True,  use_cfg=True,  use_residual=True,  use_temporal_conv=True
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM     = 15   # 12 arm + 3 trunk markers
CONDITION_DIM = 1    # FMA score (normalised 0–1)
HIDDEN_DIM    = 256
LATENT_DIM    = 32
SEQ_LEN       = 100


# ── Sub-modules ───────────────────────────────────────────────────────────────

class TemporalConvBlock(nn.Module):
    """Bottleneck 1D-CNN with residual + LayerNorm.

    Sits after the LSTM in the decoder and learns FMA-dependent temporal
    shaping. Uses a bottleneck (hidden→64→64→hidden) so parameter count
    stays low. Kernel=5 gives a ±2-frame receptive field.
    """
    def __init__(self, hidden_dim: int = HIDDEN_DIM, bottleneck: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim,  bottleneck, kernel_size=1)
        self.conv2 = nn.Conv1d(bottleneck,  bottleneck, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(bottleneck,  hidden_dim, kernel_size=1)
        self.norm  = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(0.1)

    def forward(self, x):
        # x: (B, T, H)
        residual = x
        x = x.transpose(1, 2)          # → (B, H, T) for Conv1d
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.conv3(x)
        x = x.transpose(1, 2)          # → (B, T, H)
        return self.norm(self.drop(x) + residual)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scales + shifts hidden state by condition."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Linear(CONDITION_DIM, HIDDEN_DIM)
        self.shift = nn.Linear(CONDITION_DIM, HIDDEN_DIM)

    def forward(self, x, c):
        # x: (B, H) or (B, T, H);  c: (B, C)
        s, sh = self.scale(c), self.shift(c)
        if x.dim() == 3:
            s, sh = s.unsqueeze(1), sh.unsqueeze(1)
        return x * s + sh


class Encoder(nn.Module):
    def __init__(self, use_film: bool, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.use_film = use_film
        lstm_in = INPUT_DIM if use_film else INPUT_DIM + CONDITION_DIM
        self.lstm = nn.LSTM(lstm_in, HIDDEN_DIM // 2, num_layers=2,
                            batch_first=True, bidirectional=True)
        if use_film:
            self.film = FiLMLayer()
        self.fc_mu     = nn.Linear(HIDDEN_DIM, latent_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, latent_dim)

    def forward(self, x, c):
        if self.use_film:
            _, (h, _) = self.lstm(x)
        else:
            c_exp = c.unsqueeze(1).expand(-1, x.size(1), -1)
            _, (h, _) = self.lstm(torch.cat([x, c_exp], dim=-1))

        pooled = torch.cat([h[-2], h[-1]], dim=1)   # bidirectional concat
        if self.use_film:
            pooled = self.film(pooled, c)
        return self.fc_mu(pooled), self.fc_logvar(pooled)


class Decoder(nn.Module):
    def __init__(self, use_film: bool, use_residual: bool,
                 use_temporal_conv: bool = False, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.use_film          = use_film
        self.use_residual      = use_residual
        self.use_temporal_conv = use_temporal_conv
        fc_in = latent_dim if use_film else latent_dim + CONDITION_DIM
        self.fc_start = nn.Linear(fc_in, HIDDEN_DIM)
        if use_film:
            self.film1 = FiLMLayer()
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=2, batch_first=True)
        if use_film:
            self.film2 = FiLMLayer()
        if use_residual:
            self.res_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        if use_temporal_conv:
            self.temporal_conv = TemporalConvBlock()
        self.fc_out = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, z, c, seq_len):
        inp = z if self.use_film else torch.cat([z, c], dim=-1)
        h = F.gelu(self.fc_start(inp))
        if self.use_film:
            h = self.film1(h, c)

        lstm_in = h.unsqueeze(1).expand(-1, seq_len, -1)   # broadcast → (B, T, H)
        out, _  = self.lstm(lstm_in)

        if self.use_film:
            out = self.film2(out, c)
        if self.use_residual:
            out = F.gelu(self.res_proj(out) + lstm_in)
        if self.use_temporal_conv:
            out = self.temporal_conv(out)

        return self.fc_out(out)


# ── Main model ────────────────────────────────────────────────────────────────

class MotionCVAE(nn.Module):
    """Configurable CVAE for FMA-conditioned stroke motion generation.

    Config dict keys:
      use_film          (bool, default True)   — FiLM conditioning
      use_cfg           (bool, default True)   — Classifier-free guidance
      use_residual      (bool, default False)  — Residual skip in decoder
      use_temporal_conv (bool, default False)  — TemporalConvBlock after LSTM
      cond_drop_prob  (float, default 0.1)     — CFG dropout probability
      latent_dim       (int,  default 32)      — Latent space size

    Best validated config (D_base, wrist_rho=0.914):
      use_film=True, use_cfg=True, use_residual=False, latent_dim=32
    """

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or {}
        self.use_film          = cfg.get('use_film',          True)
        self.use_residual      = cfg.get('use_residual',      False)
        self.use_cfg           = cfg.get('use_cfg',           True)
        self.use_temporal_conv = cfg.get('use_temporal_conv', False)
        self.cond_drop_prob    = cfg.get('cond_drop_prob',    0.1)
        self.latent_dim        = cfg.get('latent_dim',        LATENT_DIM)

        self.encoder = Encoder(use_film=self.use_film, latent_dim=self.latent_dim)
        self.decoder = Decoder(use_film=self.use_film,
                               use_residual=self.use_residual,
                               use_temporal_conv=self.use_temporal_conv,
                               latent_dim=self.latent_dim)

    def describe(self) -> str:
        parts = []
        if self.use_film: parts.append("FiLM")
        else:             parts.append("Concat")
        if self.use_cfg:  parts.append(f"CFG(p={self.cond_drop_prob})")
        if self.use_residual:      parts.append("Residual")
        if self.use_temporal_conv: parts.append("TCB")
        n = sum(p.numel() for p in self.parameters())
        return f"MotionCVAE[{'·'.join(parts)}] — {n/1e3:.1f}k params"

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, c):
        if self.training and self.use_cfg and self.cond_drop_prob > 0:
            mask = torch.rand(c.size(0), 1, device=c.device) < self.cond_drop_prob
            c = c * (~mask).float()
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, c, x.size(1)), mu, logvar

    def inference(self, c, seq_len: int = SEQ_LEN, guidance_scale: float = 2.0):
        with torch.no_grad():
            z = torch.randn(c.size(0), self.latent_dim, device=c.device)
            if self.use_cfg and guidance_scale > 1.0:
                out_cond   = self.decoder(z, c,                   seq_len)
                out_uncond = self.decoder(z, torch.zeros_like(c), seq_len)
                return out_uncond + guidance_scale * (out_cond - out_uncond)
            return self.decoder(z, c, seq_len)
