"""
experiments.py — Clean experimental pipeline (no sagittal constraint, kinematics only).

Each experiment is a dict with:
  desc   : human-readable description
  model  : MotionCVAE config (use_film, use_cfg, use_residual, use_temporal_conv, cond_drop_prob)
  aug    : augmentation dataset ("smote" | "dtw" | "linear")
  losses : loss weight overrides (w_vel, w_acc, w_dyn, w_kl)

Pipeline:
  Phase A — Architecture search (Stage 0–4) × 3 augmented datasets (15 experiments)
             Winner: Stage 2 (FiLM+CFG, no residual) + SMOTE
  Phase B — Loss term ablation on Stage 2 + SMOTE (5 experiments, SMOTE only)
             B0: full loss reference  B1: -dyn  B2: -acc  B3: -vel  B4: -kl
  Phase C — Loss weight sweep on Stage 2 + SMOTE (12 experiments)
             One term varied at a time; other weights fixed at defaults.

All experiments evaluated at N=10 (average of 10 samples) for stable metrics.

Primary metric : rho_wrist  (Spearman ρ, FMA vs wrist vertical range, target ≥ 0.9)
Secondary      : rho_trunk  (Spearman ρ, FMA vs trunk/wrist ratio, target < 0)

No physical constraints (no w_sag, no w_seg).
"""

# ── Shared loss defaults ──────────────────────────────────────────────────────
# All experiments use these unless overridden in the losses dict.
# w_vel=10, w_acc=5, w_kl=0.1, w_dyn=2 — same across all phases for fair comparison.
_FULL_LOSS = {"w_vel": 10.0, "w_acc": 5.0, "w_kl": 0.1, "w_dyn": 2.0}

EXPERIMENTS = {

    # ════════════════════════════════════════════════════════════════════════
    # PHASE A: Architecture search
    #
    # Additive ablation: start from the absolute baseline (Stage 0) and add
    # one architectural component at a time. Each stage is trained on all
    # three augmented datasets so that the best architecture AND augmentation
    # are identified simultaneously.
    #
    # Stage 0 — BiLSTM encoder + LSTM decoder + concat conditioning
    #           No CFG, no FiLM, no residual. Pure baseline.
    # Stage 1 — + Classifier-free guidance (CFG)
    #           Adds unconditional branch; guidance scale amplifies FMA signal
    #           at inference without retraining.
    # Stage 2 — + Feature-wise Linear Modulation (FiLM)
    #           Replaces concat conditioning with multiplicative modulation
    #           of every hidden state by the FMA condition.
    # Stage 3 — + Residual skip connection in decoder
    #           Decoder learns a correction on top of a stable residual
    #           baseline rather than reconstructing the full trajectory
    #           from scratch. Also eliminates LSTM edge artefacts.
    # Stage 4 — Stage 3 + TemporalConvBlock (TCB)
    #           Bottleneck 1D-CNN (hidden→64→64→hidden, kernel=5) placed
    #           after the LSTM. Tested in sag-era Phase I but contaminated
    #           by the sagittal constraint. Clean test here.
    # ════════════════════════════════════════════════════════════════════════

    # Stage 0 — baseline
    "A0_smote": {
        "desc": "Stage 0 baseline — no CFG, no FiLM, no residual | SMOTE",
        "model": {"use_film": False, "use_cfg": False, "use_residual": False},
        "aug":   "smote",
        "losses": _FULL_LOSS,
    },
    "A0_dtw": {
        "desc": "Stage 0 baseline — no CFG, no FiLM, no residual | DTW",
        "model": {"use_film": False, "use_cfg": False, "use_residual": False},
        "aug":   "dtw",
        "losses": _FULL_LOSS,
    },
    "A0_linear": {
        "desc": "Stage 0 baseline — no CFG, no FiLM, no residual | Linear",
        "model": {"use_film": False, "use_cfg": False, "use_residual": False},
        "aug":   "linear",
        "losses": _FULL_LOSS,
    },

    # Stage 1 — + CFG
    "A1_smote": {
        "desc": "Stage 1 — + CFG | SMOTE",
        "model": {"use_film": False, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": _FULL_LOSS,
    },
    "A1_dtw": {
        "desc": "Stage 1 — + CFG | DTW",
        "model": {"use_film": False, "use_cfg": True, "use_residual": False},
        "aug":   "dtw",
        "losses": _FULL_LOSS,
    },
    "A1_linear": {
        "desc": "Stage 1 — + CFG | Linear",
        "model": {"use_film": False, "use_cfg": True, "use_residual": False},
        "aug":   "linear",
        "losses": _FULL_LOSS,
    },

    # Stage 2 — + FiLM
    "A2_smote": {
        "desc": "Stage 2 — + FiLM | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": _FULL_LOSS,
    },
    "A2_dtw": {
        "desc": "Stage 2 — + FiLM | DTW",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "dtw",
        "losses": _FULL_LOSS,
    },
    "A2_linear": {
        "desc": "Stage 2 — + FiLM | Linear",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "linear",
        "losses": _FULL_LOSS,
    },

    # Stage 3 — + Residual decoder
    "A3_smote": {
        "desc": "Stage 3 — + Residual decoder | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": True},
        "aug":   "smote",
        "losses": _FULL_LOSS,
    },
    "A3_dtw": {
        "desc": "Stage 3 — + Residual decoder | DTW",
        "model": {"use_film": True, "use_cfg": True, "use_residual": True},
        "aug":   "dtw",
        "losses": _FULL_LOSS,
    },
    "A3_linear": {
        "desc": "Stage 3 — + Residual decoder | Linear",
        "model": {"use_film": True, "use_cfg": True, "use_residual": True},
        "aug":   "linear",
        "losses": _FULL_LOSS,
    },

    # Stage 4 — + TemporalConvBlock (TCB)
    "A4_smote": {
        "desc": "Stage 4 — Stage 3 + TemporalConvBlock | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": True,
                  "use_temporal_conv": True},
        "aug":   "smote",
        "losses": _FULL_LOSS,
    },
    "A4_dtw": {
        "desc": "Stage 4 — Stage 3 + TemporalConvBlock | DTW",
        "model": {"use_film": True, "use_cfg": True, "use_residual": True,
                  "use_temporal_conv": True},
        "aug":   "dtw",
        "losses": _FULL_LOSS,
    },
    "A4_linear": {
        "desc": "Stage 4 — Stage 3 + TemporalConvBlock | Linear",
        "model": {"use_film": True, "use_cfg": True, "use_residual": True,
                  "use_temporal_conv": True},
        "aug":   "linear",
        "losses": _FULL_LOSS,
    },

    # ════════════════════════════════════════════════════════════════════════
    # PHASE B: Loss function ablation
    #
    # Base architecture: Stage 3 (expected Phase A winner).
    # Each experiment removes exactly one loss term to measure its contribution.
    # Run on all three augmented datasets — some loss terms may matter more
    # depending on how well-structured the training data is.
    #
    # B0 — Full reference loss (pos + vel + acc + KL + dyn)
    #      Should match A3_* results — confirms Phase B is correctly wired.
    # B1 — Remove dynamics correlation (w_dyn=0)
    #      Tests whether the FMA→velocity/jerk clinical prior adds value
    #      beyond what the architecture learns from data alone.
    # B2 — Remove acceleration loss (w_acc=0)
    #      Tests temporal jerk suppression — acc loss penalises frame-to-frame
    #      velocity changes, discouraging unrealistic motion spikes.
    # B3 — Remove velocity loss (w_vel=0)
    #      Tests temporal consistency — vel loss drives smooth frame-to-frame
    #      transitions; without it the model may produce positionally accurate
    #      but kinematically discontinuous trajectories.
    # B4 — Remove KL divergence (w_kl=0)
    #      Tests whether latent regularisation matters for FMA gradient quality.
    #      Without KL the latent space is unconstrained — generation from
    #      random z may degrade significantly.
    # ════════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════════════
    # Phase B — Loss term ablation
    # Fixed architecture: Stage 2 (FiLM+CFG, no residual), SMOTE only.
    # One term removed at a time; B0 is the full-loss reference.
    # ════════════════════════════════════════════════════════════════════════

    "B0_smote": {
        "desc": "Full loss reference | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": _FULL_LOSS,
    },
    "B1_smote": {
        "desc": "No dynamics correlation (w_dyn=0) | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 5.0, "w_kl": 0.1, "w_dyn": 0.0},
    },
    "B2_smote": {
        "desc": "No acceleration loss (w_acc=0) | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 0.0, "w_kl": 0.1, "w_dyn": 2.0},
    },
    "B3_smote": {
        "desc": "No velocity loss (w_vel=0) | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 0.0, "w_acc": 5.0, "w_kl": 0.1, "w_dyn": 2.0},
    },
    "B4_smote": {
        "desc": "No KL divergence (w_kl=0) | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 5.0, "w_kl": 0.0, "w_dyn": 2.0},
    },

    # ════════════════════════════════════════════════════════════════════════
    # Phase C — Loss weight sweep
    # Fixed architecture: Stage 2 (FiLM+CFG, no residual), SMOTE only.
    # One weight varied at a time; all others fixed at defaults.
    # Reference is B0_smote (all defaults).
    # ════════════════════════════════════════════════════════════════════════

    # w_vel sweep {2.5, 5.0, 20.0}  (default 10.0 = B0_smote)
    "C_vel_2p5": {
        "desc": "w_vel=2.5 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 2.5, "w_acc": 5.0, "w_kl": 0.1, "w_dyn": 2.0},
    },
    "C_vel_5": {
        "desc": "w_vel=5.0 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 5.0, "w_acc": 5.0, "w_kl": 0.1, "w_dyn": 2.0},
    },
    "C_vel_20": {
        "desc": "w_vel=20.0 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 20.0, "w_acc": 5.0, "w_kl": 0.1, "w_dyn": 2.0},
    },

    # w_acc sweep {1.25, 2.5, 10.0}  (default 5.0 = B0_smote)
    "C_acc_1p25": {
        "desc": "w_acc=1.25 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 1.25, "w_kl": 0.1, "w_dyn": 2.0},
    },
    "C_acc_2p5": {
        "desc": "w_acc=2.5 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 2.5, "w_kl": 0.1, "w_dyn": 2.0},
    },
    "C_acc_10": {
        "desc": "w_acc=10.0 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 10.0, "w_kl": 0.1, "w_dyn": 2.0},
    },

    # w_kl sweep {0.025, 0.05, 0.2}  (default 0.1 = B0_smote)
    "C_kl_0p025": {
        "desc": "w_kl=0.025 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 5.0, "w_kl": 0.025, "w_dyn": 2.0},
    },
    "C_kl_0p05": {
        "desc": "w_kl=0.05 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 5.0, "w_kl": 0.05, "w_dyn": 2.0},
    },
    "C_kl_0p2": {
        "desc": "w_kl=0.2 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 5.0, "w_kl": 0.2, "w_dyn": 2.0},
    },

    # w_dyn sweep {0.5, 1.0, 4.0}  (default 2.0 = B0_smote)
    "C_dyn_0p5": {
        "desc": "w_dyn=0.5 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 5.0, "w_kl": 0.1, "w_dyn": 0.5},
    },
    "C_dyn_1": {
        "desc": "w_dyn=1.0 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 5.0, "w_kl": 0.1, "w_dyn": 1.0},
    },
    "C_dyn_4": {
        "desc": "w_dyn=4.0 | Stage 2 | SMOTE",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 5.0, "w_kl": 0.1, "w_dyn": 4.0},
    },

    # ════════════════════════════════════════════════════════════════════════
    # Phase D: Hyperparameter sweep (paper: Phase 3)
    #
    # Base config: Stage 2 (FiLM+CFG, no residual), SMOTE, 300 epochs.
    # Loss weights: Phase C optimal — w_vel=10, w_acc=2.5, w_kl=0.2, w_dyn=2.
    # All sweeps hold other params at base values.
    #
    # Sweeps:
    #   Latent dim   : {16, [32], 64}
    #   KL weight (β): {[0.2], 0.5, 1.0, 2.0}
    #   Learning rate: {1e-4, 5e-4, [1e-3]}
    #
    # [] = base (covered by D_base, shared across all three sweep tables).
    # ════════════════════════════════════════════════════════════════════════

    # Shared base for all Phase D sweeps
    "D_base": {
        "desc": "Phase D base — latent=32, w_kl=0.2, lr=1e-3 | Stage 2 | SMOTE | 300ep",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False, "latent_dim": 32},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 2.5, "w_kl": 0.2, "w_dyn": 2.0},
        "learning_rate": 1e-3,
    },

    # Latent dimension sweep
    "D_latent_16": {
        "desc": "Latent dim=16 | Stage 2 | SMOTE | 300ep",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False, "latent_dim": 16},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 2.5, "w_kl": 0.2, "w_dyn": 2.0},
        "learning_rate": 1e-3,
    },
    "D_latent_64": {
        "desc": "Latent dim=64 | Stage 2 | SMOTE | 300ep",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False, "latent_dim": 64},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 2.5, "w_kl": 0.2, "w_dyn": 2.0},
        "learning_rate": 1e-3,
    },

    # KL regularisation (β) sweep — extends Phase C range upward
    "D_kl_0p5": {
        "desc": "β=0.5 | Stage 2 | SMOTE | 300ep",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False, "latent_dim": 32},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 2.5, "w_kl": 0.5, "w_dyn": 2.0},
        "learning_rate": 1e-3,
    },
    "D_kl_1p0": {
        "desc": "β=1.0 | Stage 2 | SMOTE | 300ep",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False, "latent_dim": 32},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 2.5, "w_kl": 1.0, "w_dyn": 2.0},
        "learning_rate": 1e-3,
    },
    "D_kl_2p0": {
        "desc": "β=2.0 | Stage 2 | SMOTE | 300ep",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False, "latent_dim": 32},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 2.5, "w_kl": 2.0, "w_dyn": 2.0},
        "learning_rate": 1e-3,
    },

    # Learning rate sweep
    "D_lr_1e4": {
        "desc": "lr=1e-4 | Stage 2 | SMOTE | 300ep",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False, "latent_dim": 32},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 2.5, "w_kl": 0.2, "w_dyn": 2.0},
        "learning_rate": 1e-4,
    },
    "D_lr_5e4": {
        "desc": "lr=5e-4 | Stage 2 | SMOTE | 300ep",
        "model": {"use_film": True, "use_cfg": True, "use_residual": False, "latent_dim": 32},
        "aug":   "smote",
        "losses": {"w_vel": 10.0, "w_acc": 2.5, "w_kl": 0.2, "w_dyn": 2.0},
        "learning_rate": 5e-4,
    },
}

# ── Run order ─────────────────────────────────────────────────────────────────
#
# Phase A (done):  15 experiments, Stage 0–4 × SMOTE/DTW/Linear.
#                  Winner: Stage 2 (FiLM+CFG) + SMOTE  (rho_wrist=0.908 @ N=10)
#
# Phase B (done):  5 experiments, loss term ablation on Stage 2 + SMOTE.
#
# Phase C (done):  12 experiments, one-dimensional weight sweep per term.
#                  Optimal: w_vel=10, w_acc=2.5, w_kl=0.2, w_dyn=2.0
#
# Phase D (next):  8 experiments, hyperparameter sweep (latent dim, β, lr).
#                  Base: Stage 2 + SMOTE + Phase C optimal weights + 300 epochs.
#
# ── Experiment count ──────────────────────────────────────────────────────────
# Phase A: 5 stages × 3 datasets = 15 experiments
# Phase B: 5 ablation configs     = 5 experiments
# Phase C: 3 non-default × 4 terms = 12 experiments
# Phase D: 1 base + 2 latent + 3 kl + 2 lr = 8 experiments
