# Cutoff Pipeline: FMA-Targeted Motion Generation

This directory contains a specialized pipeline for generating synthetic upper-limb motions conditioned on **Fugl-Meyer Assessment (FMA)** scores. The model learns the relationship between motor impairment severity and movement kinematics, enabling generation of biomechanically plausible motions at any FMA score from 18-66.

---

## Complete Pipeline: Raw Data → CVAE Training

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Raw MoCap Data                                                     │
│  data/kinematic/Healthy/*.csv, data/kinematic/Stroke/*.csv                  │
│  - Multi-level headers (Marker, X/Y/Z)                                      │
│  - 200 Hz sampling rate                                                     │
│  - Markers: WRA, WRB, ELB_L, ELB_M, SA_1-3, CS_1-4, etc.                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Manual Phase Selection                                             │
│  Script: IK/visual/new_interactive_phase_selector.py                        │
│  Input: Raw data (Healthy/*.csv, Stroke/*.csv)                              │
│  Output: manual_phase_indices.json (temp) → rename to:                      │
│          - stroke_phase_indices.json (for stroke data)                      │
│          - healthy/healthy_phase_indices.json (for healthy data)            │
│  - Interactive 3D trajectory + velocity plot                                │
│  - Select 3 keypoints per file (Pick, Drink, Place)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Preprocess & Cut Phases                                            │
│  Script: IK/preprocess_markers.py (or manual preprocessing)                 │
│  Input: Raw data + phase indices JSON                                       │
│  Output: data/kinematic/cutoff/original/*.csv (variable frame counts)       │
│  Process:                                                                   │
│    1. Calculate virtual joints (Shoulder, Elbow, Wrist centers)             │
│    2. Normalize by chest (subtract CS marker centroid)                      │
│    3. Compute wrist vector (WRB - WRA)                                      │
│    4. Cut to functional phase using JSON indices                            │
│  Output format: 12 columns (Sh, El, Wr XYZ + WrVec XYZ)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Resample to 100 Frames                                             │
│  (Part of preprocessing workflow)                                           │
│  Output: data/kinematic/cutoff/processed/*.csv (100 frames each)            │
│  - Temporal normalization via scipy.signal.resample                         │
│  - Enables consistent sequence length for CVAE                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Generate FMA-Labeled Augmented Data                                │
│  Script: IK/cutoff/scripts/generate_augmented_fma.py                        │
│  Input: cutoff/processed/, IK/output/scores.csv (FMA scores)                │
│  Output: data/kinematic/cutoff/augmented/ (56,574 files)                    │
│  - For each stroke patient (FMA X): morph to FMA X+1, X+2, ..., 65          │
│  - Linear interpolation: alpha = (target - stroke_FMA) / (66 - stroke_FMA)  │
│  - FMA score embedded in filename: S11_12_1_x_01_12_1_FMA45.csv             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Train CVAE                                                         │
│  Script: IK/cutoff/scripts/cvae_train_cutoff.py                             │
│  Input: cutoff/augmented/                                                   │
│  Output: IK/cutoff/models/cvae_cutoff_fma_best.pth, scaler_cutoff_fma.pkl   │
│  - BiLSTM Encoder + Self-Attention + BiLSTM Decoder                         │
│  - Classifier-free guidance (10% unconditional dropout)                     │
│  - Multi-term loss: position + velocity + acceleration + KL                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: Generate New Motions                                               │
│  Script: IK/cutoff/scripts/cvae_generate_motion_cutoff.py                   │
│  Output: IK/cutoff/output/generated/FMA_XX.csv                              │
│  - Generate motion for any FMA score 0-66                                   │
│  - Guidance scale controls conditioning strength                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Quick Start Commands

```bash
cd /home/abdul/Desktop/myosuite/custom_workspace

# Step 2: Manual phase selection (interactive)
python IK/visual/new_interactive_phase_selector.py --dataset healthy
python IK/visual/new_interactive_phase_selector.py --dataset stroke

# Step 3-4: Preprocess & cut phases (already done - cutoff/original/ and cutoff/processed/ exist)
# This was done using IK/preprocess_markers.py logic + phase indices

# Step 5: Generate FMA-labeled augmented data
python IK/cutoff/scripts/generate_augmented_fma.py

# Step 7: Train CVAE
python IK/cutoff/scripts/cvae_train_cutoff.py -n 15000 -e 300

# Step 8: Generate motion for FMA score 35
python IK/cutoff/scripts/cvae_generate_motion_cutoff.py --fma 35
```

### Data Directory Summary

| Directory | Description | Files | Status |
|-----------|-------------|-------|--------|
| `data/kinematic/Healthy/` | Raw healthy MoCap (multi-level headers) | 54 | Source |
| `data/kinematic/Stroke/` | Raw stroke MoCap (multi-level headers) | 21 | Source |
| `data/kinematic/cutoff/original/` | Phase-cut, chest-normalized (12 cols, variable frames) | 75 | Step 3 output |
| `data/kinematic/cutoff/processed/` | Resampled to 100 frames | 75 | Step 4 output |
| `data/kinematic/cutoff/augmented/` | FMA-labeled training data (15 cols with trunk) | 56,574 | Step 5 output |

### Scripts Reference

| Script | Location | Purpose |
|--------|----------|---------|
| `new_interactive_phase_selector.py` | `IK/visual/` | Manual phase boundary selection |
| `preprocess_markers.py` | `IK/` | Virtual joints + chest normalization |
| `generate_augmented_fma.py` | `IK/cutoff/scripts/` | Create FMA-labeled training data |
| `cvae_train_cutoff.py` | `IK/cutoff/scripts/` | Train CVAE model |
| `cvae_generate_motion_cutoff.py` | `IK/cutoff/scripts/` | Generate new motions |
| `verify_training.py` | `IK/cutoff/scripts/` | Training verification dashboard |
| `interactive_viewer.py` | `IK/cutoff/scripts/` | Real-time FMA slider visualization |
| `fma_trend_analysis.py` | `IK/cutoff/scripts/` | FMA trend graphs |
| `analyze_trunk_compensation.py` | `IK/cutoff/scripts/` | Trunk compensation analysis |

---

## Key Results

The `training_verification_fma.png` dashboard demonstrates successful FMA conditioning:

| Metric | FMA 66 (Healthy) | FMA 18 (Impaired) | Expected Trend |
|--------|------------------|-------------------|----------------|
| Wrist Range Y | ~250mm | ~224mm | Higher FMA = More Range |
| Peak Velocity | Higher | Lower | Higher FMA = Faster |
| Trunk Compensation | Lower | Higher | Lower FMA = More Trunk |
| Motion Smoothness | Smoother | Jerkier | Higher FMA = Smoother |

The model correctly learned all four clinical trends, confirming FMA conditioning works.

## Directory Structure

```
IK/cutoff/
├── models/
│   ├── cvae_cutoff_fma_best.pth    # Best trained model (use this)
│   ├── cvae_cutoff_fma.pth         # Final epoch model
│   └── scaler_cutoff_fma.pkl       # StandardScaler (MUST use with model)
├── scripts/
│   ├── preprocess_unified.py       # Step 1: Data preprocessing
│   ├── generate_augmented_fma.py   # Step 2: Generate FMA-labeled training data
│   ├── cvae_train_cutoff.py        # Step 3: Train the CVAE
│   ├── verify_training.py          # Step 4: Generate verification dashboard
│   ├── cvae_generate_motion_cutoff.py  # Inference: Generate new motions
│   ├── fma_trend_analysis.py       # Analysis: FMA trend visualization
│   └── analyze_trunk_compensation.py   # Analysis: Trunk compensation metrics
└── output/
    ├── fma_trend_analysis.png
    └── trunk_compensation_analysis.png
```

## What Made This Work

### 1. FMA Score Embedded in Filenames

Instead of inferring FMA from ratio-based interpolation, FMA scores are directly embedded in filenames:
```
S11_12_1_x_01_12_1_FMA45.csv  # Stroke+Healthy morphed to FMA 45
S14_12_2_FMA22.csv            # Original stroke patient (FMA 22)
01_12_1_FMA66.csv             # Healthy subject (FMA 66)
```

This gives the model explicit supervision for each FMA score.

### 2. Massive Data Augmentation (56,574 files)

For each stroke patient with FMA score X:
- Pair with every healthy subject
- Generate morphed motions for FMA X+1, X+2, ..., 65
- Linear interpolation: `alpha = (target_FMA - stroke_FMA) / (66 - stroke_FMA)`

This creates continuous coverage across the full FMA range.

### 3. Trunk Compensation Modeling

Added 3 trunk marker columns (CS chest markers centroid) to capture compensatory movements:
- **15 input dimensions**: 12 arm + 3 trunk
- Stroke patients exhibit more trunk displacement to compensate for arm weakness
- Model learns this relationship and generates appropriate trunk motion per FMA

### 4. Improved Architecture

| Component | Description |
|-----------|-------------|
| Encoder | 2-layer BiLSTM + Multi-head Self-Attention |
| Decoder | BiLSTM + 2 Residual Blocks |
| Hidden Dim | 256 |
| Latent Dim | 32 |
| Attention Heads | 4 |

### 5. Multi-Term Loss Function

```python
loss = recon_loss + 10*velocity_loss + 5*acceleration_loss + 0.1*KL_divergence
```

- **Position loss**: Reconstruct joint positions
- **Velocity loss**: Match velocity profiles (bell-shaped for healthy)
- **Acceleration loss**: Ensures smooth, non-jerky motion
- **KL loss**: Regularize latent space

### 6. Classifier-Free Guidance

During training, 10% of samples have their FMA condition zeroed out. At inference, this enables stronger conditioning via guidance scale:

```python
output = uncond + scale * (cond - uncond)  # scale=2.0 default
```

### 7. Delta Format Preprocessing

All data converted to **displacement from first frame**:
- First frame = [0, 0, 0, ...]
- Subsequent frames = displacement from start
- Removes dependence on absolute starting position
- Model learns pure motion dynamics

## Pipeline Reproduction

### Step 1: Preprocess Data
```bash
python scripts/preprocess_unified.py
```
Converts raw data to delta format with trunk markers.

### Step 2: Generate FMA-Labeled Training Data
```bash
python scripts/generate_augmented_fma.py
```
Creates 56k+ files with FMA scores in filenames.

### Step 3: Train CVAE
```bash
python scripts/cvae_train_cutoff.py -n 15000 -e 300
```
Arguments:
- `-n`: Number of training samples (15000 recommended, or "all")
- `-e`: Number of epochs (300 default)

### Step 4: Verify Training
```bash
python scripts/verify_training.py
```
Generates the verification dashboard (`training_verification_fma.png`).

### Step 5: Generate New Motions
```bash
python scripts/cvae_generate_motion_cutoff.py --fma 35
```
Generate motion for any FMA score 0-66.

## Data Locations

| Dataset | Path | Files |
|---------|------|-------|
| Raw cutoff (arm) | `data/kinematic/cutoff/original/` | 78 |
| Processed (100 frames) | `data/kinematic/cutoff/processed/` | 78 |
| FMA-Augmented | `data/kinematic/cutoff/augmented/` | 56,574 |

## Model Architecture Details

```
Input: (batch, 100, 15)  # 100 frames, 15 features
       │
       ▼
┌──────────────────────┐
│  Encoder             │
│  ├─ BiLSTM (2-layer) │
│  ├─ Self-Attention   │
│  └─ Mean Pooling     │
└──────────────────────┘
       │
       ▼
   μ, log(σ²)  →  z ~ N(μ, σ²)  # Latent: 32-dim
       │
       ▼
┌──────────────────────┐
│  Decoder             │
│  ├─ FC: z+c → hidden │
│  ├─ BiLSTM (2-layer) │
│  ├─ Residual Block   │
│  ├─ Residual Block   │
│  └─ FC: hidden → 15  │
└──────────────────────┘
       │
       ▼
Output: (batch, 100, 15)
```

## Clinical Interpretation

The verification dashboard confirms the model captures key stroke rehabilitation biomechanics:

1. **Reduced Range of Motion**: Lower FMA → smaller wrist reach distance
2. **Slower Velocity**: Lower FMA → reduced peak speed (bell-curve flattening)
3. **Trunk Compensation**: Lower FMA → more trunk forward lean to compensate for arm weakness
4. **Increased Jerk**: Lower FMA → less smooth motion (motor control deficits)

These are exactly the impairments observed in clinical stroke populations.

## Files Not to Modify

- `models/scaler_cutoff_fma.pkl` - Must match the model; retraining requires new scaler
- `models/cvae_cutoff_fma_best.pth` - Best validation loss checkpoint

## Running the Visualization Scripts

All scripts can be run from the workspace root:

```bash
cd /home/abdul/Desktop/myosuite/custom_workspace

# 1. FMA Trend Analysis - generates graphs showing metric trends across FMA scores
python3 IK/cutoff/scripts/fma_trend_analysis.py
# Output: IK/cutoff/output/fma_trend_graphs.png, fma_trend_summary.png

# 2. Trunk Compensation Analysis - compares real healthy vs stroke trunk movement
python3 IK/cutoff/scripts/analyze_trunk_compensation.py
# Output: IK/cutoff/output/trunk_compensation_analysis.png

# 3. Training Verification Dashboard - compares generated vs real motion
python3 IK/cutoff/scripts/verify_training.py
# Output: IK/cutoff/scripts/training_verification_fma.png

# 4. Interactive Viewer - real-time 3D visualization with FMA slider
python3 IK/cutoff/scripts/interactive_viewer.py
# (Opens interactive matplotlib window)
```

## Generated Outputs

| File | Description |
|------|-------------|
| `output/fma_trend_graphs.png` | 16 plots showing metric trends vs FMA score |
| `output/fma_trend_summary.png` | Clinical summary with statistics |
| `output/fma_trend_data.csv` | Raw metrics data for all generated motions |
| `output/trunk_compensation_analysis.png` | Real data: stroke vs healthy trunk comparison |
| `output/trunk_compensation_data.csv` | Raw trunk metrics from real data |
| `scripts/training_verification_fma.png` | Dashboard comparing generated vs real |

## Next Steps

1. **RL Integration**: Use generated motions as reference trajectories for imitation learning
2. **Personalized Rehabilitation**: Generate patient-specific target motions at incremental FMA scores
3. **Outcome Prediction**: Train classifier to predict FMA from kinematic features
