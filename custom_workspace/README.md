# Stroke Rehabilitation Pipeline

A research platform for processing motion capture data from stroke patients and healthy subjects, generating synthetic motion with Conditional VAEs, and running musculoskeletal simulations via MyoSuite/MuJoCo.

## End-to-End Pipeline

```
                         ┌─────────────────────────────────────────────────┐
                         │              DATA PROCESSING                    │
                         │                                                 │
  Raw MHH MoCap CSVs ──> │  Preprocessing & Cutoff Extraction             │
  (Healthy + Stroke)     │  src/data_processing/create_cutoff.py          │
                         │         │                                       │
                         │         v                                       │
                         │  Augmented CSVs (FMA 0-66 labels)              │
                         │  data/kinematic/cutoff/augmented/              │
                         └────────────┬────────────────────────────────────┘
                                      │
                         ┌────────────v────────────────────────────────────┐
                         │           CVAE TRAINING                         │
                         │                                                 │
                         │  Train: src/generation/cvae_train.py           │
                         │  Model: src/generation/model.py (MotionCVAE)   │
                         │         │                                       │
                         │         v                                       │
                         │  Trained weights + scaler                       │
                         │  models/cvae/cvae_cutoff_fma.pth               │
                         │  models/cvae/scaler_cutoff_fma.pkl             │
                         └────────────┬────────────────────────────────────┘
                                      │
                         ┌────────────v────────────────────────────────────┐
                         │         MOTION GENERATION                       │
                         │                                                 │
                         │  Generate: src/generation/cvae_generate.py     │
                         │  Input: FMA score (0-66)                        │
                         │         │                                       │
                         │         v                                       │
                         │  Synthetic CSVs (e.g. FMA_50.csv)              │
                         │  output/generated/csv/                          │
                         └────────────┬────────────────────────────────────┘
                                      │
                         ┌────────────v────────────────────────────────────┐
                         │        INVERSE KINEMATICS                       │
                         │                                                 │
                         │  CSV -> TRC (marker trajectories)               │
                         │  src/data_processing/convert_csv2trc.py        │
                         │         │                                       │
                         │         v                                       │
                         │  TRC -> MOT (joint angles via MuJoCo IK)       │
                         │  src/inverse_kinematics/convert_trc2mot.py     │
                         │  Model: models/model/myo_sim/arm/myoarm.xml   │
                         └────────────┬────────────────────────────────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    v                 v                   v
         ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │ INV DYNAMICS  │  │  VISUALIZATION   │  │   RL TRAINING    │
         │               │  │                  │  │                  │
         │ MOT -> Torques│  │ MOT -> MP4 Video │  │ MOT -> Imitation │
         │ calc_mot2     │  │ convert_mot2     │  │ train_drinking   │
         │  invdyn.py    │  │  video.py        │  │  _task.py        │
         └──────────────┘  └──────────────────┘  └──────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Activate the conda environment
conda activate MyoSuite    # Python 3.9 with dm_control, mujoco, torch
```

### Run the Full Pipeline

```bash
# 1. Process original MoCap recordings (batch mode)
python scripts/run_pipeline.py

# 2. Train the CVAE on augmented data
python src/generation/cvae_train.py -n 15000 -e 300

# 3. Generate synthetic motion for a given FMA score
python src/generation/cvae_generate.py 50

# 4. Convert generated CSVs through IK/ID
python scripts/run_generated_pipeline.py
```

### Process a Single File

```bash
# Modular single-file pipeline (CSV -> TRC -> MOT -> MP4)
python scripts/run_modular_pipeline.py
```

### Individual Conversion Steps

```bash
# CSV to TRC (marker trajectories)
python src/data_processing/convert_csv2trc.py input.csv output.trc

# TRC to MOT (inverse kinematics via MuJoCo)
python src/inverse_kinematics/convert_trc2mot.py \
    models/model/myo_sim/arm/myoarm.xml output.trc output.mot

# MOT to MP4 (render simulation video)
python src/visualization/convert_mot2video.py \
    models/model/myo_sim/arm/myoarm.xml output.mot output.mp4

# MOT to Inverse Dynamics (joint torques/forces)
python src/inverse_dynamics/calc_mot2invdyn.py
```

## Directory Structure

```
custom_workspace/
├── config/
│   └── settings.yaml                # Centralized configuration (paths, hyperparams)
├── data/
│   └── kinematic/
│       ├── healthy/                  # Healthy subject CSVs (e.g. 01_12_1.csv)
│       ├── stroke/                   # Stroke patient CSVs (e.g. S3_12_1.csv)
│       └── cutoff/
│           ├── processed/            # Preprocessed cutoff data
│           ├── original/             # Original cutoff segments
│           └── augmented/            # FMA-labeled augmented data (~56k files)
├── models/
│   ├── cvae/                         # Trained CVAE weights + scaler
│   │   ├── cvae_cutoff_fma.pth
│   │   ├── cvae_cutoff_fma_best.pth
│   │   └── scaler_cutoff_fma.pkl
│   └── model/myo_sim/               # MyoSuite musculoskeletal models
│       ├── arm/myoarm.xml            # Primary model used for IK (27 DoF, 63 muscles)
│       ├── elbow/                    # 2 DoF, 6 muscles
│       ├── hand/                     # 23 DoF, 39 muscles
│       ├── finger/                   # 4 DoF, 5 muscles
│       ├── leg/                      # 20 DoF, 80 muscles
│       ├── osl/                      # Prosthesis model (19 DoF, 54 muscles + 2 actuators)
│       ├── torso/                    # Back model (18 DoF, 210 muscles)
│       └── meshes/                   # STL bone/body meshes
├── src/
│   ├── data_processing/
│   │   ├── convert_csv2trc.py        # Raw CSV -> TRC marker file
│   │   ├── create_cutoff.py          # Extract cutoff segments from raw data
│   │   └── batch_processor.py        # Multi-file batch processing logic
│   ├── generation/
│   │   ├── model.py                  # MotionCVAE architecture (shared)
│   │   ├── cvae_train.py             # Train CVAE on augmented data
│   │   ├── cvae_generate.py          # Generate motion for a given FMA score
│   │   └── generate_augmented_fma.py # Create FMA-labeled augmented dataset
│   ├── inverse_kinematics/
│   │   ├── convert_trc2mot.py        # TRC -> MOT via MuJoCo IK solver
│   │   ├── interactive_alignment.py  # GUI for manual coordinate alignment
│   │   └── trc_data_scaler.py        # Scale markers to model dimensions
│   ├── inverse_dynamics/
│   │   └── calc_mot2invdyn.py        # MOT -> joint torques/forces
│   ├── visualization/
│   │   ├── convert_mot2video.py      # Render MOT to MP4 video
│   │   ├── visualise_trc.py          # TRC marker visualization
│   │   ├── visualise_id_results.py   # Inverse dynamics result plots
│   │   ├── analyze_trunk_compensation.py
│   │   ├── fma_trend_analysis.py     # FMA score trend plots
│   │   ├── advanced_analysis.py      # PCA, manifold, and cluster analysis
│   │   ├── interactive_viewer.py     # MuJoCo interactive viewer
│   │   ├── play_kinematics.py        # Playback joint angles
│   │   └── phase_selector.py         # Motion phase selection tool
│   ├── rl/
│   │   └── train_drinking_task.py    # PPO imitation learning agent
│   └── utils/
│       ├── config.py                 # Centralized config loader
│       ├── data_io.py                # Data read/write helpers
│       ├── ik_solver.py              # IK solver implementation
│       ├── markers.py                # Marker name mappings
│       ├── transforms.py             # Coordinate transforms
│       └── visualization_utils.py    # Shared plotting utilities
├── scripts/
│   ├── run_pipeline.py               # Batch IK pipeline entry point
│   ├── run_modular_pipeline.py       # Single-file modular pipeline
│   └── run_generated_pipeline.py     # CVAE output -> IK/ID bridge
├── output/
│   ├── originals/                    # Full-frame MoCap outputs (trc/, mot/, videos/, id/)
│   ├── compressed/                   # 100-frame resampled outputs
│   ├── generated/                    # CVAE synthetic outputs (csv/, mot/, trc/, videos/, id/)
│   └── analysis/                     # Analysis plots and dashboards
├── docs/info/                        # Patient details, data release notes
├── tests/                            # Test suite (placeholder)
├── requirements.txt
└── CLAUDE.md                         # AI assistant instructions
```

## Configuration

All paths and hyperparameters are centralized in `config/settings.yaml` and accessed via:

```python
from src.utils.config import get_path, get, get_project_root

model_path = get_path("mujoco_arm_model")    # -> absolute path to myoarm.xml
data_rate  = get("pipeline", "data_rate")    # -> 200.0
latent_dim = get("cvae", "latent_dim")       # -> 32
```

### Key Configuration Sections

| Section | Key Settings |
|---------|-------------|
| `paths` | Model files, data dirs, output dirs, reference files |
| `pipeline` | Data rate (200 Hz), alignment mode, shoulder locking |
| `cvae` | Input dim (15), latent dim (32), hidden dim (256), seq len (100) |
| `video` | Camera angles, resolution (640x480) |
| `inverse_dynamics` | Solve joints, torque scaling, low-pass cutoff |

## Pipeline Details

### 1. Data Processing

Raw motion capture CSVs from MHH (Medizinische Hochschule Hannover) contain multi-level headers with marker names and X/Y/Z coordinates at 200 Hz in millimeters.

**Cutoff extraction** (`src/data_processing/create_cutoff.py`): Segments the reaching phase from raw recordings and normalizes positions relative to the chest center.

**Augmentation** (`src/generation/generate_augmented_fma.py`): Labels each trial with its FMA-UE (Fugl-Meyer Assessment - Upper Extremity) score and creates the training dataset.

### 2. CVAE Training

The Conditional Variational Autoencoder (`src/generation/model.py`) learns to generate motion trajectories conditioned on FMA scores:

- **Architecture**: Encoder-Decoder with self-attention (4 heads)
- **Input**: 15 dimensions (12 arm joint positions + 3 trunk compensation)
- **Conditioning**: FMA score normalized to [0, 1] range
- **Latent space**: 32 dimensions
- **Sequence length**: 100 frames per trajectory

```bash
python src/generation/cvae_train.py -n 15000 -e 300
# Saves: models/cvae/cvae_cutoff_fma.pth + scaler_cutoff_fma.pkl
```

### 3. Motion Generation

```bash
python src/generation/cvae_generate.py 50    # Generate for FMA=50
# Output: output/generated/csv/FMA_50.csv
```

The generated CSV contains reconstructed marker positions. The **V_Vector fix** is applied: the CVAE outputs wrist direction vectors (WRB-WRA), which must be reconstructed as `V_Vector = V_Wrist + WrVec` for the IK solver.

### 4. Inverse Kinematics (IK)

The IK solver uses MuJoCo to find joint angles that best match marker positions:

1. **CSV -> TRC**: Converts marker data into OpenSim-compatible TRC format
2. **TRC -> MOT**: MuJoCo-based IK solver fits the MyoArm model (27 DoF, 63 muscles) to marker trajectories

**IK Error Guidelines**: Original recordings ~18mm, generated motions ~24-30mm, errors >200mm indicate bad samples.

Reference file `S5_12_1.mot` must be processed first in batch mode (provides the initial pose reference).

### 5. Inverse Dynamics (ID)

Computes the joint torques and forces required to produce the motion described in a MOT file:

```bash
python src/inverse_dynamics/calc_mot2invdyn.py
```

### 6. Visualization

```bash
# Render motion to video
python src/visualization/convert_mot2video.py model.xml input.mot output.mp4

# Trunk compensation analysis
python src/visualization/analyze_trunk_compensation.py

# FMA trend analysis
python src/visualization/fma_trend_analysis.py
```

### 7. Reinforcement Learning

PPO agent (Stable Baselines 3) trained with imitation learning to perform a drinking task by tracking reference MOT files:

```bash
python src/rl/train_drinking_task.py
```

## Data Formats

| Format | Description | Example |
|--------|-------------|---------|
| CSV | Raw MHH MoCap, multi-level headers, 200 Hz, millimeters | `01_12_1.csv` |
| TRC | OpenSim-compatible marker trajectories with header block | `01_12_1.trc` |
| MOT | Tab-separated joint angles with time column | `01_12_1.mot` |
| MP4 | Rendered simulation video | `01_12_1.mp4` |

## Naming Conventions

| Pattern | Description | Example |
|---------|-------------|---------|
| `{num}_{session}_{trial}` | Healthy subject | `01_12_1.csv` |
| `S{num}_{session}_{trial}` | Stroke patient | `S3_12_1.csv` |
| `FMA_{score}` | Generated motion | `FMA_50.csv` |

## Environment

```bash
conda activate MyoSuite
# Python 3.9 | dm_control | mujoco | torch | stable-baselines3
```

## Musculoskeletal Models

This project uses [MyoSuite's](https://github.com/facebookresearch/myoSuite) musculoskeletal model library. The primary model for upper-limb IK is **MyoArm** (27 DoF, 63 muscles), converted from the MoBL OpenSim model. See `models/model/myo_sim/` for all available models and their documentation.
