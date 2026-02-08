# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **stroke rehabilitation research platform** that processes motion capture data from stroke patients and healthy subjects, converts it into biomechanically valid motion files, and uses generative AI (Conditional VAEs) to synthesize augmented motion data. The system integrates with MyoSuite/MuJoCo for musculoskeletal simulation and supports RL-based imitation learning.

## Key Commands

### Run the Full IK Pipeline (Batch Mode)
```bash
cd /home/abdul/Desktop/myosuite/custom_workspace
python scripts/run_pipeline.py  # Set RUN_BATCH_MODE = True in the script
```

### Process a Single File (Modular Pipeline)
```bash
python scripts/run_modular_pipeline.py
```

### Individual Conversion Steps
```bash
# CSV to TRC (marker trajectories)
python src/data_processing/convert_csv2trc.py /path/to/input.csv /path/to/output.trc

# TRC to MOT (inverse kinematics)
python src/inverse_kinematics/convert_trc2mot.py models/myo_sim/arm/myoarm.xml output/01_12_1.trc output/01_12_1.mot

# MOT to Video
python src/visualization/convert_mot2video.py models/myo_sim/arm/myoarm.xml output/file.mot output/file.mp4

# MOT to Inverse Dynamics
python src/inverse_dynamics/calc_mot2invdyn.py
```

### Run Generated Motion Pipeline (CVAE output -> IK/ID)
```bash
python scripts/run_generated_pipeline.py                # Process all generated FMA files
python scripts/run_generated_pipeline.py FMA_50.csv     # Process a single file
python scripts/run_generated_pipeline.py --skip-id      # Skip inverse dynamics
```

### Train CVAE Model
```bash
python src/generation/cvae_train.py -n 15000 -e 300
# Outputs: models/cvae/cvae_cutoff_fma.pth and models/cvae/scaler_cutoff_fma.pkl
```

### Generate Synthetic Motion
```bash
python src/generation/cvae_generate.py 50  # Generate motion for FMA score 50
```

### Train RL Agent (Imitation Learning)
```bash
python src/rl/train_drinking_task.py
```

## Architecture

### Data Pipeline
```
CSV (Raw MHH Mocap) -> TRC (Markers) -> MOT (Joint Angles) -> MP4 (Video)
                                     \-> Inverse Dynamics (Forces/Torques)
```

### ML Pipeline
```
Augmented Data (FMA_0 to FMA_66 CSVs) -> CVAE Training -> Motion Generation
```

### Directory Structure
```
custom_workspace/
├── config/settings.yaml          # Centralized configuration (all paths + hyperparams)
├── data/kinematic/               # Input motion capture data
│   ├── healthy/                  # Healthy subject recordings
│   ├── stroke/                   # Stroke patient recordings
│   └── cutoff/                   # Processed cutoff data + augmented (56k files)
├── models/
│   ├── myo_sim/arm/myoarm.xml   # MuJoCo musculoskeletal model
│   └── cvae/                     # Trained CVAE weights + scaler
├── src/
│   ├── data_processing/          # CSV->TRC, preprocessing, resampling
│   ├── inverse_kinematics/       # TRC->MOT, alignment, scaling
│   ├── inverse_dynamics/         # MOT->torques/forces
│   ├── generation/               # CVAE model, train, generate, augment
│   │   └── model.py              # Shared MotionCVAE architecture
│   ├── rl/                       # Imitation learning
│   ├── visualization/            # All rendering, plotting, analysis
│   └── utils/                    # Config loader, shared I/O, transforms
├── scripts/                      # Thin CLI entry points
│   ├── run_pipeline.py           # Batch IK pipeline
│   ├── run_modular_pipeline.py   # Single-file modular pipeline
│   └── run_generated_pipeline.py # CVAE -> IK/ID bridge
├── output/                       # All generated outputs
├── docs/info/                    # Patient docs, data release notes
├── tests/
└── requirements.txt
```

### Configuration

All paths are centralized in `config/settings.yaml` and loaded via `src/utils/config.py`:
```python
from src.utils.config import get_path, get, get_project_root

model_path = get_path("mujoco_arm_model")   # Absolute path from config
data_rate = get("pipeline", "data_rate")     # Config value from any section
```

### Key Settings (config/settings.yaml)
- `paths.mujoco_arm_model` - MuJoCo arm XML
- `paths.reference_mot` - Reference MOT for IK alignment
- `pipeline.data_rate` - 200 Hz
- `pipeline.interactive_align` - Must be false for batch mode

### CVAE Hyperparameters (src/generation/model.py)
```python
INPUT_DIM = 15        # 12 arm + 3 trunk
CONDITION_DIM = 1     # FMA score (normalized 0-66)
HIDDEN_DIM = 256
LATENT_DIM = 32
NUM_HEADS = 4         # Self-attention heads
SEQ_LEN = 100         # Frames per trajectory
```

## Data Formats

**Input CSV**: Multi-level header (marker names + X/Y/Z), 200Hz, millimeters

**TRC**: OpenSim-compatible marker trajectories with header block

**MOT**: Tab-separated joint angles with time column

**Naming Convention**:
- Stroke patients: `S{num}_{session}_{trial}.csv` (e.g., S3_12_1.csv)
- Healthy subjects: `{num}_{session}_{trial}.csv` (e.g., 01_12_1.csv)
- Generated motions: `FMA_{score}.csv` (e.g., FMA_50.csv)

## Critical Implementation Notes

1. **Reference File Priority**: `S5_12_1.mot` must be processed first in batch mode (provides initial pose reference)

2. **Scaler Persistence**: Always save/load the StandardScaler with the CVAE model - same scaler must be used for training AND generation

3. **V_Vector Fix**: CVAE outputs WrVec as direction (WRB-WRA). Model site V_Vector is a position. Must reconstruct: `V_Vector = V_Wrist + WrVec`

4. **Environment**: Use conda env `MyoSuite` at `/home/abdul/miniconda3/envs/MyoSuite/bin/python` (Python 3.9, has dm_control)

5. **IK Error Checking**: Review batch reports for IK convergence errors after batch processing. Typical errors: originals ~18mm, generated ~24-30mm, >200mm = bad sample
