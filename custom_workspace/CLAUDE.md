# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **stroke rehabilitation research platform** that processes motion capture data from stroke patients and healthy subjects, converts it into biomechanically valid motion files, and uses generative AI (Conditional VAEs) to synthesize augmented motion data. The system integrates with MyoSuite/MuJoCo for musculoskeletal simulation and supports RL-based imitation learning.

## Key Commands

### Run the Full IK Pipeline (Batch Mode)
```bash
cd /home/abdul/Desktop/myosuite/custom_workspace
python IK/run.py  # Set RUN_BATCH_MODE = True in the script
```

### Process a Single File
Edit `IK/run.py` and set `RUN_BATCH_MODE = False`, then:
```bash
python IK/run.py
```

### Individual Conversion Steps
```bash
# CSV to TRC (marker trajectories)
python IK/convert_csv2trc.py /path/to/input.csv /path/to/output.trc

# TRC to MOT (inverse kinematics)
python IK/convert_trc2mot.py model/myo_sim/arm/myoarm.xml IK/output/file.trc IK/output/file.mot

# MOT to Video
python IK/convert_mot2video.py model/myo_sim/arm/myoarm.xml IK/output/file.mot IK/output/file.mp4
```

### Train CVAE Model
```bash
python IK/cvae_train_model.py
# Outputs: IK/output/cvae/cvae.pth and IK/output/cvae/scaler.pkl
```

### Generate Synthetic Motion
```bash
python IK/cvae_generate_motion.py 50  # Generate motion for FMA score 50
```

### Train RL Agent (Imitation Learning)
```bash
cd RL
python train_drinking_task.py
```

## Architecture

### Data Pipeline
```
CSV (Raw MHH Mocap) → TRC (Markers) → MOT (Joint Angles) → MP4 (Video)
                                   ↘ Inverse Dynamics (Forces/Torques)
```

### ML Pipeline
```
Augmented Data (FMA_0 to FMA_66 CSVs) → CVAE Training → Motion Generation
```

### Directory Structure
- `data/kinematic/` - Input motion capture data (Healthy/, Stroke/, Augmented/)
- `IK/` - Inverse kinematics pipeline and generative models
- `IK/modular/` - Reusable IK utilities (data_io, ik_solver, transforms, visualization)
- `IK/output/` - Generated TRC, MOT, MP4, and trained models
- `IK/visual/` - Visualization and analysis scripts (~30+)
- `model/myo_sim/arm/` - MuJoCo musculoskeletal model (myoarm.xml)
- `RL/` - Reinforcement learning with imitation learning

### Key Configuration (convert_trc2mot.py)
```python
INTERACTIVE_ALIGN = False  # Must be False for batch processing
SCALE_DATA = True          # Apply retargeting
LOCK_SHOULDER = True       # Shoulder joint constrained
REFERENCE_MOT_PATH = "S5_12_1.mot"  # Initial pose reference
```

### CVAE Hyperparameters (cvae_train_model.py)
```python
INPUT_DIM = 12        # Shoulder(3) + Elbow(3) + Wrist(3) + WristVector(3)
CONDITION_DIM = 1     # FMA score (normalized 0-66)
HIDDEN_DIM = 128
LATENT_DIM = 16
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

3. **IK Error Checking**: Review `IK/output/batch_report.csv` for IK convergence errors after batch processing

4. **Visualization Dashboard**: Run `python IK/visual/generate_master_dashboard.py` to create a unified HTML report at `IK/visual/master_dashboard.html`
