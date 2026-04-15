# Project Overview

This repository contains a Stroke Rehabilitation Pipeline. It is a research platform that processes motion capture data from stroke patients and healthy subjects, generates synthetic motion using Conditional Variational Autoencoders (CVAEs), and runs musculoskeletal simulations using MyoSuite and MuJoCo. The project incorporates data preprocessing, Inverse Kinematics (IK), Inverse Dynamics (ID), visualization, and Imitation Learning (Reinforcement Learning) for a reaching/drinking task.

**Main Technologies:**
- Python 3.9
- PyTorch
- MuJoCo & MyoSuite (Musculoskeletal Simulation)
- Stable Baselines 3 (Reinforcement Learning)
- Scikit-learn, Pandas, NumPy, Matplotlib

**Architecture:**
The pipeline consists of several interconnected modules:
1. **Data Processing**: Converts raw CSV motion capture data into OpenSim-compatible TRC format and extracts reaching phase segments.
2. **CVAE Generation**: Trains on augmented data to synthesize novel kinematic trajectories based on Fugl-Meyer Assessment (FMA) scores.
3. **Inverse Kinematics (IK)**: Uses MuJoCo to convert TRC marker trajectories into MOT files containing biomechanically valid joint angles.
4. **Inverse Dynamics (ID)**: Computes joint torques and forces from MOT files.
5. **Visualization**: Renders MOT files to MP4 videos and provides analytical dashboards.
6. **Reinforcement Learning (RL)**: Trains an agent via imitation learning to perform tasks by tracking the generated trajectories.

# Building and Running

### Prerequisites
The project requires a specific Conda environment:
```bash
conda activate MyoSuite # Python 3.9 environment with dm_control, mujoco, torch
```

### Full Batch Pipeline
To process original motion capture recordings in batch mode:
```bash
python scripts/run_pipeline.py
```

### Modular Pipeline (Single File)
To process a single file through CSV -> TRC -> MOT -> MP4:
```bash
python scripts/run_modular_pipeline.py
```

### Generative AI Pipeline
Train the Conditional VAE on augmented motion data:
```bash
python src/generation/cvae_train.py -n 15000 -e 300
```
Generate synthetic motion for a specific FMA score (e.g., 50):
```bash
python src/generation/cvae_generate.py 50
```
Process generated CVAE output through the IK/ID pipeline:
```bash
python scripts/run_generated_pipeline.py
```

### Individual Steps
- **CSV to TRC**: `python src/data_processing/convert_csv2trc.py <input.csv> <output.trc>`
- **TRC to MOT (IK)**: `python src/inverse_kinematics/convert_trc2mot.py <model.xml> <input.trc> <output.mot>`
- **MOT to Video**: `python src/visualization/convert_mot2video.py <model.xml> <input.mot> <output.mp4>`
- **MOT to Inverse Dynamics**: `python src/inverse_dynamics/calc_mot2invdyn.py`

# Development Conventions

- **Centralized Configuration**: All paths and hyperparameters must be accessed via the central configuration loader (`src/utils/config.py`), reading from `config/settings.yaml`. Hardcoded paths should be avoided.
- **Reference File**: For IK batch processing, the file `S5_12_1.mot` must be processed first as it provides the initial pose reference.
- **V_Vector Fix**: When modifying or debugging CVAE generation, note that the model outputs wrist direction vectors (WRB-WRA) which must be reconstructed as `V_Vector = V_Wrist + WrVec` for the IK solver.
- **Scaler Persistence**: The `StandardScaler` must always be saved and loaded with the CVAE model to ensure data is identically scaled during both training and generation phases.
- **Data Formats**:
  - `CSV`: Raw MHH MoCap (Multi-level headers, 200 Hz).
  - `TRC`: OpenSim-compatible marker trajectories.
  - `MOT`: Tab-separated joint angles.
- **Naming Conventions**:
  - Healthy Subjects: `{num}_{session}_{trial}.csv`
  - Stroke Patients: `S{num}_{session}_{trial}.csv`
  - Generated Motions: `FMA_{score}.csv`