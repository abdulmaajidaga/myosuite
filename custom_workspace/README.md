# Custom Workspace for MyoSuite

This workspace contains custom scripts and tools for processing kinematic data (Inverse Kinematics) and training Reinforcement Learning (RL) agents within the MyoSuite environment.

## Directory Structure

```
custom_workspace/
├── alignment.json          # Configuration for coordinate alignment
├── data/                   # Input data (Kinematics, etc.)
│   └── kinematic/          # CSV files from motion capture (e.g., Stroke patients)
├── IK/                     # Inverse Kinematics Pipeline
│   ├── run.py              # MASTER SCRIPT: Main entry point for the pipeline
│   ├── run_modular.py      # Modularized version of the IK pipeline
│   ├── batch_processor.py  # Logic for batch processing multiple files
│   ├── convert_*.py        # Individual conversion scripts (CSV->TRC->MOT->Video)
│   ├── modular/            # Helper Python package for IK operations
│   └── output/             # Generated files (.trc, .mot, .mp4)
├── model/                  # Simulation models (MJCF/XML)
└── RL/                     # Reinforcement Learning experiments
    └── train_drinking_task.py  # PPO training with imitation learning
```

## Inverse Kinematics (IK) Pipeline

The IK pipeline converts raw motion capture data (CSV) into OpenSim/Mujoco compatible motion files (.mot) and renders videos.

### IK Scripts Description

The `IK/` directory contains several scripts for processing kinematic data, scaling models, and analyzing motions:

*   **`run.py`**: The main entry point. It orchestrates the full pipeline: CSV -> TRC -> MOT -> MP4. It can run in single-file mode or batch mode.
*   **`run_modular.py`**: A refactored, modular version of the IK pipeline that uses the `modular/` sub-package for cleaner logic separation.
*   **`batch_processor.py`**: Contains the logic used by `run.py` to iterate through multiple files, ensuring specific reference files are processed first.
*   **`convert_csv2trc.py`**: Converts raw motion capture CSV data (with multi-level headers) into OpenSim-compatible TRC marker files.
*   **`convert_trc2mot.py`**: The core IK solver. It uses MuJoCo to find joint angles that best match the marker positions in the TRC file, producing a MOT file.
*   **`convert_mot2video.py`**: Loads a MOT file and a MuJoCo model, then renders the motion to an MP4 video file.
*   **`trc_data_scaler.py`**: Scales the TRC marker data itself to match the dimensions of a specific MuJoCo model, useful for retargeting.
*   **`calc_mot2invdyn.py`**: Performs Inverse Dynamics. It calculates the forces/torques required to produce the motion described in a MOT file.
*   **`interactive_alignment.py`**: Provides a GUI (using MuJoCo viewer) to manually align the motion capture coordinate system with the MuJoCo model's frame.
*   **`train_motion_model.py`**: Trains a statistical model (using PCA and Support Vector Regression) to generate new motions based on clinical scores (like FMA-UE).
*   **`IK/visual/visualise_all.py`**: A unified dashboard that generates both static model quality checks (PCA variance, regression paths) and interactive 3D manifold comparisons (PCA vs t-SNE vs UMAP) to analyze patient clusters.
*   **`IK/visual/visualise_similarity.py`**: Uses **Dynamic Time Warping (DTW)** to calculate the exact similarity between every pair of patient motions, generating a clustered heatmap to reveal subgroups with similar movement strategies.
*   **`IK/visual/visualise_trajectories.py`**: Generates **temporal plots** (joint angles over time) and **spatial plots** (3D wrist paths), grouping patients by impairment level (Severe, Moderate, Healthy) to highlight qualitative differences.
*   **`IK/visual/visualise_metrics.py`**: Calculates clinical scalar metrics (Range of Motion, Peak Velocity, Path Efficiency) and plots their correlation with FMA scores to quantify recovery trends.
*   **`IK/visual/visualise_trc_manifolds.py`**: Applies PCA/t-SNE/UMAP directly to the **raw TRC marker data** (instead of calculated joint angles). Comparing this with `visualise_all.py` helps verify if the Inverse Kinematics process is preserving the underlying motion structure.
*   **`IK/visual/visualise_csv_manifolds.py`**: Visualizes the **raw source CSV data** (MHH format) using the same manifold techniques. This is the earliest possible check to see if patient clustering exists in the original recording before any conversion or processing.
*   **`IK/visual/generate_master_dashboard.py`**: Aggregates all the above visualizations (interactive plots and static images) into a single HTML file (`master_dashboard.html`) for a comprehensive overview of the entire dataset.

## Data Visualization

The visualization tools are located in `custom_workspace/IK/visual/`.

1.  **Raw Data**: `visualise_csv_manifolds.py` (Source CSV) & `visualise_trc_manifolds.py` (Markers).
2.  **Processed Motion**: `visualise_all.py` (Joint Angles PCA/t-SNE/UMAP).
3.  **Temporal & Spatial**: `visualise_trajectories.py` (Joint curves & 3D paths).
4.  **Clinical Metrics**: `visualise_metrics.py` (ROM, Velocity correlations).
5.  **Similarity**: `visualise_similarity.py` (Patient-to-patient correlation heatmap).
6.  **Master Dashboard**: `generate_master_dashboard.py`.

The final output is: **`custom_workspace/IK/visual/master_dashboard.html`**.

## Reinforcement Learning (RL)

The `RL` directory contains scripts for training agents to perform tasks.

### Training (`RL/train_drinking_task.py`)

This script trains a PPO agent (using Stable Baselines 3) to perform a drinking task. It uses **Imitation Learning** by rewarding the agent for tracking a reference motion (.mot file).

**Usage:**

```bash
cd custom_workspace/RL
python train_drinking_task.py
```

*   **Input**: Requires a valid `.mot` file (generated by the IK pipeline) to use as a reference.
*   **Environment**: Uses `myo_sim` environments (e.g., `RelocateEnvV0`).

## Alignment

`alignment.json` contains manual offsets (x, y, z) and rotations (r, p, yw) used to align the motion capture data with the MyoSuite model frame.
