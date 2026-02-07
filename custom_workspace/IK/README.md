# IK (Inverse Kinematics) & Motion Generation Pipeline

This directory contains a comprehensive pipeline for processing, analyzing, and synthesizing human arm kinematic data, specifically tailored for stroke rehabilitation research. The system bridges the gap between raw motion capture data, Inverse Kinematics (IK) for musculoskeletal simulation (MuJoCo/OpenSim), and Generative AI models (VAEs) for data augmentation.

## 🚀 Pipeline Overview

1.  **Data Processing:** Converts raw MHH (Medical School Hannover) CSV kinematic data into standardized TRC (Track Row Column) and MOT (Motion) formats.
2.  **Inverse Kinematics (IK):** Solves joint angles from marker positions using a musculoskeletal model.
3.  **Machine Learning:** Trains Variational Autoencoders (ConvVAE, CVAE, TransformerVAE) to learn the manifold of human arm motion conditioned on FMA (Fugl-Meyer Assessment) scores.
4.  **Generation & Augmentation:** Synthesizes new motion trajectories for specific impairment levels (FMA scores) to augment datasets.
5.  **Visualization & Analysis:** Tools to visualize motions, compare real vs. generated data, and analyze latent spaces and spatial trends.

---

## 📂 File Descriptions

### 1. Main Execution & Pipeline
*   **`run.py`**: The master entry point. Can be configured to run in **Batch Mode** (process all files in a folder) or **Single Mode**. Orchestrates the CSV $\to$ TRC $\to$ MOT $\to$ Video pipeline.
*   **`run_modular.py`**: A cleaner, modularized version of `run.py` that imports from the `modular/` package.
*   **`batch_processor.py`**: Handles the logic for batch processing multiple files. It manages file discovery, prioritization (e.g., ensuring reference files are processed first), and error handling.

### 2. Data Conversion & Preprocessing
*   **`convert_csv2trc.py`**: Converts raw CSV marker data into the TRC format used by OpenSim/MuJoCo. Includes low-pass filtering (`filter_data`).
*   **`convert_trc2mot.py`**: Performs Inverse Kinematics. Reads a TRC file and solves for joint angles to create a `.mot` file. Uses `myosuite` and `mujoco`.
*   **`convert_mot2video.py`**: Renders a `.mot` motion file into an `.mp4` video using the MuJoCo physics engine.
*   **`preprocess_markers.py`**: Extracts and calculates specific kinematic features (Shoulder, Elbow, Wrist positions/vectors) from raw data for ML training.
*   **`trc_data_scaler.py`**: Automatically scales TRC data to match the dimensions of the robot/musculoskeletal model (Retargeting).
*   **`interactive_alignment.py`**: A GUI tool for manually aligning marker data with the model to fix offset issues before IK.

### 3. Generative Models (AI)
These scripts define the neural network architectures used for motion synthesis.
*   **`cvae.py`**: **Conditional Variational Autoencoder (CVAE)**. A dense network that generates trajectories based on FMA scores.
*   **`conv_vae.py`**: **Convolutional CVAE**. Uses 1D convolutions to capture temporal dependencies in motion data.
*   **`transformer_vae.py`**: **Transformer VAE**. Uses self-attention mechanisms to handle long-range temporal correlations and generate high-quality smooth motions.

### 4. Training & Tuning
*   **`train_motion_model.py`**: Trains a statistical model (PCA + SVR) to map FMA scores to motion components.
*   **`train_skeleton_transformer.py`**: Training script for the `TransformerVAE` model.
*   **`auto_tune_conv.py` / `auto_tune_cvae.py` / `auto_tune_trans.py`**: Hyperparameter tuning scripts. They use grid search to find the best learning rates, latent dimensions, and loss weights for their respective models.

### 5. FMA Interpolation & Morphing
Scripts for generating intermediate motion stages between different impairment levels.
*   **`fma_interpolation.py`**: Interpolates between motion trajectories based on FMA scores.
*   **`fma_interpolation_standardised.py`**: Standardized version of the interpolation logic, likely normalizing origins/frames first.
*   **`fma_morph_smooth.py`**: Generates smooth morphed motions between stroke and healthy data.
*   **`fma_skeleton_morph.py`**: Morphs skeleton poses directly.
*   **`fma_skeleton_morph_time.py`**: Accounts for temporal warping (time-series alignment) during morphing.
*   **`cvae_morph.py`**: CVAE-based morphing of motions conditioned on FMA scores.
*   **`cvae_morph_time.py`**: Models and predicts the duration of motions for time-normalized morphing.
*   **`advanced_kinematics_morph.py`**: Advanced analysis and morphing logic for kinematic data.

### 6. Visualization & Analytics
*   **`visualise_processed.py`**: Simple 3D viewer for processed CSV skeleton data.
*   **`visualise_motion_morph.py`**: Visualizes the morphing process between two motions.
*   **`visualise_conv.py` / `visualise_cvae.py` / `visualise_transformer.py`**: Visualizes the output of the trained models (reconstruction quality).
*   **`visualise_conv_analytics.py` / `visualise_cvae_analytics.py` / `visualise_transformer_analytics.py`**: Specific analytics for the models (e.g., loss curves, latent space distribution).
*   **`visualise_transformer_cycle.py`**: Visualizes a cycle of motion generation.
*   **`visualise_skeleton_transformer.py`**: Visualizes the results of the Skeleton Transformer model.
*   **`visualise_spatial_trends.py`**: Visualizes spatial trends across Stroke, Healthy, and Augmented groups.
*   **`cvae_morph_visualise.py`**: Visualizes generated motions from the CVAE morphing model.
*   **`cvae_morph_visualise_real.py`**: Visualizes real motions alongside CVAE generated ones for comparison.

### 7. Comparison & Validation
*   **`compare_real_vs_generated.py`**: Quantitatively compares real patient data vs. AI-generated data (metrics like correlation, jerk, etc.).
*   **`compare_kinematics_morph.py`**: Compares kinematic features between source, target, and morphed motions.
*   **`compare_kinematics_morph_all.py`**: Batch comparison of kinematics morphing across all datasets.
*   **`compare.py`**: General purpose comparison tool for kinematic data.
*   **`calc_mot2invdyn.py`**: Calculates Inverse Dynamics (forces/torques) from motion files to validate physical feasibility.
*   **`sanity_check.py` / `sanity_check1.py` / `sanity_check1_fma.py`**: Basic scripts to verify data integrity and quick model/data tests.

---

## 📁 Subdirectories

### [`processed/`](processed/README.md) — Data Preprocessing & Format Conversion
Scripts that bridge **chest-normalized processed CSVs** (100-frame, 12-column format) to the full IK/ID simulation pipeline. Handles resampling back to original frame counts, TRC conversion with wrist vector support, batch IK solving, inverse dynamics, and video rendering.

| Script | Purpose |
|--------|---------|
| `preprocess_markers.py` | Raw MoCap → 12-column processed CSV (joint centers + chest normalization) |
| `resample_p_time.py` | Restore original frame count from 100-frame normalization |
| `p_convert_csv2trc.py` | Processed CSV → TRC with V_Vector (wrist orientation) |
| `p_convert_trc2mot.py` | Batch IK solver → MOT joint angles via MuJoCo |
| `p_convert_mot2video.py` | MOT → MP4 video rendering |
| `p_calc_mot2invdyn.py` | Inverse dynamics: joint torques and muscle forces |

### [`cutoff/`](cutoff/README.md) — FMA-Targeted Motion Generation (CVAE Pipeline)
The production pipeline for generating synthetic upper-limb motions conditioned on **FMA (Fugl-Meyer Assessment)** scores. Contains the trained BiLSTM+Attention CVAE model, training/generation scripts, and analysis tools.

| Component | Description |
|-----------|-------------|
| `models/` | Trained CVAE checkpoint (`cvae_cutoff_fma_best.pth`) + scaler |
| `scripts/` | Training, generation, verification, trend analysis, interactive viewer |
| `output/` | Generated motions (`FMA_18.csv` to `FMA_66.csv`) + analysis plots |

Key results: the model correctly learns all four clinical trends (reduced ROM, slower velocity, increased trunk compensation, increased jerk at lower FMA scores).

### [`modular/`](modular/) — Reusable IK Utilities
Modular helper package for the IK pipeline: data I/O, IK solver, transforms, and visualization utilities. Imported by `run_modular.py`.

### [`visual/`](visual/) — Visualization & Analysis Scripts
30+ visualization scripts for dashboards, phase selection, spatial trends, manifold plots, and clinical analysis. Includes `new_interactive_phase_selector.py` for manual phase boundary selection and `generate_master_dashboard.py` for unified HTML reports.

---

## 🛠️ Usage Examples

**1. Run the Full IK Pipeline (Batch Mode):**
Open `run.py`, set `RUN_BATCH_MODE = True`, and execute:
```bash
python run.py
```

**2. Train the Transformer VAE:**
```bash
python train_skeleton_transformer.py
```

**3. Visualize a Generated Motion:**
```bash
python visualise_transformer.py
```

**4. Convert a Single CSV to TRC:**
```bash
python convert_csv2trc.py /path/to/input.csv /path/to/output.trc
```
