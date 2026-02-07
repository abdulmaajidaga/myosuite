# Processed: Data Preprocessing & Format Conversion

This directory contains scripts that bridge the gap between **chest-normalized processed CSVs** (100-frame, 12-column format) and the **IK/ID simulation pipeline** (TRC, MOT, inverse dynamics, video). These scripts handle resampling back to original frame counts, converting to TRC format, running inverse kinematics, computing inverse dynamics, and rendering videos.

---

## Pipeline Overview

```
cutoff/processed/*.csv (100 frames, 12 cols)
        |
        v
  [preprocess_markers.py]  <-- Raw MoCap -> processed CSV (standalone preprocessor)
        |
        v
  [resample_p_time.py]     <-- Restore original frame count from raw files
        |
        v
  [p_convert_csv2trc.py]   <-- Convert to TRC (with optional filtering)
        |
        v
  [p_convert_trc2mot.py]   <-- Inverse kinematics via MuJoCo/MyoSuite
        |
        v
  [p_convert_mot2video.py] <-- Render MOT to MP4
        |
  [p_calc_mot2invdyn.py]   <-- Compute joint torques and muscle forces
```

---

## Scripts

### `preprocess_markers.py`
Extracts virtual joint centers from raw multi-marker MoCap CSVs and produces the 12-column processed format.

- **Input**: Raw MHH CSVs from `data/kinematic/Healthy/` and `data/kinematic/Stroke/` (multi-level headers, 200Hz)
- **Output**: `data/kinematic/{Healthy,Stroke}/processed/*_processed.csv` (100 frames, 12 columns)
- **Process**:
  1. Parses multi-level CSV headers to locate marker columns
  2. Computes joint centers: Chest (CS_1-4 avg), Shoulder (SA_1-3 avg), Elbow (ELB_L/M avg), Wrist (WRA/WRB avg)
  3. Normalizes by chest centroid (subtracts chest center from Sh, El, Wr)
  4. Computes wrist vector: `WrVec = WRB - WRA` (directional, not normalized by position)
  5. Resamples to 100 frames via `scipy.signal.resample`

### `resample_p_time.py`
Restores temporal information by resampling 100-frame data back to the original frame count.

- **Input**: Processed 100-frame CSVs + raw CSVs (to read original frame count)
- **Output**: `IK/output/trc/p_originals/*.trc`
- **Why**: The 100-frame normalization strips timing. This script reads the raw file to determine the original frame count, then upsamples back so IK and video playback reflect real movement speed at 200Hz.

### `p_convert_csv2trc.py`
Converts processed CSVs to TRC format with wrist vector support.

- **Input**: Processed 100-frame CSVs + raw files (for frame count restoration)
- **Output**: `IK/output/trc/p_originals_w_vector/*.trc`
- **Features**:
  - Restores original frame count before writing TRC
  - Includes V_Vector marker (wrist orientation) in the TRC
  - Optional low-pass Butterworth filter (6Hz cutoff)

### `p_convert_trc2mot.py`
Batch IK solver that converts TRC files to MOT joint angle files using the MuJoCo musculoskeletal model.

- **Input**: TRC files from `IK/output/trc/` (originals or augmented)
- **Output**: `IK/output/mot/` (originals or augmented)
- **Configuration**:
  - `SCALE_DATA = True` (retarget markers to model dimensions)
  - `LOCK_SHOULDER = True` (constrain shoulder base joints)
  - Uses `interactive_alignment` and `trc_data_scaler` helpers from parent `IK/` directory

### `p_convert_mot2video.py`
Renders MOT motion files to MP4 video using the MuJoCo renderer.

- **Input**: MOT files (single file or batch from folder)
- **Output**: MP4 videos in `IK/output/videos/`
- **Camera**: Side view (azimuth=90, elevation=-30, distance=1.5)

### `p_calc_mot2invdyn.py`
Computes inverse dynamics (joint torques and muscle forces) from MOT files.

- **Input**: MOT files from `IK/output/mot/p_originals/`
- **Output**: `IK/output/ID_results/` (torque plots, muscle activation videos)
- **Features**:
  - OSQP-based muscle force optimization
  - Per-joint torque decomposition with low-pass filtering (1.5Hz)
  - Shoulder torque scaling fix (`1e-5` for locked shoulder joints)
  - Auto-calibrates elbow torques to ~12 Nm peak

---

## Output Locations

| Script | Output Directory |
|--------|-----------------|
| `preprocess_markers.py` | `data/kinematic/{Healthy,Stroke}/processed/` |
| `resample_p_time.py` | `IK/output/trc/p_originals/` |
| `p_convert_csv2trc.py` | `IK/output/trc/p_originals_w_vector/` |
| `p_convert_trc2mot.py` | `IK/output/mot/{p_originals,augmented}/` |
| `p_convert_mot2video.py` | `IK/output/videos/{p_originals,augmented}/` |
| `p_calc_mot2invdyn.py` | `IK/output/ID_results/` |

---

## Usage

All scripts can be run from the workspace root:

```bash
cd /home/abdul/Desktop/myosuite/custom_workspace

# Step 1: Preprocess raw MoCap to 12-column format
python IK/processed/preprocess_markers.py

# Step 2: Resample back to original frames and write TRC
python IK/processed/resample_p_time.py

# Step 3: Run IK to generate MOT files (requires MyoSuite conda env)
conda activate MyoSuite
python IK/processed/p_convert_trc2mot.py

# Step 4: Render videos
python IK/processed/p_convert_mot2video.py

# Step 5: Compute inverse dynamics
python IK/processed/p_calc_mot2invdyn.py
```

## Dependencies

- **Python 3.9** (MyoSuite conda environment)
- `myosuite`, `mujoco` (IK solver and rendering)
- `scipy` (resampling, filtering)
- `pandas`, `numpy` (data handling)
- `skvideo` (video rendering)
- `osqp` (inverse dynamics muscle optimization)
