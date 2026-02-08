"""
Create cutoff data from raw MoCap CSVs.

Replaces the ad-hoc Step 3 ("preprocess + cut by phase indices") that was never
saved.

Usage:
    python -m src.data_processing.create_cutoff

Configure via the FLAGS section below.
"""
import os
import sys
import json
import glob

import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path, get_project_root
from src.utils.markers import process_raw_to_arrays, ARM_COLS, TRUNK_COLS

# =============================================================================
# FLAGS — edit these to control behaviour
# =============================================================================
DATASET = "both"            # "healthy", "stroke", or "both"
CUT_BY_PHASE = True         # False = process full file without phase slicing
RESAMPLE = True             # Resample to TARGET_FRAMES
TARGET_FRAMES = 100         # Target frame count
INCLUDE_TRUNK = True        # Add Trunk_x/y/z columns (needed for CVAE with 15 dims)
SMOOTH = False              # Low-pass Butterworth filter before saving
CUTOFF_FREQ = 6.0           # Filter cutoff Hz (if SMOOTH=True)
DATA_RATE = 200.0           # Raw data sample rate (Hz)
# =============================================================================


def _output_dir(dataset_label):
    """
    Determine the output directory based on flag combination.

    CUT_BY_PHASE | RESAMPLE | Output
    -------------|----------|-----------------------------------------
    True         | True     | data/kinematic/cutoff/processed/
    True         | False    | data/kinematic/cutoff/original/
    False        | True     | data/kinematic/{dataset_label}/processed/
    False        | False    | data/kinematic/{dataset_label}/processed_full/
    """
    root = get_project_root()
    if CUT_BY_PHASE:
        subdir = "processed" if RESAMPLE else "original"
        return os.path.join(root, "data", "kinematic", "cutoff", subdir)
    else:
        subdir = "processed" if RESAMPLE else "processed_full"
        return os.path.join(root, "data", "kinematic", dataset_label, subdir)


def _load_phase_indices(json_path):
    """Load phase-indices JSON → dict[filename] → [idx0, idx1, idx2]."""
    with open(json_path, "r") as f:
        return json.load(f)


def _butterworth_filter(data, cutoff_freq, fs, order=4):
    """Zero-phase low-pass Butterworth filter applied column-wise."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff_freq / nyq, btype='low')
    filtered = np.empty_like(data)
    for col in range(data.shape[1]):
        filtered[:, col] = filtfilt(b, a, data[:, col])
    return filtered


def _process_single(filepath, phase_indices, dataset_label):
    """
    Process one raw CSV through the pipeline and save the result.

    Returns True on success, False on skip/error.
    """
    fname = os.path.basename(filepath)

    # 1. Extract joint-center arrays via shared utilities
    try:
        data, columns = process_raw_to_arrays(filepath, include_trunk=INCLUDE_TRUNK)
    except ValueError as e:
        print(f"  SKIP {fname}: {e}")
        return False

    # 2. Optionally cut by phase indices
    if CUT_BY_PHASE:
        if fname not in phase_indices:
            print(f"  SKIP {fname}: no phase indices")
            return False
        idx = phase_indices[fname]
        start, end = idx[0], idx[2]
        if end > len(data):
            print(f"  SKIP {fname}: phase end ({end}) > data length ({len(data)})")
            return False
        data = data[start:end]

    # 3. Optionally smooth
    if SMOOTH and len(data) > 12:
        data = _butterworth_filter(data, CUTOFF_FREQ, DATA_RATE)

    # 4. Optionally resample
    if RESAMPLE:
        data = resample(data, TARGET_FRAMES)

    # 5. Save
    out_dir = _output_dir(dataset_label)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, fname)
    df_out = pd.DataFrame(data, columns=columns)
    df_out.to_csv(out_path, index=False)
    return True


def main():
    project_root = get_project_root()

    # Determine which datasets to process
    datasets = []
    if DATASET in ("healthy", "both"):
        datasets.append(("healthy", get_path("data_healthy")))
    if DATASET in ("stroke", "both"):
        datasets.append(("stroke", get_path("data_stroke")))

    # Load phase indices if needed
    phase_healthy = {}
    phase_stroke = {}
    if CUT_BY_PHASE:
        h_json = get_path("output_phase_indices_healthy")
        s_json = get_path("output_phase_indices_stroke")
        if os.path.exists(h_json):
            phase_healthy = _load_phase_indices(h_json)
            print(f"Loaded {len(phase_healthy)} healthy phase indices")
        if os.path.exists(s_json):
            phase_stroke = _load_phase_indices(s_json)
            print(f"Loaded {len(phase_stroke)} stroke phase indices")

    total_ok = 0
    total_skip = 0

    for label, data_dir in datasets:
        if not os.path.isdir(data_dir):
            print(f"Directory not found: {data_dir}")
            continue

        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        phase = phase_healthy if label == "healthy" else phase_stroke

        print(f"\n{'='*50}")
        print(f"Processing {label}: {len(csv_files)} raw CSV files")
        print(f"  CUT_BY_PHASE={CUT_BY_PHASE}  RESAMPLE={RESAMPLE}  "
              f"INCLUDE_TRUNK={INCLUDE_TRUNK}  SMOOTH={SMOOTH}")
        print(f"  Output → {_output_dir(label)}")
        print(f"{'='*50}")

        for fpath in csv_files:
            ok = _process_single(fpath, phase, label)
            if ok:
                total_ok += 1
            else:
                total_skip += 1

    print(f"\nDone.  Saved: {total_ok}  Skipped: {total_skip}")


if __name__ == "__main__":
    main()
