"""
Generate Augmented Data via Linear Interpolation
- For each stroke patient (FMA X), generate motions for FMA X+1, X+2, ... 65
- Simple linear blend: morphed = (1-alpha)*stroke + alpha*healthy
  where alpha = (target_fma - stroke_fma) / (66 - stroke_fma)
- No DTW alignment, no weighted morphing — pure linear baseline
- Outlier winsorization (p90 clamping) applied to stroke trajectories
  for consistency with the DTW pipeline

Output: data/kinematic/cutoff/augmented_linear/
Naming: {stroke}_x_{healthy}_FMA{n}.csv  (same as DTW convention)
"""

import numpy as np
import pandas as pd
from scipy.signal import resample
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

# --- Configuration ---
RAW_HEALTHY_DIR = get_path("data_healthy")
RAW_STROKE_DIR  = get_path("data_stroke")
PROCESSED_DIR   = get_path("data_cutoff_processed")
OUTPUT_DIR      = get_path("data_cutoff_augmented_linear")
SCORES_FILE     = get_path("scores_file")

HEALTHY_FMA          = 66
TARGET_LEN           = 100
WINSORIZE_PERCENTILE = 90

ARM_COLS  = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
             'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x','Trunk_y','Trunk_z']
ALL_COLS   = ARM_COLS + TRUNK_COLS

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Data loading
# =============================================================================

def load_raw_trunk(raw_path):
    """Extract trunk centroid from raw MoCap data (CS markers)."""
    try:
        df_raw = pd.read_csv(raw_path, header=[0, 1])
        new_cols = []
        current_marker = None
        for c0, c1 in df_raw.columns:
            c0_clean = c0.strip() if isinstance(c0, str) else c0
            c1_clean = c1.strip() if isinstance(c1, str) else c1
            if not str(c0_clean).startswith('Unnamed'):
                current_marker = c0_clean
            new_cols.append((current_marker, c1_clean))
        df_raw.columns = pd.MultiIndex.from_tuples(new_cols)
        cs_markers = []
        for marker in ['CS_1', 'CS_2', 'CS_3', 'CS_4']:
            if marker in df_raw.columns.get_level_values(0):
                cs_data = df_raw[marker][['X', 'Y', 'Z']].values.astype(float)
                cs_markers.append(cs_data)
        if len(cs_markers) >= 2:
            return np.mean(cs_markers, axis=0)
        return None
    except Exception:
        return None


def load_motion_with_trunk(processed_path, raw_dir):
    """Load processed arm data and add trunk from raw."""
    df = pd.read_csv(processed_path)
    for col in ARM_COLS:
        if col not in df.columns:
            df[col] = 0.0
    fname = os.path.basename(processed_path)
    trunk = load_raw_trunk(os.path.join(raw_dir, fname))
    if trunk is not None:
        if len(trunk) != len(df):
            trunk = resample(trunk, len(df))
        df['Trunk_x'] = trunk[:, 0]
        df['Trunk_y'] = trunk[:, 1]
        df['Trunk_z'] = trunk[:, 2]
    else:
        df['Trunk_x'] = 0.0
        df['Trunk_y'] = 0.0
        df['Trunk_z'] = 0.0
    return df[ALL_COLS]


def to_delta_format(df):
    delta = df.copy()
    first = df.iloc[0].copy()
    for col in ALL_COLS:
        delta[col] = df[col] - first[col]
    return delta


def resample_df(df, target_len):
    return pd.DataFrame(
        {col: resample(df[col].values, target_len) for col in df.columns},
        columns=df.columns
    )


# =============================================================================
# Winsorization (same as DTW pipeline for consistency)
# =============================================================================

def compute_trajectory_metrics(df_delta):
    vals = df_delta[ALL_COLS].values
    wr_y        = vals[:, 7]
    wrist_range = np.ptp(wr_y)
    trunk       = vals[:, 12:15]
    trunk_disp  = np.max(np.linalg.norm(trunk, axis=1))
    wrist       = vals[:, 6:9]
    peak_vel    = np.max(np.linalg.norm(np.diff(wrist, axis=0), axis=1)) if len(wrist) > 1 else 0.0
    return wrist_range, trunk_disp, peak_vel


def winsorize_stroke_trajectories(stroke_deltas):
    if not stroke_deltas:
        return stroke_deltas
    metrics = {n: compute_trajectory_metrics(df) for n, df in stroke_deltas.items()}
    thresh_wr    = np.percentile([m[0] for m in metrics.values()], WINSORIZE_PERCENTILE)
    thresh_trunk = np.percentile([m[1] for m in metrics.values()], WINSORIZE_PERCENTILE)
    thresh_vel   = np.percentile([m[2] for m in metrics.values()], WINSORIZE_PERCENTILE)

    print(f"\n  Winsorization p{WINSORIZE_PERCENTILE} thresholds:")
    print(f"    Wrist range Y: {thresh_wr:.1f} mm")
    print(f"    Trunk disp:    {thresh_trunk:.1f} mm")
    print(f"    Peak velocity: {thresh_vel:.1f} mm/frame")

    winsorized = {}
    n_clamped = 0
    for s_name, df in stroke_deltas.items():
        wr_range, trunk_d, peak_v = metrics[s_name]
        df_out = df.copy()
        clamped = False
        if wr_range > thresh_wr and wr_range > 0:
            scale = thresh_wr / wr_range
            for col in ['Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']:
                df_out[col] *= scale
            clamped = True
        if trunk_d > thresh_trunk and trunk_d > 0:
            for col in TRUNK_COLS:
                df_out[col] *= thresh_trunk / trunk_d
            clamped = True
        if peak_v > thresh_vel and peak_v > 0:
            for col in ALL_COLS:
                df_out[col] *= thresh_vel / peak_v
            clamped = True
        if clamped:
            n_clamped += 1
        winsorized[s_name] = df_out

    print(f"  Winsorized {n_clamped}/{len(stroke_deltas)} stroke trajectories")
    return winsorized


# =============================================================================
# Linear morphing
# =============================================================================

def morph_linear(df_stroke, df_healthy, target_fma, stroke_fma):
    """
    Pure linear blend between resampled stroke and healthy trajectories.
    alpha = (target_fma - stroke_fma) / (66 - stroke_fma)
    """
    alpha = (target_fma - stroke_fma) / (HEALTHY_FMA - stroke_fma)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    df_s = resample_df(df_stroke, TARGET_LEN)
    df_h = resample_df(df_healthy, TARGET_LEN)

    morphed = pd.DataFrame(
        {col: (1.0 - alpha) * df_s[col].values + alpha * df_h[col].values
         for col in ALL_COLS},
        columns=ALL_COLS
    )
    return morphed


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Generating Linear Interpolation Augmented Data")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Load FMA scores
    scores_df = pd.read_csv(SCORES_FILE)
    score_map = {}
    for _, row in scores_df.iterrows():
        name = str(row.iloc[0]).replace('.mot', '').replace('.csv', '').strip()
        score_map[name] = int(row.iloc[1])
    print(f"Loaded {len(score_map)} FMA scores")

    stroke_files  = sorted(glob.glob(os.path.join(PROCESSED_DIR, "S*.csv")))
    healthy_files = sorted([f for f in glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
                            if not os.path.basename(f).startswith('S')])

    print(f"Stroke files:  {len(stroke_files)}")
    print(f"Healthy files: {len(healthy_files)}")

    if not stroke_files or not healthy_files:
        print("ERROR: No files found!")
        return

    # --- Phase 1: Load stroke + winsorize ---
    print("\n--- Phase 1: Loading stroke data & winsorization ---")
    stroke_data = {}  # {s_name: (stroke_fma, df_delta)}

    for s_file in stroke_files:
        s_name = os.path.basename(s_file).replace('.csv', '')
        stroke_fma = score_map.get(s_name, score_map.get(s_name.replace('S', '')))
        if stroke_fma is None:
            print(f"  Skipping {s_name}: no FMA score")
            continue
        if stroke_fma >= 60:
            print(f"  Skipping {s_name}: FMA {stroke_fma} too high")
            continue
        df = load_motion_with_trunk(s_file, RAW_STROKE_DIR)
        stroke_data[s_name] = (stroke_fma, to_delta_format(df))

    stroke_deltas = {n: d[1] for n, d in stroke_data.items()}
    winsorized    = winsorize_stroke_trajectories(stroke_deltas)
    for s_name in stroke_data:
        fma, _ = stroke_data[s_name]
        stroke_data[s_name] = (fma, winsorized[s_name])

    # --- Phase 2: Load healthy ---
    print("\n--- Phase 2: Loading healthy data ---")
    healthy_data = {}  # {h_name: df_delta}
    for h_file in healthy_files:
        h_name = os.path.basename(h_file).replace('.csv', '')
        df = load_motion_with_trunk(h_file, RAW_HEALTHY_DIR)
        healthy_data[h_name] = to_delta_format(df)
    print(f"  Loaded {len(healthy_data)} healthy trajectories")

    # Clear output directory
    existing = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    if existing:
        print(f"\n  Clearing {len(existing)} existing files in {OUTPUT_DIR}")
        for f in existing:
            os.remove(f)

    # --- Phase 3: Linear morphing ---
    print("\n--- Phase 3: Linear interpolation ---")
    total_generated = 0
    fma_counts = {}

    for s_name, (stroke_fma, df_stroke_delta) in stroke_data.items():
        print(f"\n{s_name} (FMA {stroke_fma}):")

        # Save original stroke (winsorized, resampled)
        stroke_out = os.path.join(OUTPUT_DIR, f"{s_name}_FMA{stroke_fma}.csv")
        resample_df(df_stroke_delta, TARGET_LEN).to_csv(stroke_out, index=False)

        for h_name, df_healthy_delta in healthy_data.items():
            for target_fma in range(stroke_fma + 1, HEALTHY_FMA):
                morphed = morph_linear(df_stroke_delta, df_healthy_delta, target_fma, stroke_fma)
                out_name = f"{s_name}_x_{h_name}_FMA{target_fma}.csv"
                morphed.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
                total_generated += 1
                fma_counts[target_fma] = fma_counts.get(target_fma, 0) + 1

        print(f"  Generated FMA {stroke_fma+1}–65 × {len(healthy_data)} healthy templates")

    # Save healthy reference files
    print("\n--- Saving healthy reference files ---")
    for h_name, df_healthy_delta in healthy_data.items():
        healthy_out = os.path.join(OUTPUT_DIR, f"{h_name}_FMA66.csv")
        resample_df(df_healthy_delta, TARGET_LEN).to_csv(healthy_out, index=False)

    print("\n" + "=" * 60)
    print(f"DONE! Generated {total_generated} augmented files")
    print(f"Output: {OUTPUT_DIR}")
    print("\nFMA Distribution:")
    for fma in sorted(fma_counts.keys()):
        print(f"  FMA {fma}: {fma_counts[fma]} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
