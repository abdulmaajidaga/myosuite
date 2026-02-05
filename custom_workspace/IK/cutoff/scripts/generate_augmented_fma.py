"""
Generate Augmented Data with Direct FMA Score Targeting
- For each stroke patient (FMA X), generate motions for FMA X+1, X+2, ... 66
- Includes trunk marker interpolation
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import resample
import os
import glob

# --- Configuration ---
BASE_DIR = "/home/abdul/Desktop/myosuite/custom_workspace"

# Raw data (has trunk markers)
RAW_HEALTHY_DIR = os.path.join(BASE_DIR, "data/kinematic/Healthy")
RAW_STROKE_DIR = os.path.join(BASE_DIR, "data/kinematic/Stroke")

# Processed cutoff data (arm only)
STROKE_DIR = os.path.join(BASE_DIR, "data/kinematic/cutoff/processed")
HEALTHY_DIR = os.path.join(BASE_DIR, "data/kinematic/cutoff/processed")

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, "data/kinematic/cutoff/augmented")

# Scores
SCORES_FILE = os.path.join(BASE_DIR, "IK/output/scores.csv")

HEALTHY_FMA = 66
TARGET_LEN = 100  # Resample all to 100 frames

# Columns
ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
ALL_COLS = ARM_COLS + TRUNK_COLS

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_raw_trunk(raw_path):
    """Extract trunk centroid from raw MoCap data (CS markers)."""
    try:
        df_raw = pd.read_csv(raw_path, header=[0, 1])

        # Fix column names (leading spaces, Unnamed for Y/Z)
        new_cols = []
        current_marker = None
        for c0, c1 in df_raw.columns:
            c0_clean = c0.strip() if isinstance(c0, str) else c0
            c1_clean = c1.strip() if isinstance(c1, str) else c1
            if not str(c0_clean).startswith('Unnamed'):
                current_marker = c0_clean
            new_cols.append((current_marker, c1_clean))
        df_raw.columns = pd.MultiIndex.from_tuples(new_cols)

        # Extract CS markers
        cs_markers = []
        for marker in ['CS_1', 'CS_2', 'CS_3', 'CS_4']:
            if marker in df_raw.columns.get_level_values(0):
                cs_data = df_raw[marker][['X', 'Y', 'Z']].values.astype(float)
                cs_markers.append(cs_data)

        if len(cs_markers) >= 2:
            return np.mean(cs_markers, axis=0)  # (n_frames, 3)
        return None
    except:
        return None


def load_motion_with_trunk(processed_path, raw_dir):
    """Load processed arm data and add trunk from raw."""
    # Load arm data
    df = pd.read_csv(processed_path)
    for col in ARM_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Load trunk from raw
    fname = os.path.basename(processed_path)
    raw_path = os.path.join(raw_dir, fname)
    trunk = load_raw_trunk(raw_path)

    if trunk is not None:
        # Resample trunk to match arm length
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
    """Convert to delta format (first frame = 0)."""
    delta = df.copy()
    first = df.iloc[0].copy()
    for col in ALL_COLS:
        delta[col] = df[col] - first[col]
    return delta


def resample_df(df, target_len):
    """Resample all columns to target length."""
    new_data = {}
    for col in df.columns:
        new_data[col] = resample(df[col].values, target_len)
    return pd.DataFrame(new_data, columns=df.columns)


def morph_motion(df_stroke, df_healthy, target_fma, stroke_fma):
    """
    Morph stroke motion toward healthy motion to achieve target FMA.

    alpha = 0 -> pure stroke
    alpha = 1 -> pure healthy
    """
    alpha = (target_fma - stroke_fma) / (HEALTHY_FMA - stroke_fma)
    alpha = np.clip(alpha, 0, 1)

    # Resample both to same length
    df_s = resample_df(df_stroke, TARGET_LEN)
    df_h = resample_df(df_healthy, TARGET_LEN)

    # Linear interpolation
    morphed = pd.DataFrame()
    for col in ALL_COLS:
        morphed[col] = (1 - alpha) * df_s[col].values + alpha * df_h[col].values

    return morphed


def main():
    print("=" * 60)
    print("Generating FMA-Targeted Augmented Data")
    print("=" * 60)

    # Load FMA scores
    try:
        scores_df = pd.read_csv(SCORES_FILE)
        score_map = {}
        for _, row in scores_df.iterrows():
            name = str(row.iloc[0]).replace('.mot', '').replace('.csv', '').strip()
            score_map[name] = int(row.iloc[1])
        print(f"Loaded {len(score_map)} FMA scores")
    except Exception as e:
        print(f"Error loading scores: {e}")
        return

    # Get files
    stroke_files = sorted([f for f in glob.glob(os.path.join(STROKE_DIR, "S*.csv"))])
    healthy_files = sorted([f for f in glob.glob(os.path.join(HEALTHY_DIR, "*.csv"))
                           if not os.path.basename(f).startswith('S')])

    print(f"Stroke files: {len(stroke_files)}")
    print(f"Healthy files: {len(healthy_files)}")

    if not stroke_files or not healthy_files:
        print("ERROR: No files found!")
        return

    # Determine raw directories for trunk
    total_generated = 0
    fma_counts = {}

    for s_file in stroke_files:
        s_name = os.path.basename(s_file).replace('.csv', '')

        # Get stroke FMA score
        stroke_fma = score_map.get(s_name)
        if stroke_fma is None:
            # Try without S prefix
            stroke_fma = score_map.get(s_name.replace('S', ''))
        if stroke_fma is None:
            print(f"  Skipping {s_name}: no FMA score found")
            continue

        if stroke_fma >= 60:
            print(f"  Skipping {s_name}: FMA {stroke_fma} too high")
            continue

        print(f"\n{s_name} (FMA {stroke_fma}):")

        # Load stroke motion with trunk
        df_stroke = load_motion_with_trunk(s_file, RAW_STROKE_DIR)
        df_stroke_delta = to_delta_format(df_stroke)

        # Save original stroke motion
        stroke_out = os.path.join(OUTPUT_DIR, f"{s_name}_FMA{stroke_fma}.csv")
        resample_df(df_stroke_delta, TARGET_LEN).to_csv(stroke_out, index=False)

        # Pair with each healthy file
        for h_file in healthy_files:
            h_name = os.path.basename(h_file).replace('.csv', '')

            # Load healthy motion with trunk
            df_healthy = load_motion_with_trunk(h_file, RAW_HEALTHY_DIR)
            df_healthy_delta = to_delta_format(df_healthy)

            # Generate each FMA score from stroke+1 to 65
            for target_fma in range(stroke_fma + 1, HEALTHY_FMA):
                morphed = morph_motion(df_stroke_delta, df_healthy_delta, target_fma, stroke_fma)

                # Save
                out_name = f"{s_name}_x_{h_name}_FMA{target_fma}.csv"
                morphed.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)

                total_generated += 1
                fma_counts[target_fma] = fma_counts.get(target_fma, 0) + 1

            print(f"  + {h_name}: FMA {stroke_fma+1}-65")

        # Also save healthy files
        for h_file in healthy_files:
            h_name = os.path.basename(h_file).replace('.csv', '')
            df_healthy = load_motion_with_trunk(h_file, RAW_HEALTHY_DIR)
            df_healthy_delta = to_delta_format(df_healthy)
            healthy_out = os.path.join(OUTPUT_DIR, f"{h_name}_FMA66.csv")
            if not os.path.exists(healthy_out):
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
