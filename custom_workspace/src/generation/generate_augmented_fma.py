"""
Generate Augmented Data with Direct FMA Score Targeting
- For each stroke patient (FMA X), generate motions for FMA X+1, X+2, ... 66
- Includes trunk marker interpolation
- Outlier winsorization (p90 clamping on extreme stroke trajectories)
- DTW alignment (temporal phase alignment before morphing)
- DTW-distance weighted morphing (closer healthy templates contribute more)
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import resample
from scipy.spatial.distance import cdist
import os
import sys
import glob
import shutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

# --- Configuration ---
# Raw data (has trunk markers)
RAW_HEALTHY_DIR = get_path("data_healthy")
RAW_STROKE_DIR = get_path("data_stroke")

# Processed cutoff data (arm only)
STROKE_DIR = get_path("data_cutoff_processed")
HEALTHY_DIR = get_path("data_cutoff_processed")

# Output
OUTPUT_DIR = get_path("data_cutoff_augmented")

# Scores
SCORES_FILE = get_path("scores_file")

HEALTHY_FMA = 66
TARGET_LEN = 100  # Resample all to 100 frames
WINSORIZE_PERCENTILE = 90  # p90 threshold for outlier clamping

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


# =============================================================================
# Tier 1 Fix #1: Outlier Winsorization
# =============================================================================

def compute_trajectory_metrics(df_delta):
    """Compute wrist_range_y, trunk_disp, peak_velocity from a delta-format trajectory."""
    vals = df_delta[ALL_COLS].values  # (n_frames, 15)

    # Wrist range in Y (index 7 = Wr_y)
    wr_y = vals[:, 7]
    wrist_range_y = np.ptp(wr_y)

    # Trunk displacement: max Euclidean distance from origin across all frames
    trunk = vals[:, 12:15]  # Trunk_x, Trunk_y, Trunk_z
    trunk_disp = np.max(np.linalg.norm(trunk, axis=1))

    # Peak velocity: max frame-to-frame Euclidean distance of wrist (indices 6,7,8)
    wrist = vals[:, 6:9]
    if len(wrist) > 1:
        frame_diffs = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
        peak_velocity = np.max(frame_diffs)
    else:
        peak_velocity = 0.0

    return wrist_range_y, trunk_disp, peak_velocity


def winsorize_stroke_trajectories(stroke_deltas):
    """
    Clamp extreme stroke trajectories to p90 thresholds.

    Args:
        stroke_deltas: dict of {s_name: df_delta} for all stroke trajectories

    Returns:
        dict of {s_name: df_delta} with outliers winsorized
    """
    if not stroke_deltas:
        return stroke_deltas

    # Compute metrics for all stroke trajectories
    metrics = {}
    for s_name, df in stroke_deltas.items():
        wr_range, trunk_d, peak_v = compute_trajectory_metrics(df)
        metrics[s_name] = (wr_range, trunk_d, peak_v)

    all_wr = np.array([m[0] for m in metrics.values()])
    all_trunk = np.array([m[1] for m in metrics.values()])
    all_vel = np.array([m[2] for m in metrics.values()])

    # p90 thresholds
    thresh_wr = np.percentile(all_wr, WINSORIZE_PERCENTILE)
    thresh_trunk = np.percentile(all_trunk, WINSORIZE_PERCENTILE)
    thresh_vel = np.percentile(all_vel, WINSORIZE_PERCENTILE)

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

        # Scale wrist columns if wrist range exceeds threshold
        if wr_range > thresh_wr and wr_range > 0:
            scale = thresh_wr / wr_range
            wrist_cols = ['Wr_x', 'Wr_y', 'Wr_z', 'WrVec_x', 'WrVec_y', 'WrVec_z']
            for col in wrist_cols:
                df_out[col] = df_out[col] * scale
            clamped = True

        # Scale trunk columns if trunk displacement exceeds threshold
        if trunk_d > thresh_trunk and trunk_d > 0:
            scale = thresh_trunk / trunk_d
            for col in TRUNK_COLS:
                df_out[col] = df_out[col] * scale
            clamped = True

        # Scale all columns if peak velocity exceeds threshold
        if peak_v > thresh_vel and peak_v > 0:
            scale = thresh_vel / peak_v
            for col in ALL_COLS:
                df_out[col] = df_out[col] * scale
            clamped = True

        if clamped:
            n_clamped += 1
            print(f"    Clamped {s_name}: wr={wr_range:.0f}->{thresh_wr:.0f}, "
                  f"trunk={trunk_d:.0f}->{thresh_trunk:.0f}, vel={peak_v:.1f}->{thresh_vel:.1f}")

        winsorized[s_name] = df_out

    print(f"  Winsorized {n_clamped}/{len(stroke_deltas)} stroke trajectories")
    return winsorized


# =============================================================================
# Tier 1 Fix #2: DTW Alignment
# =============================================================================

def dtw_align(seq_a, seq_b):
    """
    Compute DTW warping path between two sequences using pure numpy/scipy.

    Args:
        seq_a: (n, d) array
        seq_b: (m, d) array

    Returns:
        path: list of (i, j) index pairs
        cost: total DTW cost
    """
    n, m = len(seq_a), len(seq_b)

    # Pairwise Euclidean distance matrix
    dist_matrix = cdist(seq_a, seq_b, metric='euclidean')

    # Accumulate cost matrix
    acc = np.full((n, m), np.inf)
    acc[0, 0] = dist_matrix[0, 0]

    for i in range(1, n):
        acc[i, 0] = acc[i-1, 0] + dist_matrix[i, 0]
    for j in range(1, m):
        acc[0, j] = acc[0, j-1] + dist_matrix[0, j]
    for i in range(1, n):
        for j in range(1, m):
            acc[i, j] = dist_matrix[i, j] + min(acc[i-1, j], acc[i, j-1], acc[i-1, j-1])

    # Backtrack to find optimal path
    path = []
    i, j = n - 1, m - 1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [acc[i-1, j-1], acc[i-1, j], acc[i, j-1]]
            argmin = np.argmin(candidates)
            if argmin == 0:
                i -= 1
                j -= 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))

    path.reverse()
    cost = acc[n-1, m-1]
    return path, cost


def dtw_warp_pair(df_a, df_b, target_len):
    """
    DTW-align two trajectories, warp both to the common path, then resample.

    Args:
        df_a: DataFrame (stroke delta, variable length)
        df_b: DataFrame (healthy delta, variable length)
        target_len: resample output length

    Returns:
        df_a_aligned: (target_len, 15)
        df_b_aligned: (target_len, 15)
        dtw_cost: float
    """
    # Resample to TARGET_LEN first for consistent DTW computation
    arr_a = resample_df(df_a, target_len)[ALL_COLS].values  # (target_len, 15)
    arr_b = resample_df(df_b, target_len)[ALL_COLS].values

    path, cost = dtw_align(arr_a, arr_b)

    # Warp both sequences along the path
    warped_a = np.array([arr_a[i] for i, j in path])
    warped_b = np.array([arr_b[j] for i, j in path])

    # Resample warped sequences back to target_len
    warped_a_resampled = np.zeros((target_len, warped_a.shape[1]))
    warped_b_resampled = np.zeros((target_len, warped_b.shape[1]))
    for col_idx in range(warped_a.shape[1]):
        warped_a_resampled[:, col_idx] = resample(warped_a[:, col_idx], target_len)
        warped_b_resampled[:, col_idx] = resample(warped_b[:, col_idx], target_len)

    df_a_aligned = pd.DataFrame(warped_a_resampled, columns=ALL_COLS)
    df_b_aligned = pd.DataFrame(warped_b_resampled, columns=ALL_COLS)

    return df_a_aligned, df_b_aligned, cost


# =============================================================================
# Tier 1 Fix #3: DTW-Distance Weighted Morphing
# =============================================================================

def compute_dtw_weights(dtw_costs):
    """
    Compute softmax weights from DTW distances with adaptive temperature.

    Args:
        dtw_costs: dict of {h_name: dtw_cost}

    Returns:
        dict of {h_name: weight} (sums to 1.0)
    """
    names = list(dtw_costs.keys())
    costs = np.array([dtw_costs[n] for n in names])

    # Adaptive temperature = median distance
    temperature = np.median(costs)
    if temperature < 1e-6:
        # All distances are ~0, equal weights
        weights = np.ones(len(costs)) / len(costs)
    else:
        # Negative exponent: lower cost -> higher weight
        neg_scaled = -costs / temperature
        # Numerical stability
        neg_scaled -= np.max(neg_scaled)
        exp_vals = np.exp(neg_scaled)
        weights = exp_vals / np.sum(exp_vals)

    return {names[i]: weights[i] for i in range(len(names))}


def morph_motion_weighted(df_stroke_aligned, df_healthy_aligned, target_fma, stroke_fma, weight):
    """
    Morph stroke toward healthy with DTW-distance weighting.

    effective_alpha scales the base alpha by the healthy template's weight,
    so dissimilar templates produce morphed motions closer to the stroke input.
    """
    base_alpha = (target_fma - stroke_fma) / (HEALTHY_FMA - stroke_fma)
    base_alpha = np.clip(base_alpha, 0, 1)

    # Weight modulates how far toward healthy we go
    effective_alpha = base_alpha * weight

    morphed = pd.DataFrame()
    for col in ALL_COLS:
        morphed[col] = ((1 - effective_alpha) * df_stroke_aligned[col].values +
                        effective_alpha * df_healthy_aligned[col].values)

    return morphed


# =============================================================================
# Original helpers (unchanged)
# =============================================================================

def morph_motion(df_stroke, df_healthy, target_fma, stroke_fma):
    """
    Morph stroke motion toward healthy motion to achieve target FMA.
    (Kept for reference; main loop now uses morph_motion_weighted)
    """
    alpha = (target_fma - stroke_fma) / (HEALTHY_FMA - stroke_fma)
    alpha = np.clip(alpha, 0, 1)

    df_s = resample_df(df_stroke, TARGET_LEN)
    df_h = resample_df(df_healthy, TARGET_LEN)

    morphed = pd.DataFrame()
    for col in ALL_COLS:
        morphed[col] = (1 - alpha) * df_s[col].values + alpha * df_h[col].values

    return morphed


def backup_augmented_dir():
    """Back up existing augmented data directory before regenerating."""
    if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = OUTPUT_DIR.rstrip('/') + f"_backup_{timestamp}"
        print(f"Backing up existing augmented data to: {backup_dir}")
        shutil.copytree(OUTPUT_DIR, backup_dir)
        # Clear output dir
        for f in glob.glob(os.path.join(OUTPUT_DIR, "*.csv")):
            os.remove(f)
        print(f"  Cleared {OUTPUT_DIR} for fresh generation")


def main():
    print("=" * 60)
    print("Generating FMA-Targeted Augmented Data")
    print("  + Outlier Winsorization (p90)")
    print("  + DTW Temporal Alignment")
    print("  + DTW-Distance Weighted Morphing")
    print("=" * 60)

    # Back up existing augmented data
    backup_augmented_dir()

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

    # =========================================================================
    # Phase 1: Load all stroke trajectories and apply winsorization
    # =========================================================================
    print("\n--- Phase 1: Loading stroke data & winsorization ---")
    stroke_data = {}  # {s_name: (s_file, stroke_fma, df_delta)}

    for s_file in stroke_files:
        s_name = os.path.basename(s_file).replace('.csv', '')

        stroke_fma = score_map.get(s_name)
        if stroke_fma is None:
            stroke_fma = score_map.get(s_name.replace('S', ''))
        if stroke_fma is None:
            print(f"  Skipping {s_name}: no FMA score found")
            continue
        if stroke_fma >= 60:
            print(f"  Skipping {s_name}: FMA {stroke_fma} too high")
            continue

        df_stroke = load_motion_with_trunk(s_file, RAW_STROKE_DIR)
        df_stroke_delta = to_delta_format(df_stroke)
        stroke_data[s_name] = (s_file, stroke_fma, df_stroke_delta)

    # Winsorize
    stroke_deltas = {name: data[2] for name, data in stroke_data.items()}
    winsorized = winsorize_stroke_trajectories(stroke_deltas)

    # Update stroke_data with winsorized deltas
    for s_name in stroke_data:
        s_file, stroke_fma, _ = stroke_data[s_name]
        stroke_data[s_name] = (s_file, stroke_fma, winsorized[s_name])

    # =========================================================================
    # Phase 2: Load all healthy trajectories
    # =========================================================================
    print("\n--- Phase 2: Loading healthy data ---")
    healthy_data = {}  # {h_name: df_delta}

    for h_file in healthy_files:
        h_name = os.path.basename(h_file).replace('.csv', '')
        df_healthy = load_motion_with_trunk(h_file, RAW_HEALTHY_DIR)
        df_healthy_delta = to_delta_format(df_healthy)
        healthy_data[h_name] = df_healthy_delta

    print(f"  Loaded {len(healthy_data)} healthy trajectories")

    # =========================================================================
    # Phase 3: DTW alignment + weighted morphing
    # =========================================================================
    print("\n--- Phase 3: DTW alignment & weighted morphing ---")
    total_generated = 0
    fma_counts = {}

    for s_name, (s_file, stroke_fma, df_stroke_delta) in stroke_data.items():
        print(f"\n{s_name} (FMA {stroke_fma}):")

        # Save original stroke motion (winsorized)
        stroke_out = os.path.join(OUTPUT_DIR, f"{s_name}_FMA{stroke_fma}.csv")
        resample_df(df_stroke_delta, TARGET_LEN).to_csv(stroke_out, index=False)

        # DTW-align with each healthy and collect costs
        aligned_pairs = {}  # {h_name: (df_s_aligned, df_h_aligned)}
        dtw_costs = {}      # {h_name: cost}

        for h_name, df_healthy_delta in healthy_data.items():
            df_s_aligned, df_h_aligned, cost = dtw_warp_pair(
                df_stroke_delta, df_healthy_delta, TARGET_LEN)
            aligned_pairs[h_name] = (df_s_aligned, df_h_aligned)
            dtw_costs[h_name] = cost

        # Compute softmax weights from DTW distances
        weights = compute_dtw_weights(dtw_costs)

        # Normalize weights so max weight = 1.0 (preserves full contribution from closest)
        max_weight = max(weights.values())
        if max_weight > 0:
            weights = {k: v / max_weight for k, v in weights.items()}

        # Generate morphed motions
        for h_name in healthy_data:
            df_s_aligned, df_h_aligned = aligned_pairs[h_name]
            w = weights[h_name]

            for target_fma in range(stroke_fma + 1, HEALTHY_FMA):
                morphed = morph_motion_weighted(
                    df_s_aligned, df_h_aligned, target_fma, stroke_fma, w)

                out_name = f"{s_name}_x_{h_name}_FMA{target_fma}.csv"
                morphed.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)

                total_generated += 1
                fma_counts[target_fma] = fma_counts.get(target_fma, 0) + 1

            print(f"  + {h_name}: FMA {stroke_fma+1}-65 (w={w:.3f}, dtw={dtw_costs[h_name]:.0f})")

    # Save healthy files
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
