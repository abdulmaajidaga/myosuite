"""
SMOTE-Based Augmentation for Stroke Rehabilitation Motion Data

SMOTE-GAN Hybrid approach (literature-based):
  Phase 1: Within-class SMOTE — expand sparse real data (77 files) by
           k-NN interpolation within each FMA class
  Phase 2: Cross-class interpolation — generate intermediate FMA scores
           by blending between expanded stroke and healthy pools

Output goes to a SEPARATE directory (augmented_smote/) so DTW-augmented
data is fully preserved and we can switch back via --data-source flag.
"""
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.neighbors import NearestNeighbors
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

# --- Configuration ---
RAW_HEALTHY_DIR = get_path("data_healthy")
RAW_STROKE_DIR = get_path("data_stroke")
PROCESSED_DIR = get_path("data_cutoff_processed")
OUTPUT_DIR = get_path("data_cutoff_augmented_smote")
SCORES_FILE = get_path("scores_file")

HEALTHY_FMA = 66
TARGET_LEN = 100

ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
            'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
ALL_COLS = ARM_COLS + TRUNK_COLS
N_FEATURES = len(ALL_COLS) * TARGET_LEN  # 1500


# =============================================================================
# Data loading (reused from generate_augmented_fma.py)
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
    raw_path = os.path.join(raw_dir, fname)
    trunk = load_raw_trunk(raw_path)

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
# Phase 1: Within-class SMOTE
# =============================================================================

def smote_expand_class(samples, target_n, k=3):
    """
    Expand a class using SMOTE k-NN interpolation.

    Args:
        samples: list of 1D arrays (flattened trajectories, each 1500-dim)
        target_n: desired number of samples
        k: number of nearest neighbors

    Returns:
        list of 1D arrays (original + synthetic)
    """
    n = len(samples)
    if n >= target_n:
        # Subsample to exact target
        indices = np.random.choice(n, target_n, replace=False)
        return [samples[i] for i in indices]

    samples_arr = np.array(samples)
    k_actual = min(k, n - 1)

    if k_actual < 1:
        # Only 1 sample — add small Gaussian noise
        expanded = list(samples)
        std = np.std(samples_arr) if samples_arr.std() > 0 else 1.0
        for _ in range(target_n - n):
            noise = np.random.normal(0, 0.01 * std, samples[0].shape)
            expanded.append(samples[0] + noise)
        return expanded

    nn = NearestNeighbors(n_neighbors=k_actual + 1).fit(samples_arr)

    expanded = list(samples)
    while len(expanded) < target_n:
        # Pick a random original sample
        idx = np.random.randint(n)
        _, indices = nn.kneighbors([samples_arr[idx]])
        # Pick a random neighbor (skip self at index 0)
        nn_idx = indices[0][np.random.randint(1, k_actual + 1)]
        # Interpolate at random point along the segment
        lam = np.random.random()
        synthetic = samples_arr[idx] + lam * (samples_arr[nn_idx] - samples_arr[idx])
        expanded.append(synthetic)

    return expanded


# =============================================================================
# Phase 2: Cross-class interpolation
# =============================================================================

def generate_cross_class(stroke_pool, healthy_pool, target_fma, n_samples):
    """
    Generate n_samples at target_fma by interpolating between stroke and healthy.

    Args:
        stroke_pool: list of (flattened_array, fma_score) tuples
        healthy_pool: list of flattened arrays (all FMA 66)
        target_fma: desired FMA score
        n_samples: number of samples to generate

    Returns:
        list of 1D arrays (flattened trajectories)
    """
    synthetics = []
    for _ in range(n_samples):
        s_idx = np.random.randint(len(stroke_pool))
        h_idx = np.random.randint(len(healthy_pool))
        s_arr, s_fma = stroke_pool[s_idx]
        h_arr = healthy_pool[h_idx]

        alpha = (target_fma - s_fma) / (HEALTHY_FMA - s_fma)
        alpha = np.clip(alpha, 0.0, 1.0)

        # Small noise on alpha for diversity (±2%)
        alpha += np.random.normal(0, 0.02)
        alpha = np.clip(alpha, 0.0, 1.0)

        synthetic = (1.0 - alpha) * s_arr + alpha * h_arr
        synthetics.append(synthetic)

    return synthetics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SMOTE-based motion augmentation")
    parser.add_argument("--samples-per-class", type=int, default=50,
                        help="Within-class SMOTE target per FMA class (default: 50)")
    parser.add_argument("--samples-per-fma", type=int, default=500,
                        help="Cross-class samples per intermediate FMA (default: 500)")
    parser.add_argument("--k", type=int, default=3,
                        help="SMOTE k-nearest neighbors (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("SMOTE-Based Augmentation")
    print(f"  Within-class target: {args.samples_per_class}/class")
    print(f"  Cross-class target:  {args.samples_per_fma}/FMA")
    print(f"  k-neighbors:         {args.k}")
    print(f"  Output:              {OUTPUT_DIR}")
    print("=" * 60)

    # --- Load FMA scores ---
    scores_df = pd.read_csv(SCORES_FILE)
    score_map = {}
    for _, row in scores_df.iterrows():
        name = str(row.iloc[0]).replace('.mot', '').replace('.csv', '').strip()
        score_map[name] = int(row.iloc[1])
    print(f"\nLoaded {len(score_map)} FMA scores")

    # --- Load all 77 real trajectories ---
    all_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv")))

    stroke_by_fma = {}   # {fma_score: [flattened_array, ...]}
    healthy_flat = []     # [flattened_array, ...]

    for fpath in all_files:
        fname = os.path.basename(fpath).replace('.csv', '')
        is_stroke = fname.startswith('S')
        raw_dir = RAW_STROKE_DIR if is_stroke else RAW_HEALTHY_DIR

        df = load_motion_with_trunk(fpath, raw_dir)
        df_delta = to_delta_format(df)
        df_resampled = resample_df(df_delta, TARGET_LEN)
        flat = df_resampled[ALL_COLS].values.flatten()  # (1500,)

        if is_stroke:
            fma = score_map.get(fname, score_map.get(fname.replace('S', '')))
            if fma is None:
                print(f"  Skipping {fname}: no FMA score")
                continue
            stroke_by_fma.setdefault(fma, []).append(flat)
        else:
            healthy_flat.append(flat)

    n_stroke = sum(len(v) for v in stroke_by_fma.values())
    print(f"\nReal data: {n_stroke} stroke + {len(healthy_flat)} healthy = {n_stroke + len(healthy_flat)} total")
    for fma in sorted(stroke_by_fma.keys()):
        print(f"  FMA {fma}: {len(stroke_by_fma[fma])} files")
    print(f"  FMA 66:  {len(healthy_flat)} files")

    # =========================================================================
    # Phase 1: Within-class SMOTE expansion
    # =========================================================================
    print(f"\n--- Phase 1: Within-class SMOTE (target {args.samples_per_class}/class) ---")

    expanded_stroke = {}  # {fma: [flattened arrays]}
    for fma in sorted(stroke_by_fma.keys()):
        original = stroke_by_fma[fma]
        expanded = smote_expand_class(original, args.samples_per_class, k=args.k)
        expanded_stroke[fma] = expanded
        print(f"  FMA {fma}: {len(original)} → {len(expanded)}")

    expanded_healthy = smote_expand_class(
        healthy_flat, max(args.samples_per_class, len(healthy_flat)), k=args.k)
    print(f"  FMA 66:  {len(healthy_flat)} → {len(expanded_healthy)}")

    # =========================================================================
    # Phase 2: Cross-class interpolation
    # =========================================================================
    print(f"\n--- Phase 2: Cross-class interpolation ({args.samples_per_fma}/FMA) ---")

    # Build stroke pool for cross-class: [(flat_arr, fma_score), ...]
    stroke_pool = []
    for fma, samples in expanded_stroke.items():
        for s in samples:
            stroke_pool.append((s, fma))

    min_fma = min(stroke_by_fma.keys())
    # FMA range for cross-class: from min_stroke_fma to 65
    cross_fma_range = list(range(min_fma, HEALTHY_FMA))

    # Clear output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    if existing:
        print(f"\n  Clearing {len(existing)} existing files in {OUTPUT_DIR}")
        for f in existing:
            os.remove(f)

    file_idx = 0
    fma_counts = {}

    # Save expanded stroke base samples
    for fma, samples in expanded_stroke.items():
        for s in samples:
            arr = s.reshape(TARGET_LEN, len(ALL_COLS))
            df = pd.DataFrame(arr, columns=ALL_COLS)
            df.to_csv(os.path.join(OUTPUT_DIR, f"smote_{file_idx:05d}_FMA{fma}.csv"), index=False)
            file_idx += 1
            fma_counts[fma] = fma_counts.get(fma, 0) + 1

    # Save expanded healthy base samples
    for s in expanded_healthy:
        arr = s.reshape(TARGET_LEN, len(ALL_COLS))
        df = pd.DataFrame(arr, columns=ALL_COLS)
        df.to_csv(os.path.join(OUTPUT_DIR, f"smote_{file_idx:05d}_FMA{HEALTHY_FMA}.csv"), index=False)
        file_idx += 1
        fma_counts[HEALTHY_FMA] = fma_counts.get(HEALTHY_FMA, 0) + 1

    # Generate cross-class samples for all intermediate FMA values
    for target_fma in cross_fma_range:
        # Skip FMA scores that already have SMOTE-expanded base samples
        # (they'll get cross-class samples too, for diversity)
        synthetics = generate_cross_class(
            stroke_pool, expanded_healthy, target_fma, args.samples_per_fma)

        for s in synthetics:
            arr = s.reshape(TARGET_LEN, len(ALL_COLS))
            df = pd.DataFrame(arr, columns=ALL_COLS)
            df.to_csv(os.path.join(OUTPUT_DIR, f"smote_{file_idx:05d}_FMA{target_fma}.csv"), index=False)
            file_idx += 1
            fma_counts[target_fma] = fma_counts.get(target_fma, 0) + 1

        if target_fma % 10 == 0:
            print(f"  FMA {target_fma}: {fma_counts[target_fma]} total samples")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"DONE! Generated {file_idx} augmented files")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nFMA Distribution:")
    for fma in sorted(fma_counts.keys()):
        print(f"  FMA {fma:3d}: {fma_counts[fma]:5d} samples")
    print(f"\nTotal unique FMA scores: {len(fma_counts)}")
    print(f"Total files: {sum(fma_counts.values())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
