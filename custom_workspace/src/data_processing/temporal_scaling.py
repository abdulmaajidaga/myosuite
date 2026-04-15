"""
Temporal scaling for CVAE-generated motions.

CVAE outputs are always 100 frames (resampled from real motions that were
400-1500+ frames at 200Hz). This module learns the FMA -> duration mapping
from real cutoff data and provides cubic spline interpolation to stretch
100-frame output to a realistic frame count.
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path


# =============================================================================
# Build FMA -> Duration mapping from real data
# =============================================================================

def _build_duration_mapping():
    """
    Scan cutoff/original/ files to build FMA score -> duration statistics.

    Returns dict: {fma_score: {'median': float, 'std': float, 'durations': list}}
    """
    original_dir = get_path("data_cutoff_original")
    scores_path = get_path("scores_file")
    data_rate = 200.0

    # Load scores.csv: maps filename -> FMA score
    scores_df = pd.read_csv(scores_path)
    # scores.csv has .mot extensions; strip to get base name
    score_map = {}
    for _, row in scores_df.iterrows():
        base = os.path.splitext(row["filename"])[0]
        score_map[base] = int(row["fma_score"])

    # Scan all original CSVs
    fma_durations = {}  # {fma_score: [duration_seconds, ...]}

    for fname in os.listdir(original_dir):
        if not fname.endswith(".csv"):
            continue
        base = os.path.splitext(fname)[0]
        fpath = os.path.join(original_dir, fname)

        # Count frames (rows minus header)
        with open(fpath, "r") as f:
            num_frames = sum(1 for _ in f) - 1

        if num_frames <= 0:
            continue

        duration = num_frames / data_rate

        # Determine FMA score
        if base in score_map:
            fma = score_map[base]
        elif not base.startswith("S"):
            # Healthy subject (no S prefix) -> FMA 66
            fma = 66
        else:
            # Stroke file not in scores.csv - skip
            continue

        fma_durations.setdefault(fma, []).append(duration)

    # Compute statistics per FMA level
    mapping = {}
    for fma, durations in fma_durations.items():
        mapping[fma] = {
            "median": float(np.median(durations)),
            "std": float(np.std(durations)),
            "durations": durations,
        }

    return mapping


# Build mapping at module load time
_DURATION_MAP = _build_duration_mapping()

# Pre-compute stroke and healthy medians for interpolation
_STROKE_SCORES = sorted([s for s in _DURATION_MAP if s < 66])
_STROKE_MEDIAN = float(np.median([
    d for s in _STROKE_SCORES for d in _DURATION_MAP[s]["durations"]
])) if _STROKE_SCORES else 3.0

_HEALTHY_MEDIAN = _DURATION_MAP.get(66, {}).get("median", 3.5)

# Observed duration range (for clamping)
_ALL_DURATIONS = [d for v in _DURATION_MAP.values() for d in v["durations"]]
_MIN_DURATION = min(_ALL_DURATIONS) if _ALL_DURATIONS else 1.0
_MAX_DURATION = max(_ALL_DURATIONS) if _ALL_DURATIONS else 7.0


# =============================================================================
# Public API
# =============================================================================

def predict_duration(fma_score):
    """
    Predict realistic motion duration (seconds) for a given FMA score.

    Uses learned mapping from real cutoff data. For intermediate scores (21-65),
    linearly interpolates between stroke and healthy medians.

    Args:
        fma_score: FMA upper extremity score (0-66)

    Returns:
        Predicted duration in seconds
    """
    fma_score = max(0, min(66, fma_score))

    # Exact match in mapping
    if fma_score in _DURATION_MAP:
        return _DURATION_MAP[fma_score]["median"]

    # Interpolate between stroke median and healthy median
    # Stroke scores are ~16-20, healthy is 66
    max_stroke = max(_STROKE_SCORES) if _STROKE_SCORES else 20
    min_stroke = min(_STROKE_SCORES) if _STROKE_SCORES else 16

    if fma_score <= min_stroke:
        # Below observed stroke range -> use stroke median
        duration = _STROKE_MEDIAN
    elif fma_score >= 66:
        duration = _HEALTHY_MEDIAN
    else:
        # Linear interpolation between stroke and healthy
        t = (fma_score - max_stroke) / (66 - max_stroke)
        duration = _STROKE_MEDIAN + t * (_HEALTHY_MEDIAN - _STROKE_MEDIAN)

    # Clamp to observed range
    return float(np.clip(duration, _MIN_DURATION, _MAX_DURATION))


def apply_temporal_scaling(data, target_duration, data_rate=200.0):
    """
    Temporally scale a 100-frame motion to a realistic frame count using
    cubic spline interpolation.

    Args:
        data: numpy array of shape (100, N) - the CVAE output
        target_duration: target duration in seconds
        data_rate: output data rate in Hz (default 200)

    Returns:
        numpy array of shape (target_frames, N) with interpolated data
    """
    src_frames = data.shape[0]
    target_frames = max(int(target_duration * data_rate), src_frames)

    if target_frames == src_frames:
        return data.copy()

    # Normalized time axes [0, 1]
    t_src = np.linspace(0, 1, src_frames)
    t_dst = np.linspace(0, 1, target_frames)

    # Cubic interpolation per column
    interpolator = interp1d(t_src, data, axis=0, kind="cubic")
    scaled = interpolator(t_dst)

    return scaled


def get_duration_stats():
    """Return the full duration mapping for analysis/debugging."""
    return {
        "mapping": _DURATION_MAP,
        "stroke_median": _STROKE_MEDIAN,
        "healthy_median": _HEALTHY_MEDIAN,
        "min_duration": _MIN_DURATION,
        "max_duration": _MAX_DURATION,
    }


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    stats = get_duration_stats()
    print("=== Temporal Scaling: FMA -> Duration Mapping ===\n")
    print(f"Stroke median duration:  {stats['stroke_median']:.2f}s")
    print(f"Healthy median duration: {stats['healthy_median']:.2f}s")
    print(f"Duration range: [{stats['min_duration']:.2f}s, {stats['max_duration']:.2f}s]")

    print(f"\nPer-FMA statistics:")
    for fma in sorted(stats["mapping"].keys()):
        info = stats["mapping"][fma]
        n = len(info["durations"])
        print(f"  FMA {fma:3d}: median={info['median']:.2f}s, std={info['std']:.2f}s, n={n}")

    print(f"\nPredicted durations:")
    for score in [0, 10, 16, 18, 20, 30, 40, 50, 60, 66]:
        dur = predict_duration(score)
        frames = int(dur * 200)
        print(f"  FMA {score:3d}: {dur:.2f}s ({frames} frames)")
