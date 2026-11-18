"""
Post-process MOT files: replace trunk_forward_lean with CS-marker-derived trunk lean.

The IK arm markers (V_Shoulder, V_Elbow, V_Wrist, V_Vector) have zero gradient on
the trunk DOF — the solver never moves it regardless of damping weight. This script
injects the real trunk lean signal from the chest cluster (CS_1–CS_4) directly into
the trunk_forward_lean column of each MOT file.

Strategy:
  cs_lean[i] = arctan2(n·up_perp, n·fwd)   ← 0 when upright, + when leaning fwd
  delta[i]   = cs_lean[i] - cs_lean[0]     (change from frame 0, radians)
  trunk_lean[i] = ik_baseline[0] + delta[i]  (anchored to IK frame 0)

Sign convention: arctan2(pu, pf) gives 0 when plate is vertical (upright subject),
positive when trunk leans forward.  This matches the model's trunk_forward_lean DOF
(positive = forward lean).  This is the correct formula; the alternative arctan2(pf, pu)
gives 90° at neutral and has a discontinuity, producing inverted / noisy output.

Usage:
  python scripts/inject_cs_trunk.py [out_base]
  out_base defaults to output/originals (the standard pipeline output dir).
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root

ROOT      = get_project_root()
WORKSPACE = os.path.dirname(ROOT)

OUT_BASE   = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "output", "originals")

STROKE_CSV_DIR  = os.path.join(WORKSPACE, get_path("data_stroke"))
HEALTHY_CSV_DIR = os.path.join(WORKSPACE, get_path("data_healthy"))

PHASE_IDX_STROKE  = json.load(open(os.path.join(ROOT, "output", "phase_indices", "stroke_phase_indices.json")))
PHASE_IDX_HEALTHY = json.load(open(os.path.join(ROOT, "output", "phase_indices", "healthy_phase_indices.json")))

CS_MARKERS  = ["CS_1", "CS_2", "CS_3", "CS_4"]
UP_AXIS     = np.array([0.0, 1.0, 0.0])
TRUNK_LIMIT = 0.70   # model joint range ±0.70 rad


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_csv(path):
    raw = pd.read_csv(path, header=[0, 1])
    current, tuples = None, []
    for col in raw.columns:
        m, ax = col
        if not str(m).startswith("Unnamed"):
            current = m.strip()
        tuples.append((current, ax))
    raw.columns = pd.MultiIndex.from_tuples(tuples)
    return raw.iloc[1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)


def load_mot(path):
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "endheader" in line:
            header_end = i
            cols = lines[i + 1].strip().split("\t")
            break
    arr = np.loadtxt(path, skiprows=header_end + 2, delimiter="\t")
    return cols, arr, header_end


def write_mot(path, cols, arr, header_end_line_idx, original_lines):
    with open(path, "w") as f:
        # Rewrite header exactly as it was (nRows may change if lengths differ)
        for line in original_lines[:header_end_line_idx + 1]:
            if line.startswith("nRows="):
                f.write(f"nRows={len(arr)}\n")
            else:
                f.write(line)
        f.write("\t".join(cols) + "\n")
        for row in arr:
            f.write("\t".join(f"{x:.6f}" for x in row) + "\n")


# ── CS forward tilt ───────────────────────────────────────────────────────────

def wrist_fwd_axis(df, peak_frame=None):
    avail = [m for m in ["WRA", "WRB"] if m in df.columns.get_level_values(0)]
    if not avail:
        return np.array([1.0, 0.0, 0.0])
    wx = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
    wz = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
    if peak_frame is None:
        exc = np.sqrt((wx - wx[0])**2 + (wz - wz[0])**2)
        peak_frame = int(np.nanargmax(exc))
    pf = min(peak_frame, len(wx) - 1)
    dx, dz = wx[pf] - wx[0], wz[pf] - wz[0]
    fwd = np.array([dx, 0.0, dz])
    norm = np.linalg.norm(fwd)
    return fwd / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])


def cs_lean_rad(df, fwd_axis):
    """
    Per-frame trunk forward lean in radians from CS plate geometry.

    arctan2(n·up_perp, n·fwd):
      0   = upright  (plate normal pointing horizontally forward)
      +   = trunk leaning forward  (normal tilts upward)
      −   = trunk leaning backward (normal tilts downward)

    Ensures normal points in the general forward direction (dot(n, fwd) > 0)
    so the backward-lean case returns negative angles correctly.
    """
    avail = [m for m in CS_MARKERS if m in df.columns.get_level_values(0)]
    if len(avail) < 3:
        return None
    x = df[avail].xs("X", level=1, axis=1).values
    y = df[avail].xs("Y", level=1, axis=1).values
    z = df[avail].xs("Z", level=1, axis=1).values
    pts = np.stack([x, y, z], axis=2)

    fwd = fwd_axis / np.linalg.norm(fwd_axis)
    up  = UP_AXIS - np.dot(UP_AXIS, fwd) * fwd
    up /= np.linalg.norm(up)

    angles = np.full(len(df), np.nan)
    for i, row in enumerate(pts):
        if np.any(np.isnan(row)):
            continue
        c = row.mean(0)
        _, _, Vt = np.linalg.svd(row - c, full_matrices=False)
        n = Vt[-1]
        n /= np.linalg.norm(n)
        if np.dot(n, fwd) < 0:   # ensure normal faces forward, not backward
            n = -n
        pf = np.dot(n, fwd)
        pu = np.dot(n, up)
        angles[i] = np.arctan2(pu, pf)   # radians: 0=upright, +=forward lean
    return angles


# ── Inject per trial ──────────────────────────────────────────────────────────

def inject_trial(csv_path, mot_path, phase_idx, group_label):
    fname = os.path.basename(csv_path)
    stem  = fname.replace(".csv", "")

    phases = phase_idx.get(fname)
    peak_f = int(phases[1]) if phases is not None else None

    try:
        df = load_csv(csv_path)
    except Exception as e:
        print(f"  skip {stem}: CSV load failed — {e}")
        return False

    fwd  = wrist_fwd_axis(df, peak_f)
    tilt = cs_lean_rad(df, fwd)
    if tilt is None:
        print(f"  skip {stem}: <3 CS markers available")
        return False

    try:
        with open(mot_path) as f:
            orig_lines = f.readlines()
        cols, arr, header_end = load_mot(mot_path)
    except Exception as e:
        print(f"  skip {stem}: MOT load failed — {e}")
        return False

    if "trunk_forward_lean" not in cols:
        print(f"  skip {stem}: trunk_forward_lean not in MOT")
        return False

    trunk_col = cols.index("trunk_forward_lean")
    n = min(len(df), len(arr))
    if n < 2:
        print(f"  skip {stem}: too few frames ({n})")
        return False

    cs_seg  = tilt[:n]
    baseline_rad = arr[0, trunk_col]           # IK frame-0 trunk pose

    # NaN-fill any gaps via linear interpolation before injecting
    cs_series = pd.Series(cs_seg).interpolate(method="linear", limit_direction="both")
    cs_seg = cs_series.values

    delta_rad = cs_seg - cs_seg[0]          # already radians
    trunk_new = np.clip(baseline_rad + delta_rad, -TRUNK_LIMIT, TRUNK_LIMIT)

    arr_out = arr[:n].copy()
    arr_out[:, trunk_col] = trunk_new

    try:
        write_mot(mot_path, cols, arr_out, header_end, orig_lines)
    except Exception as e:
        print(f"  skip {stem}: MOT write failed — {e}")
        return False

    delta_deg = np.degrees(trunk_new[-1] - trunk_new[0])
    print(f"  [OK] {stem:<18} trunk range={np.degrees(trunk_new.max()-trunk_new.min()):.1f}°  "
          f"peak_delta={delta_deg:+.1f}°")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    groups = [
        ("stroke",  STROKE_CSV_DIR,  os.path.join(OUT_BASE, "stroke",  "mot"), PHASE_IDX_STROKE,  "Stroke"),
        ("healthy", HEALTHY_CSV_DIR, os.path.join(OUT_BASE, "healthy", "mot"), PHASE_IDX_HEALTHY, "Healthy"),
    ]

    total_ok = total_skip = 0
    for group, csv_dir, mot_dir, phase_idx, label in groups:
        if not os.path.isdir(mot_dir):
            print(f"\nSkipping {label}: MOT dir not found ({mot_dir})")
            continue
        print(f"\n=== Injecting CS trunk lean: {label} ===")
        for fname in sorted(os.listdir(csv_dir)):
            if not fname.endswith(".csv"):
                continue
            stem     = fname.replace(".csv", "")
            mot_path = os.path.join(mot_dir, stem + ".mot")
            if not os.path.exists(mot_path):
                print(f"  skip {stem}: no MOT file")
                total_skip += 1
                continue
            ok = inject_trial(os.path.join(csv_dir, fname), mot_path, phase_idx, label)
            if ok:
                total_ok += 1
            else:
                total_skip += 1

    print(f"\nDone: {total_ok} injected, {total_skip} skipped")
    print("Re-run correlate_ik_vs_cs.py to verify the updated trunk_forward_lean.")


if __name__ == "__main__":
    main()
