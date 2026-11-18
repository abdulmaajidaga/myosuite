"""
Trunk compensation metric per subject — forward AND lateral lean.

For each trial:
  1. Find peak wrist excursion frame (furthest reach from start).
  2. Search backwards from peak to find the local minimum in wrist excursion
     (= the moment the reach began / wrist closest before the peak).
  3. forward_comp = forward_tilt[peak] - forward_tilt[start]
     lateral_comp = lateral_tilt[peak] - lateral_tilt[start]

Forward tilt  = CS plane normal projected onto sagittal plane (fwd-UP).
Lateral tilt  = CS plane normal projected onto frontal plane (lat-UP).
Forward axis  = derived from wrist trajectory.
Lateral axis  = cross(UP, forward).
"""
import os, sys
import numpy as np
import pandas as pd
import json
from scipy.signal import argrelmin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root

ROOT            = get_project_root()
WORKSPACE       = os.path.dirname(ROOT)
STROKE_CSV_DIR  = os.path.join(WORKSPACE, get_path("data_stroke"))
HEALTHY_CSV_DIR = os.path.join(WORKSPACE, get_path("data_healthy"))
OUT_CSV         = os.path.join(ROOT, "output", "trunk_lean_validation", "trunk_comp_table.csv")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

PHASE_IDX_STROKE  = json.load(open(os.path.join(ROOT, "output", "phase_indices", "stroke_phase_indices.json")))
PHASE_IDX_HEALTHY = json.load(open(os.path.join(ROOT, "output", "phase_indices", "healthy_phase_indices.json")))

def get_phase(fname, group):
    idx = PHASE_IDX_STROKE if group == "Stroke" else PHASE_IDX_HEALTHY
    phases = idx.get(fname)
    if phases is None: return None, None
    return int(phases[0]), int(phases[1])

CS_MARKERS = ["CS_1", "CS_2", "CS_3", "CS_4"]
UP_AXIS    = np.array([0.0, 1.0, 0.0])


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def wrist_excursion(df, peak_override=None):
    avail = [m for m in ["WRA", "WRB"] if m in df.columns.get_level_values(0)]
    if not avail:
        return None, None, None
    wx = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
    wz = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
    dx, dz = wx - wx[0], wz - wz[0]
    exc = np.sqrt(dx**2 + dz**2)
    peak = peak_override if peak_override is not None else int(np.nanargmax(exc))
    peak = min(peak, len(exc) - 1)
    fwd  = np.array([dx[peak], 0.0, dz[peak]])
    norm = np.linalg.norm(fwd)
    fwd  = fwd / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    lat  = np.cross(UP_AXIS, fwd); lat /= np.linalg.norm(lat)
    return exc, fwd, lat


def cs_normals(df):
    avail = [m for m in CS_MARKERS if m in df.columns.get_level_values(0)]
    if len(avail) < 3:
        return None
    x = df[avail].xs("X", level=1, axis=1).values
    y = df[avail].xs("Y", level=1, axis=1).values
    z = df[avail].xs("Z", level=1, axis=1).values
    pts = np.stack([x, y, z], axis=2)

    normals, prev_n = np.full((len(df), 3), np.nan), None
    for i, row in enumerate(pts):
        if np.any(np.isnan(row)):
            prev_n = None; continue
        c = row.mean(0)
        _, _, Vt = np.linalg.svd(row - c, full_matrices=False)
        n = Vt[-1]; n /= np.linalg.norm(n)
        if n[1] < 0:   # enforce upward-pointing normal
            n = -n
        prev_n = n
        normals[i] = n
    return normals


def project_tilt(normals, axis):
    """Signed tilt angle of CS plane normal projected onto axis-UP plane (degrees)."""
    up  = UP_AXIS - np.dot(UP_AXIS, axis) * axis
    up /= np.linalg.norm(up)
    proj_axis = normals @ axis
    proj_up   = normals @ up
    angles = np.degrees(np.arctan2(proj_axis, proj_up))
    # Wrap to [-90, 90]: normal is upward-constrained so true lean is small
    angles = ((angles + 90) % 180) - 90
    return angles


def absolute_tilt(normals):
    return np.degrees(np.arccos(np.clip(np.abs(normals @ UP_AXIS), -1, 1)))




# ── Process all trials ────────────────────────────────────────────────────────

rows = []
for csv_dir, group in [(STROKE_CSV_DIR, "Stroke"), (HEALTHY_CSV_DIR, "Healthy")]:
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue
        stem = fname.replace(".csv", "")
        try:
            start_f, peak_f = get_phase(fname, group)
            if start_f is None:
                print(f"  skip {stem}: no phase index"); continue

            df              = load_csv(os.path.join(csv_dir, fname))
            exc, fwd, lat   = wrist_excursion(df, peak_override=peak_f)
            normals         = cs_normals(df)
            if exc is None or normals is None:
                print(f"  skip {stem}: missing markers"); continue

            start_f = min(start_f, len(exc) - 1)
            peak_f  = min(peak_f,  len(exc) - 1)

            abs_tilt  = absolute_tilt(normals)
            fwd_tilt  = project_tilt(normals, fwd)
            lat_tilt  = project_tilt(normals, lat)

            def comp(arr):
                s, p = arr[start_f], arr[peak_f]
                return (round(float(s), 2), round(float(p), 2), round(float(p-s), 2))

            abs_s, abs_p, abs_c   = comp(abs_tilt)
            fwd_s, fwd_p, fwd_c   = comp(fwd_tilt)
            lat_s, lat_p, lat_c   = comp(lat_tilt)

            if any(np.isnan(v) for v in [abs_s, abs_p, fwd_s, fwd_p, lat_s, lat_p]):
                print(f"  skip {stem}: NaN at key frames"); continue

            total_comp = round(float(np.sqrt(fwd_c**2 + lat_c**2)), 2)

            rows.append(dict(
                subject          = stem,
                group            = group,
                wrist_exc_mm     = round(float(exc[peak_f]), 1),
                abs_tilt_start   = abs_s,  abs_tilt_peak = abs_p,  abs_comp   = abs_c,
                fwd_tilt_start   = fwd_s,  fwd_tilt_peak = fwd_p,  fwd_comp   = fwd_c,
                lat_tilt_start   = lat_s,  lat_tilt_peak = lat_p,  lat_comp   = lat_c,
                total_comp       = total_comp,
            ))
        except Exception as e:
            print(f"  skip {stem}: {e}")


# ── Print table ───────────────────────────────────────────────────────────────

results = pd.DataFrame(rows).sort_values(["group", "subject"]).reset_index(drop=True)

hdr = (f"{'Subject':<16}  {'Group':<8}  "
       f"{'Abs@start':>9}  {'Abs@peak':>8}  {'AbsComp':>8}  "
       f"{'Fwd@start':>9}  {'Fwd@peak':>8}  {'FwdComp':>8}  "
       f"{'Lat@start':>9}  {'Lat@peak':>8}  {'LatComp':>8}  "
       f"{'WristExc':>8}")
print("\n" + hdr)
print("─" * len(hdr))

for _, r in results.iterrows():
    print(f"{r.subject:<16}  {r.group:<8}  "
          f"{r.abs_tilt_start:>9.1f}  {r.abs_tilt_peak:>8.1f}  {r.abs_comp:>+8.1f}  "
          f"{r.fwd_tilt_start:>9.1f}  {r.fwd_tilt_peak:>8.1f}  {r.fwd_comp:>+8.1f}  "
          f"{r.lat_tilt_start:>9.1f}  {r.lat_tilt_peak:>8.1f}  {r.lat_comp:>+8.1f}  "
          f"{r.wrist_exc_mm:>8.1f}")

print()
for g in ["Stroke", "Healthy"]:
    s = results[results.group == g]
    print(f"{g} (n={len(s)}):  "
          f"fwd_comp={s.fwd_comp.mean():+.2f}°±{s.fwd_comp.std():.2f}°  "
          f"lat_comp={s.lat_comp.mean():+.2f}°±{s.lat_comp.std():.2f}°  "
          f"abs_comp={s.abs_comp.mean():+.2f}°±{s.abs_comp.std():.2f}°")

results.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
