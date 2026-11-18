"""
Trunk damping weight sweep — empirical test.

For each TR_DAMPING in WEIGHTS:
  1. Runs the full IK pipeline (--skip-id) to output/weight_test_W{W}/
  2. Computes: mean IK error, trunk DOF range, trunk-CS Pearson r, shoulder elevation
  3. Flags SC compensation if shoulder elevation drifts >15mm from W=50 baseline

Run:
  cd /home/abdul/Desktop/myosuite/custom_workspace/part2
  python scripts/test_trunk_weight.py
"""
import os, sys, json, subprocess, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelmin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root, get

ROOT      = get_project_root()
WORKSPACE = os.path.dirname(ROOT)
PYTHON    = get("conda_python", default=sys.executable)

STROKE_CSV_DIR  = os.path.join(WORKSPACE, get_path("data_stroke"))
HEALTHY_CSV_DIR = os.path.join(WORKSPACE, get_path("data_healthy"))

PHASE_IDX_STROKE  = json.load(open(os.path.join(ROOT, "output", "phase_indices", "stroke_phase_indices.json")))
PHASE_IDX_HEALTHY = json.load(open(os.path.join(ROOT, "output", "phase_indices", "healthy_phase_indices.json")))

CS_MARKERS = ["CS_1", "CS_2", "CS_3", "CS_4"]
UP_AXIS    = np.array([0.0, 1.0, 0.0])

WEIGHTS = [0.5, 1.0, 5.0, 10.0, 50.0]

# Shoulder elevation baseline at W=50 (from prior run)
BASELINE_SHO_STROKE  = 67.0  # mm
BASELINE_SHO_HEALTHY = 43.0  # mm
SHO_FLAG_DELTA = 15.0        # mm — flag if deviation exceeds this


# ── Data loaders ──────────────────────────────────────────────────────────────

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
    for i, l in enumerate(lines):
        if "endheader" in l:
            cols = lines[i+1].strip().split("\t")
            break
    return cols, np.loadtxt(path, skiprows=i+2, delimiter="\t")


def get_phase(fname, group):
    idx = PHASE_IDX_STROKE if group == "Stroke" else PHASE_IDX_HEALTHY
    phases = idx.get(fname)
    if phases is None: return None, None
    return int(phases[0]), int(phases[1])


# ── CS plane tilt ─────────────────────────────────────────────────────────────

def cs_forward_tilt(df, peak_f):
    avail = [m for m in CS_MARKERS if m in df.columns.get_level_values(0)]
    if len(avail) < 3: return None
    x = df[avail].xs("X", level=1, axis=1).values
    y = df[avail].xs("Y", level=1, axis=1).values
    z = df[avail].xs("Z", level=1, axis=1).values
    pts = np.stack([x, y, z], axis=2)

    # Forward axis from wrist at phase peak
    avail_w = [m for m in ["WRA", "WRB"] if m in df.columns.get_level_values(0)]
    if not avail_w: return None
    wx = df[avail_w].xs("X", level=1, axis=1).mean(axis=1).values
    wz = df[avail_w].xs("Z", level=1, axis=1).mean(axis=1).values
    dx = wx[peak_f] - wx[0]; dz = wz[peak_f] - wz[0]
    norm = np.sqrt(dx**2 + dz**2)
    if norm < 1e-6: return None
    fwd = np.array([dx, 0.0, dz]) / norm
    up  = UP_AXIS - np.dot(UP_AXIS, fwd) * fwd; up /= np.linalg.norm(up)

    angles = np.full(len(df), np.nan)
    for i, row in enumerate(pts):
        if np.any(np.isnan(row)): continue
        c = row.mean(0)
        _, _, Vt = np.linalg.svd(row - c, full_matrices=False)
        n = Vt[-1]; n /= np.linalg.norm(n)
        if n[1] < 0: n = -n
        raw = np.degrees(np.arctan2(np.dot(n, fwd), np.dot(n, up)))
        angles[i] = ((raw + 90) % 180) - 90
    return angles


# ── Shoulder elevation from raw CSV ──────────────────────────────────────────

def shoulder_elev(df, sf, pf):
    if "SA_3" not in df.columns.get_level_values(0): return np.nan
    y = df["SA_3"]["Y"].values
    return float(y[pf] - y[sf])


# ── Evaluate one weight run ───────────────────────────────────────────────────

def evaluate_run(out_dir):
    rep_path = os.path.join(out_dir, "batch_report.csv")
    if not os.path.exists(rep_path):
        return None
    rep = pd.read_csv(rep_path)

    ik_errors, trunk_ranges = [], []
    cs_comps, ik_comps = [], []
    sho_stroke, sho_healthy = [], []

    for group, csv_dir, grp_label in [
        ("stroke",  STROKE_CSV_DIR,  "Stroke"),
        ("healthy", HEALTHY_CSV_DIR, "Healthy"),
    ]:
        mot_dir = os.path.join(out_dir, group, "mot")
        for fname in sorted(os.listdir(csv_dir)):
            if not fname.endswith(".csv"): continue
            stem = fname.replace(".csv", "")
            sf, pf = get_phase(fname, grp_label)
            if sf is None: continue

            mot_path = os.path.join(mot_dir, stem + ".mot")
            if not os.path.exists(mot_path): continue

            # IK error from report
            row = rep[rep["file"] == stem]
            if len(row) and row.iloc[0]["status"] == "OK":
                err = pd.to_numeric(row.iloc[0]["ik_error_mm"], errors="coerce")
                if not np.isnan(err):
                    ik_errors.append(err)

            try:
                df = load_csv(os.path.join(csv_dir, fname))
                cols, arr = load_mot(mot_path)
                n = min(len(df), len(arr))
                sf_c = min(sf, n-1); pf_c = min(pf, n-1)

                # Trunk DOF range
                tidx = cols.index("trunk_forward_lean")
                trunk = np.degrees(arr[:, tidx])
                trunk_ranges.append(trunk.max() - trunk.min())

                # CS comp vs IK comp
                tilt = cs_forward_tilt(df, pf_c)
                if tilt is not None:
                    cs_comp = float(tilt[pf_c] - tilt[sf_c])
                    ik_comp = float(np.degrees(arr[pf_c, tidx] - arr[sf_c, tidx]))
                    if not (np.isnan(cs_comp) or np.isnan(ik_comp)):
                        cs_comps.append(cs_comp)
                        ik_comps.append(ik_comp)

                # Shoulder elevation
                se = shoulder_elev(df, sf_c, pf_c)
                if not np.isnan(se):
                    if grp_label == "Stroke":
                        sho_stroke.append(se)
                    else:
                        sho_healthy.append(se)

            except Exception:
                continue

    if not ik_comps:
        return None

    r, p = stats.pearsonr(cs_comps, ik_comps) if len(cs_comps) > 2 else (np.nan, np.nan)
    return dict(
        ik_err      = np.mean(ik_errors) if ik_errors else np.nan,
        trunk_range = np.mean(trunk_ranges) if trunk_ranges else np.nan,
        r_vs_cs     = r,
        p_val       = p,
        n_trials    = len(cs_comps),
        sho_stroke  = np.mean(sho_stroke)  if sho_stroke  else np.nan,
        sho_healthy = np.mean(sho_healthy) if sho_healthy else np.nan,
    )


# ── Run pipeline for each weight ──────────────────────────────────────────────

PIPELINE = os.path.join(ROOT, "scripts", "run_originals_pipeline.py")

summary = []
for W in WEIGHTS:
    tag     = f"{W}".replace(".", "p")
    out_dir = os.path.join(ROOT, "output", f"weight_test_W{tag}")

    rep_path = os.path.join(out_dir, "batch_report.csv")
    if os.path.exists(rep_path):
        print(f"\nW={W}: output exists, skipping pipeline run")
    else:
        print(f"\n{'='*60}")
        print(f"Running IK pipeline: TR_DAMPING={W}")
        print(f"Output: {out_dir}")
        print(f"{'='*60}")
        t0 = time.time()
        proc = subprocess.run(
            [PYTHON, PIPELINE,
             "--trunk-weight", str(W),
             "--output-dir", out_dir,
             "--skip-id"],
            cwd=ROOT
        )
        elapsed = time.time() - t0
        print(f"Pipeline done in {elapsed/60:.1f} min (exit={proc.returncode})")

    metrics = evaluate_run(out_dir)
    if metrics is None:
        print(f"W={W}: no results found"); continue

    sho_s_flag = abs(metrics["sho_stroke"]  - BASELINE_SHO_STROKE)  > SHO_FLAG_DELTA
    sho_h_flag = abs(metrics["sho_healthy"] - BASELINE_SHO_HEALTHY) > SHO_FLAG_DELTA
    sc_warn    = " ⚠ SC comp?" if (sho_s_flag or sho_h_flag) else ""

    summary.append(dict(W=W, **metrics, sc_warn=sc_warn))

# ── Print results table ───────────────────────────────────────────────────────

print(f"\n{'='*90}")
print("Trunk Damping Weight Sweep Results")
print(f"{'='*90}")
print(f"{'W':>6}  {'IK err(mm)':>10}  {'Trunk range(°)':>14}  "
      f"{'r_vs_CS':>8}  {'p':>7}  {'ShoE_S(mm)':>10}  {'ShoE_H(mm)':>10}  {'':>10}")
print("-"*90)
for s in summary:
    flag = "◄ baseline" if abs(s["W"] - 50.0) < 0.1 else ""
    print(f"{s['W']:>6.1f}  {s['ik_err']:>10.2f}  {s['trunk_range']:>14.2f}  "
          f"{s['r_vs_cs']:>8.3f}  {s['p_val']:>7.4f}  "
          f"{s['sho_stroke']:>10.1f}  {s['sho_healthy']:>10.1f}  "
          f"{s['sc_warn']}{flag}")
print(f"{'='*90}")
print("r_vs_CS = Pearson r between trunk_forward_lean DOF and CS plane tilt (start→peak)")
print("SC comp? = shoulder elevation deviated >15mm from W=50 baseline")
