"""
evaluate.py — Quick evaluation of generated CSVs (no IK/ID required).

Computes from CSV:
  segment_std_mm   — temporal std of upper-arm + forearm lengths (target: ~6mm)
  wrist_range_mm   — max wrist Y displacement per FMA (should ↑ with FMA)
  trunk_ratio      — trunk_disp / wrist_disp per FMA (should ↓ with FMA)
  boundary_ratio   — edge-to-mid velocity ratio (target ~1.0; >2 = LSTM artefact)
  peak_velocity_mm — mean peak wrist speed across FMA levels
  wrist_rho        — Spearman ρ(FMA, wrist_range_mm) — primary metric, target ≥ 0.9
  trunk_rho        — Spearman ρ(FMA, trunk_ratio)    — secondary metric

Saves per-experiment:
  eval_summary.json    — aggregated scalars (for results_log.csv)
  metrics_per_fma.csv  — full per-FMA breakdown (for figure regeneration)

Usage:
  python test/evaluate.py --experiment A3_smote
  python test/evaluate.py --summary           # table across all completed experiments
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy import stats

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TEST_DIR)
sys.path.insert(0, ROOT_DIR)

ARM_COLS   = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
              'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x','Trunk_y','Trunk_z']
COLS       = ARM_COLS + TRUNK_COLS

FMA_EVAL  = [18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 66]   # key FMA levels for CCI proxy


def _segment_std(df: pd.DataFrame) -> float:
    """Upper-arm length temporal std (mm). Target: ~6mm."""
    Sh = df[['Sh_x','Sh_y','Sh_z']].values
    El = df[['El_x','El_y','El_z']].values
    Wr = df[['Wr_x','Wr_y','Wr_z']].values
    ua = np.linalg.norm(El - Sh, axis=1)
    fa = np.linalg.norm(Wr - El, axis=1)
    return float(np.mean([ua.std(), fa.std()]))


def _boundary_ratio(df: pd.DataFrame) -> float:
    """Edge-to-mid velocity ratio. Target ~1.0; LSTM artefact produces >2.0.

    Computes mean wrist speed in the first+last 10% of frames vs middle 80%.
    Values near 1.0 mean speed is uniform — no edge spikes.
    """
    Wr = df[['Wr_x','Wr_y','Wr_z']].values
    vel = np.linalg.norm(np.diff(Wr, axis=0), axis=1)   # (T-1,)
    n = len(vel)
    edge_n = max(1, n // 10)
    edge_speed = np.mean(np.concatenate([vel[:edge_n], vel[-edge_n:]]))
    mid_speed  = np.mean(vel[edge_n:-edge_n]) if n > 2 * edge_n else np.mean(vel)
    return float(edge_speed / mid_speed) if mid_speed > 1e-6 else np.nan


def _peak_velocity(df: pd.DataFrame) -> float:
    """Peak wrist speed (mm/frame)."""
    Wr = df[['Wr_x','Wr_y','Wr_z']].values
    vel = np.linalg.norm(np.diff(Wr, axis=0), axis=1)
    return float(vel.max())


def _wrist_range(df: pd.DataFrame) -> float:
    """Wrist Y displacement (mm). Should ↑ with FMA."""
    wy = df['Wr_y'].values
    return float(wy.max() - wy.min())


def _trunk_ratio(df: pd.DataFrame) -> float:
    """Trunk displacement / wrist displacement. Should ↓ with FMA."""
    wr = df[['Wr_x','Wr_y','Wr_z']].values
    tr = df[['Trunk_x','Trunk_y','Trunk_z']].values
    wd = np.linalg.norm(wr - wr[0], axis=1).max()
    td = np.linalg.norm(tr - tr[0], axis=1).max()
    return float(td / wd) if wd > 1 else np.nan


def evaluate_csv_dir(csv_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(os.listdir(csv_dir)):
        if not f.endswith(".csv"):
            continue
        try:
            fma = int(f.replace("FMA_", "").replace(".csv", ""))
        except ValueError:
            continue
        df = pd.read_csv(os.path.join(csv_dir, f))
        rows.append({
            "fma":             fma,
            "segment_std_mm":  _segment_std(df),
            "wrist_range_mm":  _wrist_range(df),
            "trunk_ratio":     _trunk_ratio(df),
            "boundary_ratio":  _boundary_ratio(df),
            "peak_velocity_mm": _peak_velocity(df),
        })
    return pd.DataFrame(rows).sort_values("fma").reset_index(drop=True)


def evaluate_mot_dir(mot_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(os.listdir(mot_dir)):
        if not f.endswith(".mot"):
            continue
        fma = int(f.replace("FMA_", "").replace(".mot", ""))
        mot = pd.read_csv(os.path.join(mot_dir, f), sep="\t", skiprows=6)
        r = {"fma": fma}
        for col, key in [("elbow_flexion","elbow_rom_deg"), ("pro_sup","pro_sup_rom_deg")]:
            if col in mot.columns:
                v = mot[col].values * (180 / np.pi)
                r[key] = float(v.max() - v.min())
            else:
                r[key] = np.nan
        rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["fma", "elbow_rom_deg", "pro_sup_rom_deg"])
    return pd.DataFrame(rows).sort_values("fma").reset_index(drop=True)


def compute_gradient_rho(metrics_df: pd.DataFrame, metric: str) -> float:
    """Spearman rho between FMA and metric (proxy for clinical gradient)."""
    sub = metrics_df[["fma", metric]].dropna()
    if len(sub) < 4:
        return np.nan
    rho, _ = stats.spearmanr(sub["fma"], sub[metric])
    return float(rho)


def summarise(metrics_df: pd.DataFrame) -> dict:
    """Aggregate per-FMA metrics to a single summary dict."""
    return {
        "n_files":            len(metrics_df),
        "wrist_rho":          round(compute_gradient_rho(metrics_df, "wrist_range_mm"), 3),
        "trunk_rho":          round(compute_gradient_rho(metrics_df, "trunk_ratio"), 3),
        "segment_std_mean":   round(metrics_df["segment_std_mm"].mean(), 2),
        "boundary_ratio_mean":round(metrics_df["boundary_ratio"].mean(), 3),
        "peak_vel_mean":      round(metrics_df["peak_velocity_mm"].mean(), 2),
    }


def evaluate_experiment(exp_name: str, verbose: bool = True) -> dict:
    out_dir = os.path.join(TEST_DIR, "output", exp_name)
    csv_dir = os.path.join(out_dir, "csv")

    if not os.path.isdir(csv_dir) or len(os.listdir(csv_dir)) == 0:
        print(f"[{exp_name}] No CSV files found. Run generate.py first.")
        return {}

    metrics_df = evaluate_csv_dir(csv_dir)

    # Save full per-FMA breakdown — needed for figure regeneration
    per_fma_path = os.path.join(out_dir, "metrics_per_fma.csv")
    metrics_df.to_csv(per_fma_path, index=False)

    summary = summarise(metrics_df)
    summary["experiment"] = exp_name

    # Save aggregated summary
    summary_path = os.path.join(out_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  {exp_name}")
        print(f"{'─'*55}")
        print(f"  Files evaluated:      {summary['n_files']}")
        print(f"  Wrist range rho:      {summary['wrist_rho']}    (primary, target ≥ 0.9)")
        print(f"  Trunk ratio rho:      {summary['trunk_rho']}    (secondary, target < 0)")
        print(f"  Segment std (mean):   {summary['segment_std_mean']} mm  (target ~6mm)")
        print(f"  Boundary ratio (mean):{summary['boundary_ratio_mean']}  (target ~1.0)")
        print(f"  Peak velocity (mean): {summary['peak_vel_mean']} mm/frame")
        print(f"  Saved: {per_fma_path}")

    return summary


def print_summary_table():
    """Print comparison table across all experiments that have eval_summary.json."""
    rows = []
    exp_dir = os.path.join(TEST_DIR, "output")
    if not os.path.isdir(exp_dir):
        print("No output directory found.")
        return

    for exp_name in sorted(os.listdir(exp_dir)):
        path = os.path.join(exp_dir, exp_name, "eval_summary.json")
        if os.path.exists(path):
            with open(path) as f:
                rows.append(json.load(f))

    if not rows:
        print("No completed evaluations yet.")
        return

    df = pd.DataFrame(rows).set_index("experiment")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\n" + "="*70)
    print("  test2 Experiment Summary")
    print("="*70)
    print(df.to_string())
    print()

    # Highlight key metrics
    if "wrist_rho" in df.columns:
        best = df["wrist_rho"].idxmax()
        print(f"  Best wrist_rho:       {best} ({df.loc[best,'wrist_rho']:.3f})")
    if "segment_std_mean" in df.columns:
        best = df["segment_std_mean"].idxmin()
        print(f"  Best segment std:     {best} ({df.loc[best,'segment_std_mean']:.2f} mm)")
    if "boundary_ratio_mean" in df.columns:
        best = df["boundary_ratio_mean"].idxmin()
        print(f"  Best boundary ratio:  {best} ({df.loc[best,'boundary_ratio_mean']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", default=None)
    parser.add_argument("--compare", default=None, help="Second experiment to compare against")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        print_summary_table()
    elif args.experiment:
        evaluate_experiment(args.experiment)
        if args.compare:
            evaluate_experiment(args.compare)
    else:
        print_summary_table()
