"""
regen_n10.py — Re-generate all completed experiments at N=10 and evaluate.

Saves to output/{exp}/csv_n10/ and output/{exp}/metrics_per_fma_n10.csv
so N=1 results are preserved alongside N=10.
"""

import os, sys, json
import numpy as np
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TEST_DIR)
sys.path.insert(0, TEST_DIR)
sys.path.insert(0, ROOT_DIR)

from experiments import EXPERIMENTS
from generate import generate_for_experiment
from evaluate import evaluate_csv_dir, summarise

N       = 10
FMA_ALL = list(range(16, 67))


def run(phase: str = "A"):
    exp_names = [k for k in EXPERIMENTS if k.startswith(phase)]
    results = []

    for exp_name in exp_names:
        out_dir     = os.path.join(TEST_DIR, "output", exp_name)
        csv_n10_dir = os.path.join(out_dir, "csv_n10")
        best_path   = os.path.join(out_dir, "model_best.pth")

        if not os.path.exists(best_path):
            print(f"[SKIP] {exp_name} — no trained model found")
            continue

        print(f"\n[{exp_name}] Generating N={N}...", flush=True)
        generate_for_experiment(exp_name, FMA_ALL,
                                checkpoint="best",
                                n_samples=N,
                                out_subdir="csv_n10")

        if not os.path.isdir(csv_n10_dir) or not os.listdir(csv_n10_dir):
            print(f"  [SKIP eval] no CSVs generated")
            continue

        metrics_df = evaluate_csv_dir(csv_n10_dir)
        metrics_df.to_csv(os.path.join(out_dir, "metrics_per_fma_n10.csv"), index=False)

        summary = summarise(metrics_df)
        summary["experiment"] = exp_name
        with open(os.path.join(out_dir, "eval_summary_n10.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  wrist_rho={summary['wrist_rho']}  trunk_rho={summary['trunk_rho']}  "
              f"seg_std={summary['segment_std_mean']}mm  boundary={summary['boundary_ratio_mean']}")
        results.append(summary)

    if not results:
        print("No results.")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    n1_log = os.path.join(TEST_DIR, "results", "results_log.csv")
    if os.path.exists(n1_log):
        df_n1 = pd.read_csv(n1_log, usecols=lambda c: c in
                            ["experiment","wrist_rho","trunk_rho","segment_std_mean",
                             "boundary_ratio_mean","peak_vel_mean","n_files"])
        df_n1 = df_n1[df_n1["experiment"].str.startswith(phase)].set_index("experiment")
    else:
        df_n1 = pd.DataFrame()

    df_n10 = pd.DataFrame(results).set_index("experiment")

    print(f"\n{'='*75}")
    print(f"Phase {phase} — N=1 vs N=10 comparison")
    print(f"{'='*75}")
    print(f"\n{'Experiment':<20} {'wrist_rho N=1':>14} {'wrist_rho N=10':>15} "
          f"{'trunk_rho N=1':>14} {'trunk_rho N=10':>15}")
    print("-" * 80)
    for exp in sorted(df_n10.index):
        r1_w  = df_n1.loc[exp, "wrist_rho"]  if (len(df_n1) > 0 and exp in df_n1.index) else float("nan")
        r10_w = df_n10.loc[exp, "wrist_rho"]
        r1_t  = df_n1.loc[exp, "trunk_rho"]  if (len(df_n1) > 0 and exp in df_n1.index) else float("nan")
        r10_t = df_n10.loc[exp, "trunk_rho"]
        print(f"{exp:<20} {r1_w:>14.3f} {r10_w:>15.3f} {r1_t:>14.3f} {r10_t:>15.3f}")

    # Save N=10 results log
    os.makedirs(os.path.join(TEST_DIR, "results"), exist_ok=True)
    n10_log = os.path.join(TEST_DIR, "results", f"results_log_{phase.lower()}_n10.csv")
    df_n10.reset_index().to_csv(n10_log, index=False)
    print(f"\nN=10 results saved to: {n10_log}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="A", help="Phase letter: A or B")
    args = parser.parse_args()
    run(args.phase)
