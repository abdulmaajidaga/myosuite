"""
run_ephase.py — Guidance scale sweep on the trained D1 model (no retraining).

Tests guidance_scale = 1.5, 2.0 (reference), 2.5, 3.0, 4.0 by regenerating
from D1 model weights with different CFG amplification at inference.

Creates test/output/E_gs_{scale}/ directories, each with its own CSV set
and eval_summary.json so they appear in the --summary table alongside A-D.

Usage:
    python test/run_ephase.py
"""

import os, sys, json, shutil
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TEST_DIR)
sys.path.insert(0, ROOT_DIR)

from generate  import generate_for_experiment
from evaluate  import evaluate_experiment

D1_DIR    = os.path.join(TEST_DIR, "output", "D1_stage3_sag")
FMA_RANGE = list(range(16, 67))

GUIDANCE_SCALES = [1.5, 2.0, 2.5, 3.0, 4.0]
SEED = 42  # fixed seed so guidance scale is the ONLY variable

def scale_to_name(gs):
    return f"E_gs_{int(gs * 100):03d}"   # 1.5 → E_gs_150, 3.0 → E_gs_300


def setup_e_dir(exp_name: str) -> str:
    """Create output dir for a guidance-scale experiment using D1 model weights."""
    out_dir = os.path.join(TEST_DIR, "output", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # Copy D1 config, model, scaler — same model, different inference param
    for fname in ["config.json", "model_best.pth", "scaler.pkl"]:
        src = os.path.join(D1_DIR, fname)
        dst = os.path.join(out_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Patch config to record the guidance scale used
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["experiment"] = exp_name
    cfg["guidance_scale_inference"] = float(exp_name.split("_")[-1]) / 100
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    return out_dir


def main():
    print("=" * 60)
    print("E-Phase: Guidance Scale Sweep on D1 model")
    print("=" * 60)
    print(f"Scales to test: {GUIDANCE_SCALES}")
    print(f"D1 model:       {D1_DIR}")
    print()

    results = {}

    for gs in GUIDANCE_SCALES:
        exp_name = scale_to_name(gs)
        print(f"\n{'─'*55}")
        print(f"  {exp_name}  (guidance_scale={gs})")
        print(f"{'─'*55}")

        setup_e_dir(exp_name)

        # Generate using D1 weights, override guidance scale.
        # Fixed seed ensures the only variable is guidance_scale.
        generate_for_experiment(exp_name, FMA_RANGE, checkpoint="best",
                                guidance=gs, apply_smooth=True, seed=SEED)

        summary = evaluate_experiment(exp_name, verbose=True)
        results[exp_name] = summary

    # Print comparison table
    print("\n" + "=" * 70)
    print("  E-Phase Summary — Guidance Scale Effect on D1 Model")
    print("=" * 70)
    header = f"{'Experiment':<20} {'guidance':>10} {'wrist_rho':>12} {'sag_dev':>10} {'seg_std':>10}"
    print(header)
    print("-" * 70)

    # Include D1 reference
    d1_path = os.path.join(D1_DIR, "eval_summary.json")
    if os.path.exists(d1_path):
        with open(d1_path) as f:
            d1 = json.load(f)
        print(f"  {'D1_stage3_sag (ref)':<18} {'2.0':>10} "
              f"{d1.get('wrist_rho','?'):>12} "
              f"{d1.get('sag_dev_mean','?'):>10} "
              f"{d1.get('segment_std_mean','?'):>10}")

    for gs in GUIDANCE_SCALES:
        exp_name = scale_to_name(gs)
        r = results.get(exp_name, {})
        print(f"  {exp_name:<20} {gs:>10.1f} "
              f"{r.get('wrist_rho','?'):>12} "
              f"{r.get('sag_dev_mean','?'):>10} "
              f"{r.get('segment_std_mean','?'):>10}")

    print()


if __name__ == "__main__":
    main()
