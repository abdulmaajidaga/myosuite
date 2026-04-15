"""
run_experiment.py — End-to-end runner for a single test2 experiment.

Phases:
  1. train     — train the model (skip if model already exists)
  2. generate  — generate CSVs for FMA 16-66 (or a subset)
  3. ik        — run inverse kinematics to produce MOT files
  4. evaluate  — compute quick metrics and write eval_summary.json

Usage:
  # Full pipeline
  python test/run_experiment.py --experiment A1_no_residual

  # Only specific phases
  python test/run_experiment.py --experiment C3_both --phases generate,ik,evaluate

  # Fast check: generate 5 key FMA levels, evaluate, no IK
  python test/run_experiment.py --experiment A1_no_residual --fma 18,30,40,50,66 --phases generate,evaluate

  # Retrain even if model exists
  python test/run_experiment.py --experiment A1_no_residual --force-train

  # List all experiments
  python test/run_experiment.py --list
"""

import os, sys, argparse, subprocess, time

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TEST_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TEST_DIR)

from experiments import EXPERIMENTS
from generate import generate_for_experiment
from evaluate import evaluate_experiment, print_summary_table

# IK pipeline from test/ (reuse — no need to duplicate)
TEST_SRC_IK = os.path.join(ROOT_DIR, "test", "src", "inverse_kinematics", "convert_trc2mot.py")
TEST_SRC_TRC = os.path.join(ROOT_DIR, "test", "src", "data_processing")
MUJOCO_MODEL = os.path.join(ROOT_DIR, "models", "model", "myo_sim", "arm", "myoarm.xml")
PYTHON       = sys.executable


def _run(cmd, desc=""):
    print(f"\n  [{desc}] {' '.join(cmd[:4])}...")
    t = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return False
    print(f"  Done in {elapsed:.1f}s")
    return True


def phase_train(exp_name, epochs, max_samples, force):
    out_dir   = os.path.join(TEST_DIR, "output", exp_name)
    best_path = os.path.join(out_dir, "model_best.pth")
    if os.path.exists(best_path) and not force:
        print(f"  [train] Model already exists at {best_path} — skipping. Use --force-train to retrain.")
        return True
    cmd = [PYTHON, os.path.join(TEST_DIR, "train.py"),
           "--experiment", exp_name,
           "--epochs", str(epochs),
           "--max-samples", str(max_samples)]
    return _run(cmd, "train")


def phase_generate(exp_name, fma_scores, checkpoint):
    try:
        generate_for_experiment(exp_name, fma_scores, checkpoint)
        return True
    except Exception as e:
        print(f"  [generate] ERROR: {e}")
        return False


def _generated_csv_to_trc(csv_path: str, trc_path: str):
    """
    Convert a CVAE-generated CSV (Sh/El/Wr/WrVec/Trunk columns, absolute mm) to TRC.
    Replicates the production pipeline in scripts/run_generated_pipeline.py:
      - 5 markers: V_Shoulder, V_Elbow, V_Wrist, V_Vector, V_Sternum
      - V_Vector = V_Wrist + WrVec (per CLAUDE.md)
      - V_Sternum = Trunk_x/y/z
      - Temporal scaling applied (100 CVAE frames → 200 Hz realistic duration)
      - Output DATA_RATE = 200 Hz
    """
    import pandas as pd
    import numpy as np
    from scipy.interpolate import interp1d

    DATA_RATE = 200.0
    df = pd.read_csv(csv_path)

    # Temporal scaling: stretch 100 CVAE frames to a realistic movement duration
    # Using the same approach as scripts/run_generated_pipeline.py
    fma_score = None
    bn = os.path.splitext(os.path.basename(csv_path))[0]
    if bn.upper().startswith("FMA_"):
        try:
            fma_score = int(bn.split("_")[1])
        except (IndexError, ValueError):
            pass

    raw_data = df.values  # (100, 15)
    if fma_score is not None:
        # Simple linear model from temporal_scaling: ~3.5s for severe, ~1.5s for healthy
        # Approximate the predict_duration function without importing it
        fma_norm = fma_score / 66.0
        target_duration = max(1.0, 3.5 - 2.0 * fma_norm)  # seconds
        target_frames = int(round(target_duration * DATA_RATE))
        n_orig = raw_data.shape[0]
        t_orig = np.linspace(0, 1, n_orig)
        t_new  = np.linspace(0, 1, target_frames)
        scaled_data = np.zeros((target_frames, raw_data.shape[1]))
        for i in range(raw_data.shape[1]):
            f = interp1d(t_orig, raw_data[:, i], kind='cubic')
            scaled_data[:, i] = f(t_new)
        df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    else:
        df_scaled = df

    cols = df.columns.tolist()
    num_frames = len(df_scaled)
    markers = ['V_Shoulder', 'V_Elbow', 'V_Wrist', 'V_Vector', 'V_Sternum']

    out = pd.DataFrame()
    out['Frame#'] = range(1, num_frames + 1)
    out['Time']   = np.arange(num_frames) / DATA_RATE

    marker_map = {
        'V_Shoulder': ('Sh_x',    'Sh_y',    'Sh_z'),
        'V_Elbow':    ('El_x',    'El_y',    'El_z'),
        'V_Wrist':    ('Wr_x',    'Wr_y',    'Wr_z'),
        'V_Vector':   None,  # Wr + WrVec
        'V_Sternum':  ('Trunk_x', 'Trunk_y', 'Trunk_z'),
    }
    for m in markers:
        if m == 'V_Vector':
            out['V_Vector_X'] = df_scaled['Wr_x'].values + df_scaled['WrVec_x'].values
            out['V_Vector_Y'] = df_scaled['Wr_y'].values + df_scaled['WrVec_y'].values
            out['V_Vector_Z'] = df_scaled['Wr_z'].values + df_scaled['WrVec_z'].values
        else:
            cx, cy, cz = marker_map[m]
            out[f'{m}_X'] = df_scaled[cx].values
            out[f'{m}_Y'] = df_scaled[cy].values
            out[f'{m}_Z'] = df_scaled[cz].values

    os.makedirs(os.path.dirname(trc_path) or '.', exist_ok=True)
    with open(trc_path, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(trc_path)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{DATA_RATE}\t{DATA_RATE}\t{num_frames}\t{len(markers)}\tmm\t{DATA_RATE}\t1\t{num_frames}\n")
        f.write("Frame#\tTime\t" + "\t".join([f"{m}\t\t" for m in markers]) + "\n")
        f.write("\t\t" + "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(len(markers))]) + "\n")
        f.write("\n")
    out.to_csv(trc_path, sep='\t', index=False, header=False, mode='a', lineterminator='\n')


def phase_ik(exp_name):
    """Convert CSVs → TRC → MOT using the shared IK pipeline."""
    out_dir = os.path.join(TEST_DIR, "output", exp_name)
    csv_dir = os.path.join(out_dir, "csv")
    trc_dir = os.path.join(out_dir, "trc")
    mot_dir = os.path.join(out_dir, "mot")
    os.makedirs(trc_dir, exist_ok=True)
    os.makedirs(mot_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    print(f"  [ik] Converting {len(csv_files)} files: CSV → TRC → MOT")

    # Use test/ IK script — it has try/except fallback for TRCParser
    # (main workspace version lacks myosuite.utils.trc_parser fallback)
    trc2mot_script = os.path.join(ROOT_DIR, "test", "src", "inverse_kinematics", "convert_trc2mot.py")
    ref_mot = os.path.join(ROOT_DIR, "output", "originals", "mot", "S5_12_1.mot")

    env = os.environ.copy()
    env["IK_REFERENCE_MOT"] = ref_mot
    env["IK_INTERACTIVE_ALIGN"] = "false"
    # PYTHONPATH: test/ FIRST so src.utils.trc_parser resolves to test/src/utils/trc_parser.py
    # then main workspace for src.utils.config, then myosuite root
    test_src_dir = os.path.join(ROOT_DIR, "test")
    env["PYTHONPATH"] = (test_src_dir + os.pathsep +
                         ROOT_DIR + os.pathsep +
                         os.path.join(ROOT_DIR, "..") + os.pathsep +
                         env.get("PYTHONPATH", ""))

    failed = 0
    for fname in csv_files:
        fma = fname.replace(".csv", "")
        csv_path = os.path.join(csv_dir, fname)
        trc_path = os.path.join(trc_dir, f"{fma}.trc")
        mot_path = os.path.join(mot_dir, f"{fma}.mot")

        if os.path.exists(mot_path):
            continue  # resume-safe

        # Step 1: CSV → TRC (inline, handles generated format directly)
        try:
            _generated_csv_to_trc(csv_path, trc_path)
        except Exception as e:
            print(f"    {fma}: CSV→TRC FAILED — {e}")
            failed += 1
            continue

        # Step 2: TRC → MOT (subprocess, reuses test/ IK solver)
        r2 = subprocess.run(
            [PYTHON, trc2mot_script, MUJOCO_MODEL, trc_path, mot_path],
            cwd=ROOT_DIR, env=env,
            capture_output=True, text=True
        )
        if r2.returncode != 0:
            print(f"    {fma}: TRC→MOT FAILED — {r2.stderr[-300:]}")
            failed += 1
        else:
            print(f"    {fma}: OK")

    print(f"  [ik] Done. {len(csv_files) - failed}/{len(csv_files)} succeeded.")
    return failed == 0


def phase_evaluate(exp_name):
    summary = evaluate_experiment(exp_name, verbose=True)
    if not summary:
        return False
    # Append to master results log
    import json, csv
    log_path = os.path.join(TEST_DIR, "results", "results_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)
    print(f"  Results logged to {log_path}")
    return True


def run(exp_name, phases, fma_scores, epochs, max_samples,
        force_train, checkpoint):

    if exp_name not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_name}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    cfg = EXPERIMENTS[exp_name]
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"  {cfg['desc']}")
    print(f"  Phases: {', '.join(phases)}")
    print(f"  FMA scores: {fma_scores[0]}–{fma_scores[-1]} ({len(fma_scores)} files)")
    print(f"{'='*60}")

    t_start = time.time()

    if "train" in phases:
        ok = phase_train(exp_name, epochs, max_samples, force_train)
        if not ok:
            print("ABORTED at train phase.")
            return

    if "generate" in phases:
        ok = phase_generate(exp_name, fma_scores, checkpoint)
        if not ok:
            print("ABORTED at generate phase.")
            return

    if "ik" in phases:
        ok = phase_ik(exp_name)
        if not ok:
            print("IK phase had failures — continuing to evaluate anyway.")

    if "evaluate" in phases:
        phase_evaluate(exp_name)

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", default=None)
    parser.add_argument("--phases", default="train,generate,ik,evaluate",
                        help="Comma-separated phases to run")
    parser.add_argument("--fma", default="16-66",
                        help="FMA range '16-66' or list '18,30,40,50,66'")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=15000)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--checkpoint", default="best", choices=["best","final"])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:\n")
        for phase in ["A", "B", "C"]:
            print(f"  Phase {phase}:")
            for name, cfg in EXPERIMENTS.items():
                if name.startswith(phase):
                    print(f"    {name:25s}  {cfg['desc']}")
            print()
        sys.exit(0)

    if args.summary:
        print_summary_table()
        sys.exit(0)

    if not args.experiment:
        parser.print_help()
        sys.exit(1)

    phases = [p.strip() for p in args.phases.split(",")]

    if "-" in args.fma:
        lo, hi = map(int, args.fma.split("-"))
        scores = list(range(lo, hi + 1))
    else:
        scores = [int(x) for x in args.fma.split(",")]

    run(args.experiment, phases, scores,
        args.epochs, args.max_samples,
        args.force_train, args.checkpoint)
