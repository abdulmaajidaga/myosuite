"""
Generate both comparison datasets for the paper:
  output/originals/    — standard IK, trunk free, NO damping (pure pseudoinverse)
  output/neutral_start/ — best method: SVD two-pass ik_clean.py

Run:
  python scripts/run_paper_comparison_batch.py                  # both
  python scripts/run_paper_comparison_batch.py --only originals
  python scripts/run_paper_comparison_batch.py --only neutral_start
  python scripts/run_paper_comparison_batch.py --skip-id
"""
import os
import sys
import glob
import time
import argparse
import subprocess
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root, get

PYTHON   = get("conda_python", default=sys.executable)
ROOT     = get_project_root()
MODEL_IK = get_path("mujoco_arm_model").replace("myoarm.xml", "myoarm_trunk.xml")

SCRIPT_CSV2TRC     = os.path.join(ROOT, "src", "data_processing", "convert_csv2trc.py")
SCRIPT_IK_STANDARD = os.path.join(ROOT, "paper_publication_assets", "code_supplementary",
                                   "ik_clean_primary_weighted.py")
SCRIPT_IK_BEST     = os.path.join(ROOT, "src", "inverse_kinematics", "ik_clean.py")
SCRIPT_ID          = os.path.join(ROOT, "src", "inverse_dynamics", "calc_mot2invdyn.py")

HEALTHY_DIR = get_path("data_healthy")
STROKE_DIR  = get_path("data_stroke")

# originals: no-damping standard IK; neutral_start: SVD two-pass best method
CONFIGS = {
    "originals":     "standard_nodamping",
    "neutral_start": "svd_best",
}

parser = argparse.ArgumentParser()
parser.add_argument("--only",    choices=list(CONFIGS.keys()), default=None)
parser.add_argument("--skip-id", action="store_true")
args = parser.parse_args()

SKIP_ID = args.skip_id
RUNS    = {args.only: CONFIGS[args.only]} if args.only else CONFIGS


def find_csvs(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*.csv")) if os.path.isfile(f))


def process_ik(csv_path, group, out_base, mode, results):
    file_id = os.path.splitext(os.path.basename(csv_path))[0]
    trc_dir = os.path.join(out_base, group, "trc")
    mot_dir = os.path.join(out_base, group, "mot")
    os.makedirs(trc_dir, exist_ok=True)
    os.makedirs(mot_dir, exist_ok=True)

    trc_path = os.path.join(trc_dir, f"{file_id}.trc")
    mot_path = os.path.join(mot_dir, f"{file_id}.mot")
    status   = "OK"
    ik_error = "N/A"
    t0       = time.time()

    try:
        subprocess.check_call(
            [PYTHON, SCRIPT_CSV2TRC, csv_path, trc_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=ROOT
        )
        env = os.environ.copy()
        env["IK_INTERACTIVE_ALIGN"] = "False"

        if mode == "standard_nodamping":
            # Standard IK: trunk free, pure pseudoinverse, no damping
            cmd = [PYTHON, SCRIPT_IK_STANDARD, MODEL_IK, trc_path, mot_path,
                   "0",    # max_iters_override: 0 = use config default
                   "0.0",  # TR_DAMPING  (unused — nodamping overrides)
                   "0.0",  # SC_DAMPING  (unused — nodamping overrides)
                   "0.0",  # DAMPING     (unused — nodamping overrides)
                   "nodamping"]
        else:
            # Best method: SVD two-pass
            cmd = [PYTHON, SCRIPT_IK_BEST, MODEL_IK, trc_path, mot_path, "0"]

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd[1],
                                                stderr=proc.stderr)
        for line in proc.stdout.splitlines():
            if "FINAL_MEAN_ERROR:" in line:
                ik_error = line.split(":")[1].strip()
    except subprocess.CalledProcessError as e:
        status = "FAIL"
        tail = (e.stderr or "")[-300:] if isinstance(e.stderr, str) else \
               (e.stderr or b"").decode(errors="ignore")[-300:]
        print(f"  FAILED {file_id}: {tail}")
    except Exception as e:
        status = "ERROR"
        print(f"  ERROR {file_id}: {e}")

    elapsed = time.time() - t0
    if status == "OK":
        print(f"  [OK] {file_id} ({elapsed:.1f}s)  IK={ik_error} mm")
    results.append({"file": file_id, "group": group, "status": status,
                    "ik_error_mm": ik_error, "time_s": round(elapsed, 1)})


def run_id(group, out_base):
    model_xml = get_path("mujoco_arm_model")
    mot_dir = os.path.join(out_base, group, "mot")
    id_root = os.path.join(out_base, group, "id")
    os.makedirs(id_root, exist_ok=True)
    print(f"\n=== ID: {group} ===")
    t0   = time.time()
    proc = subprocess.run(
        [PYTHON, SCRIPT_ID, mot_dir, id_root, model_xml],
        capture_output=True, text=True
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [WARN] ID failed ({elapsed:.1f}s):\n{proc.stderr[-400:]}")
    else:
        print(f"  [OK] ID complete ({elapsed:.1f}s)")


for run_name, mode in RUNS.items():
    out_base = os.path.join(ROOT, "output", run_name)
    results  = []

    print(f"\n{'='*60}")
    print(f"RUN: {run_name}  mode={mode}")
    print(f"Output: {out_base}")
    print(f"{'='*60}")

    healthy_csvs = find_csvs(HEALTHY_DIR)
    stroke_csvs  = find_csvs(STROKE_DIR)
    total        = len(healthy_csvs) + len(stroke_csvs)
    print(f"Files: {len(stroke_csvs)} stroke + {len(healthy_csvs)} healthy = {total}\n")

    print("--- STROKE ---")
    for i, csv_path in enumerate(stroke_csvs):
        fid = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"[{i+1}/{len(stroke_csvs)}] {fid}")
        process_ik(csv_path, "stroke", out_base, mode, results)

    print("\n--- HEALTHY ---")
    for i, csv_path in enumerate(healthy_csvs):
        fid = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"[{i+1}/{len(healthy_csvs)}] {fid}")
        process_ik(csv_path, "healthy", out_base, mode, results)

    df = pd.DataFrame(results)
    os.makedirs(out_base, exist_ok=True)
    df.to_csv(os.path.join(out_base, "batch_report.csv"), index=False)

    ok   = df[df["status"] == "OK"]
    fail = df[df["status"] != "OK"]
    print(f"\n{'-'*50}")
    print(f"IK done: {len(ok)}/{total} succeeded, {len(fail)} failed")
    if len(fail):
        print("Failed:", fail["file"].tolist())
    numeric = pd.to_numeric(ok["ik_error_mm"], errors="coerce").dropna()
    if len(numeric):
        print(f"IK error (mm): mean={numeric.mean():.2f}  "
              f"p95={numeric.quantile(0.95):.2f}  max={numeric.max():.2f}")

    if not SKIP_ID:
        for group in ["stroke", "healthy"]:
            run_id(group, out_base)

print("\nDone.")
