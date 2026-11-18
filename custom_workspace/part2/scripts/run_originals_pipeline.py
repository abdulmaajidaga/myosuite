"""
Batch pipeline: process all original healthy + stroke CSV files.
  healthy/  -> output/originals/healthy/
  stroke/   -> output/originals/stroke/

Stages:
  1. IK: CSV -> TRC -> MOT (clean sequential IK, no tweaks)
  2. ID: batch ID on each group's mot/ directory

Output layout:
  output/originals/{group}/trc/
  output/originals/{group}/mot/
  output/originals/{group}/id/{file_id}/

Reference MOT (S5_12_1) is processed first and reused as IK seed
for all subsequent files via IK_REFERENCE_MOT env var.

Run:
  cd /home/abdul/Desktop/myosuite/custom_workspace/test
  python scripts/run_originals_pipeline.py           # full IK + ID
  python scripts/run_originals_pipeline.py --skip-id # IK only
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

PYTHON = get("conda_python", default=sys.executable)
ROOT = get_project_root()

MODEL_XML    = get_path("mujoco_arm_model")
MODEL_XML_IK = MODEL_XML.replace("myoarm.xml", "myoarm_trunk.xml")
SCRIPT_CSV2TRC    = os.path.join(ROOT, "src", "data_processing", "convert_csv2trc.py")
SCRIPT_IK         = os.path.join(ROOT, "src", "inverse_kinematics", "ik_clean.py")
SCRIPT_ID         = os.path.join(ROOT, "src", "inverse_dynamics", "calc_mot2invdyn.py")
SCRIPT_INJECT_CS  = os.path.join(ROOT, "scripts", "inject_cs_trunk.py")

HEALTHY_DIR = get_path("data_healthy")
STROKE_DIR = get_path("data_stroke")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--skip-id",     action="store_true")
parser.add_argument("--trunk-weight", type=float, default=50.0)
parser.add_argument("--output-dir",   type=str,   default=None)
args, _ = parser.parse_known_args()

SKIP_ID    = args.skip_id
TR_WEIGHT  = args.trunk_weight
OUT_BASE   = args.output_dir or os.environ.get(
    "OUT_BASE", os.path.join(ROOT, "output", "originals")
)


def find_csvs(directory):
    """Return CSV files directly in directory (no subdirectories)."""
    return sorted(
        f for f in glob.glob(os.path.join(directory, "*.csv"))
        if os.path.isfile(f)
    )


def process_ik(csv_path, group, results):
    """Run CSV -> TRC -> MOT for one file. Returns mot_path on success."""
    file_id = os.path.splitext(os.path.basename(csv_path))[0]
    trc_dir = os.path.join(OUT_BASE, group, "trc")
    mot_dir = os.path.join(OUT_BASE, group, "mot")
    os.makedirs(trc_dir, exist_ok=True)
    os.makedirs(mot_dir, exist_ok=True)

    trc_path = os.path.join(trc_dir, f"{file_id}.trc")
    mot_path = os.path.join(mot_dir, f"{file_id}.mot")

    status = "OK"
    ik_error = "N/A"
    t0 = time.time()

    try:
        # CSV -> TRC
        subprocess.check_call(
            [PYTHON, SCRIPT_CSV2TRC, csv_path, trc_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=ROOT
        )

        # TRC -> MOT (clean IK)
        env = os.environ.copy()
        env["IK_INTERACTIVE_ALIGN"] = "False"

        proc = subprocess.run(
            [PYTHON, SCRIPT_IK, MODEL_XML_IK, trc_path, mot_path,
             "0", str(TR_WEIGHT)],
            capture_output=True, text=True, env=env, cwd=ROOT
        )
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, SCRIPT_IK,
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
    results.append({
        "file": file_id, "group": group,
        "status": status, "ik_error_mm": ik_error,
        "time_s": round(elapsed, 1),
    })
    return mot_path if status == "OK" else None


def run_id_for_group(group):
    """Run batch ID on a group's mot/ directory."""
    mot_dir = os.path.join(OUT_BASE, group, "mot")
    id_root = os.path.join(OUT_BASE, group, "id")
    os.makedirs(id_root, exist_ok=True)

    print(f"\n=== ID: {group} ===")
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, SCRIPT_ID, mot_dir, id_root, MODEL_XML],
        capture_output=True, text=True
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [WARN] ID failed for {group} ({elapsed:.1f}s):\n{proc.stderr[-400:]}")
    else:
        print(f"  [OK] ID complete for {group} ({elapsed:.1f}s)")


def main():
    healthy_csvs = find_csvs(HEALTHY_DIR)
    stroke_csvs = find_csvs(STROKE_DIR)

    total = len(healthy_csvs) + len(stroke_csvs)
    print(f"Found {len(stroke_csvs)} stroke + {len(healthy_csvs)} healthy = {total} files")
    print(f"Skip ID: {SKIP_ID}\n")

    results = []

    # --- Stage 1: IK for stroke ---
    print("=== IK: STROKE ===")
    for i, csv_path in enumerate(stroke_csvs):
        file_id = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"[{i+1}/{len(stroke_csvs)}] {file_id}")
        process_ik(csv_path, "stroke", results)

    # --- Stage 1: IK for healthy ---
    print("\n=== IK: HEALTHY ===")
    for i, csv_path in enumerate(healthy_csvs):
        file_id = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"[{i+1}/{len(healthy_csvs)}] {file_id}")
        process_ik(csv_path, "healthy", results)

    # --- Report IK ---
    df = pd.DataFrame(results)
    os.makedirs(OUT_BASE, exist_ok=True)
    df.to_csv(os.path.join(OUT_BASE, "batch_report.csv"), index=False)

    ok = df[df["status"] == "OK"]
    fail = df[df["status"] != "OK"]
    print(f"\n{'='*50}")
    print(f"IK done: {len(ok)}/{total} succeeded, {len(fail)} failed")
    if len(fail):
        print("Failed:", fail["file"].tolist())
    if len(ok):
        numeric = pd.to_numeric(ok["ik_error_mm"], errors="coerce").dropna()
        if len(numeric):
            print(f"IK error (mm): mean={numeric.mean():.2f}, "
                  f"p95={numeric.quantile(0.95):.2f}, max={numeric.max():.2f}")

    # --- Stage 2: ID per group ---
    if not SKIP_ID:
        for group in ["stroke", "healthy"]:
            run_id_for_group(group)

    print(f"\nReport: {os.path.join(OUT_BASE, 'batch_report.csv')}")


if __name__ == "__main__":
    main()
