"""
===============================================================================
FILE: run_generated_pipeline.py
===============================================================================
Bridge script: takes CVAE-generated CSVs and runs them through the full
MyoSuite pipeline (CSV → TRC → MOT → Video + Inverse Dynamics).

Usage:
    python run_generated_pipeline.py                    # Process all generated FMA files
    python run_generated_pipeline.py FMA_50.csv         # Process a single file
    python run_generated_pipeline.py --skip-id          # Skip inverse dynamics
===============================================================================
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED_DIR = os.path.join(WORKSPACE, "IK", "cutoff", "output", "generated")
OUTPUT_DIR = os.path.join(WORKSPACE, "IK", "output", "generated")
MODEL_PATH = os.path.join(WORKSPACE, "model", "myo_sim", "arm", "myoarm.xml")
REFERENCE_MOT = os.path.join(WORKSPACE, "IK", "output", "mot", "p_originals", "S5_12_1.mot")
IK_DIR = os.path.join(WORKSPACE, "IK")
CONDA_PYTHON = "/home/abdul/miniconda3/envs/MyoSuite/bin/python"

DATA_RATE = 200.0

# Column mapping: generated CSV columns → TRC marker names
MARKER_MAP = {
    "V_Shoulder": ("Sh_x", "Sh_y", "Sh_z"),
    "V_Elbow":    ("El_x", "El_y", "El_z"),
    "V_Wrist":    ("Wr_x", "Wr_y", "Wr_z"),
    "V_Vector":   ("WrVec_x", "WrVec_y", "WrVec_z"),
}

# =============================================================================
# STEP 1: Generated CSV → TRC
# =============================================================================
def convert_generated_csv_to_trc(csv_path, trc_path):
    """
    Reads a CVAE-generated flat CSV and writes a proper TRC file
    compatible with convert_trc2mot.py and the MyoSuite TRCParser.
    """
    df = pd.read_csv(csv_path)
    num_frames = len(df)
    markers = list(MARKER_MAP.keys())
    num_markers = len(markers)

    # Build the data block
    data_out = pd.DataFrame()
    data_out["Frame#"] = range(1, num_frames + 1)
    data_out["Time"] = np.arange(num_frames) / DATA_RATE

    for marker, (cx, cy, cz) in MARKER_MAP.items():
        if marker == "V_Vector":
            # WrVec is a direction (WRB - WRA). V_Vector site on the model is a
            # position, so reconstruct it: V_Vector = V_Wrist + WrVec
            data_out[f"{marker}_X"] = df["Wr_x"].values + df[cx].values
            data_out[f"{marker}_Y"] = df["Wr_y"].values + df[cy].values
            data_out[f"{marker}_Z"] = df["Wr_z"].values + df[cz].values
        else:
            data_out[f"{marker}_X"] = df[cx].values
            data_out[f"{marker}_Y"] = df[cy].values
            data_out[f"{marker}_Z"] = df[cz].values

    # Write TRC header (matches format expected by TRCParser)
    os.makedirs(os.path.dirname(trc_path), exist_ok=True)
    with open(trc_path, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(trc_path)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{DATA_RATE}\t{DATA_RATE}\t{num_frames}\t{num_markers}\tmm\t{DATA_RATE}\t1\t{num_frames}\n")
        # Marker name row
        f.write("Frame#\tTime\t" + "\t".join([f"{m}\t\t" for m in markers]) + "\n")
        # Axis label row
        f.write("\t\t" + "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(num_markers)]) + "\n")
        # Blank line
        f.write("\n")

    data_out.to_csv(trc_path, sep="\t", index=False, header=False, mode="a", lineterminator="\n")
    print(f"  [1/4] TRC written: {trc_path} ({num_frames} frames, {num_markers} markers)")
    return trc_path


# =============================================================================
# STEP 2: TRC → MOT (Inverse Kinematics)
# =============================================================================
def run_ik(trc_path, mot_path):
    """Runs convert_trc2mot.py with INTERACTIVE_ALIGN=False."""
    script = os.path.join(IK_DIR, "convert_trc2mot.py")

    # Disable interactive alignment and set reference MOT for automated use
    env = os.environ.copy()
    env["PYTHONPATH"] = IK_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["IK_INTERACTIVE_ALIGN"] = "false"
    env["IK_REFERENCE_MOT"] = REFERENCE_MOT

    result = subprocess.run(
        [CONDA_PYTHON, script, MODEL_PATH, trc_path, mot_path],
        capture_output=True, text=True, cwd=IK_DIR, env=env
    )

    if result.returncode != 0:
        print(f"  [2/4] IK FAILED:\n{result.stderr[-500:]}")
        return False

    # Extract mean error from output
    for line in result.stdout.splitlines():
        if "FINAL_MEAN_ERROR" in line:
            print(f"  [2/4] IK done: {line.strip()}")
            break
    else:
        print(f"  [2/4] IK done: {mot_path}")

    return os.path.exists(mot_path)


# =============================================================================
# STEP 3: MOT → Video
# =============================================================================
def run_video(mot_path, video_path):
    """Runs convert_mot2video.py to render the motion."""
    script = os.path.join(IK_DIR, "convert_mot2video.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = IK_DIR + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [CONDA_PYTHON, script, MODEL_PATH, mot_path, video_path],
        capture_output=True, text=True, cwd=IK_DIR, env=env
    )

    if result.returncode != 0:
        print(f"  [3/4] Video FAILED:\n{result.stderr[-500:]}")
        return False

    print(f"  [3/4] Video rendered: {video_path}")
    return True


# =============================================================================
# STEP 4: MOT → Inverse Dynamics
# =============================================================================
def run_inverse_dynamics(mot_path, id_output_dir):
    """Runs calc_mot2invdyn.py for forces/torques/muscle activations."""
    script = os.path.join(IK_DIR, "calc_mot2invdyn.py")

    # calc_mot2invdyn.py uses hardcoded paths, so we create a small wrapper
    wrapper_code = f"""
import sys
sys.path.insert(0, '{IK_DIR}')

# Override config before importing
import calc_mot2invdyn
calc_mot2invdyn.MOT_FILE_PATH = r'{mot_path}'
calc_mot2invdyn.MODEL_XML_PATH = r'{MODEL_PATH}'
calc_mot2invdyn.OUTPUT_DIRECTORY = r'{id_output_dir}'
calc_mot2invdyn.GENERATE_PLOTS = True
calc_mot2invdyn.GENERATE_VIDEO = False

calc_mot2invdyn.run_inverse_dynamics()
calc_mot2invdyn.plot_results()
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = IK_DIR + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [CONDA_PYTHON, "-c", wrapper_code],
        capture_output=True, text=True, cwd=IK_DIR, env=env
    )

    if result.returncode != 0:
        print(f"  [4/4] ID FAILED:\n{result.stderr[-500:]}")
        return False

    print(f"  [4/4] Inverse dynamics: {id_output_dir}")
    return True


# =============================================================================
# MAIN
# =============================================================================
def process_file(csv_path, skip_id=False):
    """Run the full pipeline on one generated CSV."""
    name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    # Output paths
    trc_path = os.path.join(OUTPUT_DIR, "trc", f"{name}.trc")
    mot_path = os.path.join(OUTPUT_DIR, "mot", f"{name}.mot")
    video_path = os.path.join(OUTPUT_DIR, "videos", f"{name}.mp4")
    id_dir = os.path.join(OUTPUT_DIR, "ID_results", name)

    os.makedirs(os.path.join(OUTPUT_DIR, "trc"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "mot"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "videos"), exist_ok=True)

    # Step 1: CSV → TRC
    convert_generated_csv_to_trc(csv_path, trc_path)

    # Step 2: TRC → MOT (IK)
    if not run_ik(trc_path, mot_path):
        print(f"  PIPELINE STOPPED for {name} — IK failed.")
        return False

    # Step 3: MOT → Video
    run_video(mot_path, video_path)

    # Step 4: MOT → Inverse Dynamics
    if not skip_id:
        run_inverse_dynamics(mot_path, id_dir)

    print(f"\nDone: {name}")
    return True


def main():
    skip_id = "--skip-id" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        # Process specific file(s)
        for arg in args:
            csv_path = os.path.join(GENERATED_DIR, arg) if not os.path.isabs(arg) else arg
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                continue
            process_file(csv_path, skip_id)
    else:
        # Process all generated files
        csvs = sorted([f for f in os.listdir(GENERATED_DIR) if f.endswith(".csv")])
        if not csvs:
            print(f"No CSV files found in {GENERATED_DIR}")
            return
        print(f"Found {len(csvs)} generated files: {', '.join(csvs)}")
        for csv_file in csvs:
            process_file(os.path.join(GENERATED_DIR, csv_file), skip_id)

    print(f"\n{'='*60}")
    print("ALL DONE. Outputs in:", OUTPUT_DIR)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
