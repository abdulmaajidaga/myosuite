"""
Helper script to run the complete pipeline: CSV → TRC → MOT → Video
All configuration is done directly in this file - no command line arguments needed.
"""
#!/usr/bin/env python3
import logging
import os
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION - Edit these paths
# ========================================

# Base directories
BASE_DIR = "/home/abdul/Desktop/myosuite/custom_workspace"
IK_DIR = os.path.join(BASE_DIR, "IK")

# Input/Output paths
INPUT_CSV = os.path.join(BASE_DIR, "data/kinematic/Stroke/S5_12_1.csv")  # Your input CSV file
MODEL_XML = "/home/abdul/Desktop/myosuite/custom_workspace/model/myo_sim/arm/myoarm.xml"  # MuJoCo model
OUTPUT_DIR = os.path.join(IK_DIR, "output")  # Where to save all outputs

# Script locations (these should now work without arguments)
CONVERT_CSV2TRC_SCRIPT = os.path.join(IK_DIR, "convert_csv2trc.py")
CONVERT_TRC2MOT_SCRIPT = os.path.join(IK_DIR, "convert_trc2mot.py")
CONVERT_MOT2VIDEO_SCRIPT = os.path.join(IK_DIR, "convert_mot2video.py")

# Pipeline settings
DATA_RATE = 200.0  # Sampling rate in Hz
VISUALIZE_ALIGNMENT = False  # Set to True to show alignment visualization instead of running full IK

# ========================================

def run_command(cmd, check=True):
    """Run a command and log output."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.error(result.stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, 
                                         output=result.stdout,
                                         stderr=result.stderr)
    return result

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate output filenames based on input CSV name
    base_name = Path(INPUT_CSV).stem
    trc_path = os.path.join(OUTPUT_DIR, f"{base_name}.trc")
    mot_path = os.path.join(OUTPUT_DIR, f"{base_name}.mot")
    video_path = os.path.join(OUTPUT_DIR, f"{base_name}.mp4")
    
    # Update the configuration in each script by temporarily modifying them
    # OR just inform the user to update the scripts manually
    
    logger.info("""
================================================================================
IMPORTANT: Before running this pipeline, make sure you've updated the 
configuration sections in these scripts with the correct paths:

1. convert_csv2trc.py - Set INPUT_CSV, OUTPUT_TRC, DATA_RATE
2. convert_trc2mot.py - Set MODEL_PATH, TRC_PATH, OUTPUT_PATH, VISUALIZE
3. convert_mot2video.py - Set MODEL_PATH, MOT_PATH, OUTPUT_VIDEO

Expected paths for this run:
- Input CSV: {INPUT_CSV}
- Model XML: {MODEL_XML}
- TRC output: {trc_path}
- MOT output: {mot_path}
- Video output: {video_path}
================================================================================
""")
    
    # 1. CSV → TRC
    logger.info("=== Step 1/3: Converting CSV to TRC ===")
    logger.info(f"Make sure convert_csv2trc.py has:")
    logger.info(f"  INPUT_CSV = '{INPUT_CSV}'")
    logger.info(f"  OUTPUT_TRC = '{trc_path}'")
    logger.info(f"  DATA_RATE = {DATA_RATE}")
    run_command(["python3", CONVERT_CSV2TRC_SCRIPT])
    
    # 2. TRC → MOT
    logger.info("=== Step 2/3: Running inverse kinematics (TRC → MOT) ===")
    logger.info(f"Make sure convert_trc2mot.py has:")
    logger.info(f"  MODEL_PATH = '{MODEL_XML}'")
    logger.info(f"  TRC_PATH = '{trc_path}'")
    logger.info(f"  OUTPUT_PATH = '{mot_path}'")
    logger.info(f"  VISUALIZE = {VISUALIZE_ALIGNMENT}")
    run_command(["python3", CONVERT_TRC2MOT_SCRIPT])
    
    # 3. MOT → Video (skip if visualizing)
    if not VISUALIZE_ALIGNMENT:
        logger.info("=== Step 3/3: Rendering video (MOT → Video) ===")
        logger.info(f"Make sure convert_mot2video.py has:")
        logger.info(f"  MODEL_PATH = '{MODEL_XML}'")
        logger.info(f"  MOT_PATH = '{mot_path}'")
        logger.info(f"  OUTPUT_VIDEO = '{video_path}'")
        run_command(["python3", CONVERT_MOT2VIDEO_SCRIPT])
        
        logger.info(f"""
================================================================================
Pipeline complete! Output files:
- TRC: {trc_path}
- MOT: {mot_path}
- Video: {video_path}
================================================================================
""")
    else:
        logger.info("""
================================================================================
Visualization mode enabled - pipeline stopped after alignment visualization.
Set VISUALIZE_ALIGNMENT = False to run the full pipeline.
================================================================================
""")

if __name__ == "__main__":
    main()
