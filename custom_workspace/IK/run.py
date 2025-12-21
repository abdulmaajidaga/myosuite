"""
===============================================================================
FILE: run.py
===============================================================================
Master script.
Set RUN_BATCH_MODE = True to process the whole folder.
Set RUN_BATCH_MODE = False to process just one file (for testing).
"""
#!/usr/bin/env python3
import logging
import os
import sys
import subprocess
# Import the new batch module
import batch_processor 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION
# ========================================

# --- MODE SELECTION ---
# True  = Process ALL files in data/kinematic/Stroke
# False = Process ONLY the single file defined below (S3_12_1)
RUN_BATCH_MODE = True 

# Base directories
BASE_DIR = "/home/abdul/Desktop/myosuite/custom_workspace"
IK_DIR = os.path.join(BASE_DIR, "IK")

# 1. BATCH SETTINGS (Used if RUN_BATCH_MODE = True)
BATCH_INPUT_DIR = os.path.join(BASE_DIR, "data/kinematic/Stroke")

# 2. SINGLE FILE SETTINGS (Used if RUN_BATCH_MODE = False)
SINGLE_INPUT_CSV = os.path.join(BATCH_INPUT_DIR, "S3_12_1.csv")

# Shared Settings
MODEL_XML = "/home/abdul/Desktop/myosuite/custom_workspace/model/myo_sim/arm/myoarm.xml"
OUTPUT_DIR = os.path.join(IK_DIR, "output")

# Script locations
SCRIPTS = {
    'TRC': os.path.join(IK_DIR, "convert_csv2trc.py"),
    'MOT': os.path.join(IK_DIR, "convert_trc2mot.py"),
    'VID': os.path.join(IK_DIR, "convert_mot2video.py")
}

DATA_RATE = 200.0
VISUALIZE_ALIGNMENT = False # Only relevant for Single Mode
# ========================================

def run_single_file():
    """Original logic for processing one file"""
    filename = os.path.basename(SINGLE_INPUT_CSV)
    base_name = os.path.splitext(filename)[0]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trc_path = os.path.join(OUTPUT_DIR, f"{base_name}.trc")
    mot_path = os.path.join(OUTPUT_DIR, f"{base_name}.mot")
    video_path = os.path.join(OUTPUT_DIR, f"{base_name}.mp4")

    logger.info(f"--- RUNNING SINGLE FILE MODE: {filename} ---")

    # 1. CSV -> TRC
    logger.info("Step 1: CSV -> TRC")
    subprocess.check_call(["python3", SCRIPTS['TRC'], SINGLE_INPUT_CSV, trc_path])

    # 2. TRC -> MOT
    logger.info("Step 2: TRC -> MOT (IK)")
    subprocess.check_call(["python3", SCRIPTS['MOT'], MODEL_XML, trc_path, mot_path])

    # 3. MOT -> VIDEO
    if not VISUALIZE_ALIGNMENT:
        logger.info("Step 3: Rendering Video")
        subprocess.check_call(["python3", SCRIPTS['VID'], MODEL_XML, mot_path, video_path])
        logger.info(f"Done! Video: {video_path}")
    else:
        logger.info("Skipping video (Visualization Enabled)")

def main():
    if RUN_BATCH_MODE:
        # Prepare config object for batch script
        config = {
            'BATCH_INPUT_DIR': BATCH_INPUT_DIR,
            'OUTPUT_DIR': OUTPUT_DIR,
            'MODEL_XML': MODEL_XML,
            'SCRIPTS': SCRIPTS
        }
        
        print("⚠️  IMPORTANT: Before running batch, ensure 'INTERACTIVE_ALIGN = False'")
        print("    inside 'convert_trc2mot.py', otherwise it will pause for every file!")
        time_to_wait = input("    Press Enter to confirm or Ctrl+C to cancel...")
        
        batch_processor.run_batch_pipeline(config)
    else:
        run_single_file()

if __name__ == "__main__":
    main()