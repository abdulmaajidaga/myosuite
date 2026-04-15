"""
===============================================================================
FILE: run.py
===============================================================================
Master script.
Set RUN_BATCH_MODE = True to process the whole folder.
Set RUN_BATCH_MODE = False to process just one file (for testing).
"""
#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import subprocess
from src.utils.config import get_path, get_project_root
from src.data_processing import batch_processor

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
BASE_DIR = get_project_root()

# 1. BATCH SETTINGS (Used if RUN_BATCH_MODE = True)
BATCH_INPUT_DIR = get_path("data_stroke")

# 2. SINGLE FILE SETTINGS (Used if RUN_BATCH_MODE = False)
SINGLE_INPUT_CSV = os.path.join(BATCH_INPUT_DIR, "S3_12_1.csv")

# Shared Settings
MODEL_XML = get_path("mujoco_arm_model")
OUTPUT_DIR = get_path("output_dir")

# Script locations
SCRIPTS = {
    'TRC': get_path("script_csv2trc"),
    'MOT': get_path("script_trc2mot"),
    'VID': get_path("script_mot2video")
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