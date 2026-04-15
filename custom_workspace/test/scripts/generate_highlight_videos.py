import os
import sys
import subprocess

# BASE_DIR is project-root/test/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CONDA_PYTHON = "/home/abdul/miniconda3/envs/MyoSuite/bin/python"

STAGES = [0, 1, 2, 3]
AUGMENTS = ["dtw", "smote", "linear"]
HIGHLIGHT_FMAS = [16, 66]

def main():
    analysis_root = os.path.join(BASE_DIR, "output/analysis")
    
    for stage in STAGES:
        for aug in AUGMENTS:
            stage_dir = os.path.join(analysis_root, f"stage{stage}_{aug}")
            if not os.path.exists(stage_dir):
                print(f"Skipping missing: {stage_dir}")
                continue
                
            print(f"\n>>> Generating highlight videos for Stage {stage} + {aug.upper()}")
            
            for fma in HIGHLIGHT_FMAS:
                csv_name = f"FMA_{fma}.csv"
                # Use the existing pipeline script but ONLY for this file and WITH videos enabled
                # We point it to the specific file to save time
                cmd = [
                    CONDA_PYTHON, "test/scripts/run_generated_pipeline.py",
                    csv_name,
                    "--input-dir", os.path.join(stage_dir, "csv"),
                    "--output-dir", stage_dir
                    # Not skipping video this time
                ]
                
                print(f"    Processing FMA {fma}...")
                subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL)
                
    print("\n" + "="*60)
    print("HIGHLIGHT VIDEOS GENERATED")
    print("Check the 'videos' folder in each stage directory.")
    print("="*60)

if __name__ == "__main__":
    main()
