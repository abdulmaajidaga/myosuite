"""
===============================================================================
FILE: batch_processor.py
===============================================================================
"""
import os
import glob
import subprocess
import time
import pandas as pd

def run_batch_pipeline(config):
    input_dir = config['BATCH_INPUT_DIR']
    output_dir = config['OUTPUT_DIR']
    model_xml = config['MODEL_XML']
    
    script_trc = config['SCRIPTS']['TRC']
    script_mot = config['SCRIPTS']['MOT']
    script_vid = config['SCRIPTS']['VID']

    # 1. Find Files
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    
    # --- CRITICAL FIX: Move S5_12_1 to the front ---
    # This ensures the "Reference MOT" is created before other files try to read it.
    ref_file_name = "S5_12_1.csv"
    ref_path = os.path.join(input_dir, ref_file_name)
    
    if ref_path in csv_files:
        csv_files.remove(ref_path)
        csv_files.insert(0, ref_path)
        print(f"⭐ Priority: {ref_file_name} will be processed FIRST (Reference).")
    # -----------------------------------------------

    total_files = len(csv_files)
    print(f"🚀 STARTING BATCH: {total_files} files found.\n")

    results = []

    for i, csv_path in enumerate(csv_files):
        filename = os.path.basename(csv_path)
        file_id = os.path.splitext(filename)[0]
        
        trc_path = os.path.join(output_dir, f"{file_id}.trc")
        mot_path = os.path.join(output_dir, f"{file_id}.mot")
        vid_path = os.path.join(output_dir, f"{file_id}.mp4")
        
        print(f"[{i+1}/{total_files}] Processing: {filename}...")
        start_t = time.time()
        status = "OK"
        ik_error = "N/A"

        try:
            # 1. CSV -> TRC
            subprocess.check_call(["python3", script_trc, csv_path, trc_path], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            # 2. TRC -> MOT (Capture output)
            cmd_mot = ["python3", script_mot, model_xml, trc_path, mot_path]
            process = subprocess.run(cmd_mot, capture_output=True, text=True, check=True)
            
            for line in process.stdout.split('\n'):
                if "FINAL_MEAN_ERROR:" in line:
                    ik_error = line.split(":")[1].strip()
            
            # 3. MOT -> VIDEO
            subprocess.check_call(["python3", script_vid, model_xml, mot_path, vid_path], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        except subprocess.CalledProcessError as e:
            status = "FAIL"
            # --- SHOW ERROR ---
            print(f"  ❌ FAILED! Error log:")
            if e.stderr:
                print(f"     {e.stderr.strip()[-200:]}") # Print last 200 chars of error
            else:
                print(f"     (No error message returned, exit code {e.returncode})")
            # ------------------
        except Exception as e:
            status = "ERROR"
            print(f"  ❌ Error: {e}")

        duration = time.time() - start_t
        if status == "OK":
            print(f"  ✓ Done ({duration:.1f}s). IK Error: {ik_error} mm")

        results.append({
            "File": filename, "Status": status, "IK_Error_mm": ik_error, "Time": round(duration, 2)
        })

    # Summary
    df = pd.DataFrame(results)
    report_path = os.path.join(output_dir, "batch_report.csv")
    df.to_csv(report_path, index=False)
    print("\n🎉 BATCH COMPLETE")
    print(df.to_string(index=False))