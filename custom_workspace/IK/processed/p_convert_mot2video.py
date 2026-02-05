import os
import glob
import mujoco
import numpy as np
import pandas as pd
import skvideo.io
from tqdm import tqdm
import sys

# ========================================
# CONFIGURATION
# ========================================
MODEL_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/model/myo_sim/arm/myoarm.xml'

# *** INPUT SELECTION ***
# OPTION A (Batch): Path to a FOLDER containing .mot files
# OPTION B (Single): Path to a single .mot FILE
# INPUT_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/mot/p_originals' 
# INPUT_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/mot/p_originals/S5_12_1.mot' # <--- Uncomment for single file
INPUT_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/mot/augmented'
# Output Directory (Videos will be saved here)
# OUTPUT_DIR = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/videos/p_originals'
OUTPUT_DIR = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/videos/augmented'
# Camera Settings (Fixed for consistency)
CAMERA_AZIMUTH = 90      # Side view usually shows flexion best
CAMERA_DISTANCE = 1.5    # Zoom in a bit more than 2.5
CAMERA_ELEVATION = -30
CAMERA_LOOKAT = [0.0, -0.2, 1.2] # Adjusted slightly for typical arm height

# Video Resolution
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
# ========================================

def read_mot_file(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return None
    
    skiprows = 0
    try:
        with open(filepath, "r") as file:
            for line in file:
                if "endheader" in line:
                    break
                skiprows += 1
        
        df = pd.read_csv(filepath, sep=r'\s+', skiprows=skiprows + 1)
        return df
    except Exception as e:
        print(f"Error reading MOT {os.path.basename(filepath)}: {e}")
        return None

def render_video(mot_path, output_path, model, renderer, camera):
    motion_df = read_mot_file(mot_path)
    if motion_df is None or motion_df.empty:
        return False

    data = mujoco.MjData(model)
    model_joint_names = [model.joint(j).name for j in range(model.njnt)]
    motion_joint_names = [col for col in motion_df.columns if col in model_joint_names]
    
    frames = []
    
    # Render Loop
    for t in range(len(motion_df)):
        for joint_name in motion_joint_names:
            jid = model.joint(joint_name).id
            qpos_addr = model.jnt_qposadr[jid]
            data.qpos[qpos_addr] = motion_df[joint_name].iloc[t]
        
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

    # Calculate FPS from time column
    time_values = motion_df['time'].values
    if len(time_values) > 1:
        avg_dt = np.mean(np.diff(time_values))
        frame_rate = int(1.0 / avg_dt)
        if frame_rate <= 0: frame_rate = 30
    else:
        frame_rate = 200

    try:
        skvideo.io.vwrite(output_path, np.asarray(frames), 
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-pix_fmt': 'yuv420p', '-r': str(frame_rate), '-crf': '20'})
        return True
    except Exception as e:
        print(f"Error saving video {output_path}: {e}")
        return False

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine Input Mode
    files_to_process = []
    if os.path.isdir(INPUT_PATH):
        print(f"--- BATCH MODE: Processing folder {INPUT_PATH} ---")
        files_to_process = glob.glob(os.path.join(INPUT_PATH, "*.mot"))
    elif os.path.isfile(INPUT_PATH):
        print(f"--- SINGLE MODE: Processing file {os.path.basename(INPUT_PATH)} ---")
        files_to_process = [INPUT_PATH]
    else:
        print(f"Error: INPUT_PATH does not exist: {INPUT_PATH}")
        return

    if not files_to_process:
        print("No .mot files found.")
        return

    # Init MuJoCo
    print("Loading MuJoCo Model...")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    
    camera = mujoco.MjvCamera()
    camera.azimuth = CAMERA_AZIMUTH
    camera.distance = CAMERA_DISTANCE
    camera.elevation = CAMERA_ELEVATION
    camera.lookat = CAMERA_LOOKAT

    # Run
    success_count = 0
    for mot_file in tqdm(files_to_process, desc="Rendering Videos"):
        base_name = os.path.basename(mot_file).replace('.mot', '.mp4')
        out_file = os.path.join(OUTPUT_DIR, base_name)
        
        if render_video(mot_file, out_file, model, renderer, camera):
            success_count += 1
            
    renderer.close()
    print(f"\nDone! Rendered {success_count}/{len(files_to_process)} videos.")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()