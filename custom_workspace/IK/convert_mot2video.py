import os
import mujoco
import numpy as np
import pandas as pd
import skvideo.io
from tqdm import tqdm

# ========================================
# CONFIGURATION - Edit these paths
# ========================================
MODEL_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/model/myo_sim/arm/myoarm.xml'  # Path to MuJoCo model XML
MOT_PATH = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/S5_12_1.mot'  # Path to input MOT file
OUTPUT_VIDEO = '/home/abdul/Desktop/myosuite/custom_workspace/IK/output/S5_12_1.mp4'  # Path to output video file

# Camera settings (adjust as needed)
CAMERA_AZIMUTH = 0
CAMERA_DISTANCE = 2.5
CAMERA_ELEVATION = -20
CAMERA_LOOKAT = [-0.1, 0.0, 1.4]  # Centered on the arm

# Video resolution
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
# ========================================

def read_mot_file(filepath):
    """
    Reads a .mot file and returns a pandas DataFrame.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return None
    
    # Find the line number where the header ends
    skiprows = 0
    with open(filepath, "r") as file:
        for line in file:
            if "endheader" in line:
                break
            skiprows += 1
    
    # Read the data, skipping the header
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=skiprows + 1)
    return df

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

# --- Load the .mot file and the MuJoCo model ---
motion_df = read_mot_file(MOT_PATH)
if motion_df is None:
    exit()

mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
mj_data = mujoco.MjData(mj_model)

# --- Map joints between the model and the .mot file ---
model_joint_names = [mj_model.joint(j).name for j in range(mj_model.njnt)]
motion_joint_names = [col for col in motion_df.columns if col in model_joint_names]
print(f"Found {len(motion_joint_names)} matching joints to animate.")

missing_joints = set(motion_df.columns) - set(model_joint_names) - {'time'}
if missing_joints:
    print(f"WARNING: Joints in .mot file but not in model: {missing_joints}")

# --- Set up the renderer and camera ---
renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
camera = mujoco.MjvCamera()
camera.azimuth = CAMERA_AZIMUTH
camera.distance = CAMERA_DISTANCE
camera.elevation = CAMERA_ELEVATION
camera.lookat = CAMERA_LOOKAT

# --- Render the video frame by frame ---
frames = []
print("Rendering video...")
for t in tqdm(range(len(motion_df))):
    # Set the qpos for each joint
    for joint_name in motion_joint_names:
        joint_qpos_addr = mj_model.joint(joint_name).qposadr[0]
        # The .mot file from our IK is in radians, so no conversion is needed
        mj_data.qpos[joint_qpos_addr] = motion_df[joint_name].iloc[t]
    
    # Step the simulation forward
    mujoco.mj_forward(mj_model, mj_data)
    
    # Render the frame
    renderer.update_scene(mj_data, camera=camera)
    frame = renderer.render()
    frames.append(frame)

renderer.close()

# --- Save the video ---
# Calculate the actual frame rate from the MOT file
time_values = motion_df['time'].values
if len(time_values) > 1:
    avg_dt = np.mean(np.diff(time_values))
    frame_rate = int(1.0 / avg_dt)
else:
    frame_rate = 200  # Default fallback

print(f"Video frame rate: {frame_rate} FPS (from MOT time stamps)")

skvideo.io.vwrite(OUTPUT_VIDEO, np.asarray(frames), 
                  inputdict={'-r': str(frame_rate)},
                  outputdict={'-pix_fmt': 'yuv420p', '-r': str(frame_rate)})

print(f"\n✓ Visualization complete. Video saved to: {OUTPUT_VIDEO}")
