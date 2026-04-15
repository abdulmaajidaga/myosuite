"""
MOT → MP4 renderer with optional per-muscle activation coloring.

What it does:
  Renders a MuJoCo musculoskeletal model stepping through joint angles from a
  .mot file. In activation mode each tendon is colored on a blue→red gradient
  proportional to its activation level each frame.

Input:
  - model.xml        : MuJoCo arm model (e.g. models/model/myo_sim/arm/myoarm.xml)
  - file.mot         : joint-angle trajectory (tab-separated, time column first)
  - output.mp4       : destination video path
  - --activations    : (optional) activations.csv from output/generated/id/<FMA>/
                       columns = muscle names, rows = frames

Output:
  - output.mp4       : rendered video at 640×480, same frame count as .mot

Usage:
  python scripts/viz/render/convert_mot2video.py model.xml file.mot output.mp4
  python scripts/viz/render/convert_mot2video.py model.xml file.mot output.mp4 --activations activations.csv
"""
import os
import sys
import mujoco
import numpy as np
import pandas as pd
import skvideo.io
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src.utils.config import get_path

# ========================================
# CONFIGURATION
# ========================================
MODEL_PATH = get_path("mujoco_arm_model")
MOT_PATH = os.path.join(get_path("output_dir"), "gen_score_15.mot")
OUTPUT_VIDEO = os.path.join(get_path("output_dir"), "gen_score_15.mp4")

# Override from CLI
if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]
    MOT_PATH = sys.argv[2]
    OUTPUT_VIDEO = sys.argv[3]

# Camera settings
CAMERA_AZIMUTH = 0
CAMERA_DISTANCE = 2.5
CAMERA_ELEVATION = -20
CAMERA_LOOKAT = [-0.1, 0.0, 1.4]

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
# ========================================


def read_mot_file(filepath):
    """Reads a .mot file and returns a pandas DataFrame."""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return None
    skiprows = 0
    with open(filepath, "r") as file:
        for line in file:
            if "endheader" in line:
                break
            skiprows += 1
    return pd.read_csv(filepath, sep=r'\s+', skiprows=skiprows + 1)


def _build_muscle_tendon_map(model, muscle_names):
    """
    Map muscle actuator names to their tendon IDs for coloring.

    In this model, muscles are spatial tendons (not geoms). Each actuator
    transmits force through a tendon (e.g., DELT1 -> DELT1_tendon).
    We color tendons via model.tendon_rgba.
    """
    tendon_map = {}  # muscle_name -> tendon_id
    for act_name in muscle_names:
        tendon_name = f"{act_name}_tendon"
        for t in range(model.ntendon):
            if model.tendon(t).name == tendon_name:
                tendon_map[act_name] = t
                break
    return tendon_map


def render_muscle_video(mot_path, model_path, output_path, activations_path):
    """
    Render video with muscle activation coloring (blue=inactive, red=active).

    Reads joint angles from .mot file and muscle activations from activations.csv.
    Colors each muscle's tendon using linear interpolation:
        color = (1 - activation) * BLUE + activation * RED

    Also overlays time stamp and top-5 active muscles on each frame.
    """
    # Load motion and activations
    motion_df = read_mot_file(mot_path)
    if motion_df is None:
        return
    activations = pd.read_csv(activations_path)

    # Sync frame counts
    n_frames = min(len(motion_df), len(activations))
    motion_df = motion_df.iloc[:n_frames]
    activations = activations.iloc[:n_frames]

    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Map joints
    model_joint_names = [model.joint(j).name for j in range(model.njnt)]
    motion_joint_names = [c for c in motion_df.columns if c in model_joint_names]

    # Map muscle names to tendon IDs
    muscle_names = [c for c in activations.columns if c != 'time']
    tendon_map = _build_muscle_tendon_map(model, muscle_names)
    print(f"   Mapped {len(tendon_map)}/{len(muscle_names)} muscles to tendons")

    # Colors: RGBA with high alpha for visibility
    BLUE = np.array([0.2, 0.3, 1.0, 0.8])    # inactive muscle
    RED = np.array([1.0, 0.1, 0.0, 0.9])      # fully active muscle
    GRAY = np.array([0.5, 0.5, 0.5, 0.15])    # unmapped tendons (faint gray)

    # Set ALL tendons to faint gray first (hides finger tendons that have
    # no activations), then color mapped arm tendons blue
    for tid in range(model.ntendon):
        model.tendon_rgba[tid] = GRAY
    for tid in tendon_map.values():
        model.tendon_rgba[tid] = BLUE
        model.tendon_width[tid] = 0.008  # thick for visibility

    # Renderer + camera
    renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    camera = mujoco.MjvCamera()
    camera.azimuth = CAMERA_AZIMUTH
    camera.distance = CAMERA_DISTANCE
    camera.elevation = CAMERA_ELEVATION
    camera.lookat = CAMERA_LOOKAT

    # Enable tendon visualization in scene options
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True

    # FPS from MOT timestamps
    time_values = motion_df['time'].values
    dt = np.mean(np.diff(time_values)) if len(time_values) > 1 else 0.005
    fps = 1.0 / dt

    # Check for PIL (optional text overlay)
    try:
        from PIL import Image, ImageDraw, ImageFont
        has_pil = True
        try:
            font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            font_lg = ImageFont.load_default()
            font_sm = ImageFont.load_default()
    except ImportError:
        has_pil = False

    # Render loop
    frames = []
    print(f"   Rendering {n_frames} frames with muscle activation coloring...")
    for t in tqdm(range(n_frames), leave=False):
        # Set joint angles
        for jname in motion_joint_names:
            qaddr = model.joint(jname).qposadr[0]
            data.qpos[qaddr] = motion_df[jname].iloc[t]

        mujoco.mj_forward(model, data)

        # Color tendons by activation level
        act_row = activations.iloc[t]
        for mname, tid in tendon_map.items():
            a = float(np.clip(act_row[mname], 0, 1))
            model.tendon_rgba[tid] = (1.0 - a) * BLUE + a * RED

        # Render with tendon visualization enabled
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        frame = renderer.render()

        # Text overlay (time + top-5 muscles)
        if has_pil:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)

            # Time stamp
            t_val = motion_df['time'].iloc[t]
            draw.text((10, 10), f"Time: {t_val:.2f}s  Frame: {t+1}/{n_frames}",
                      fill=(255, 255, 255), font=font_lg)

            # Top 5 active muscles
            act_vals = activations[muscle_names].iloc[t].values
            top5 = np.argsort(act_vals)[-5:][::-1]
            y = 50
            draw.text((10, y), "Top 5 Active Muscles:", fill=(255, 255, 0), font=font_sm)
            for idx in top5:
                y += 20
                draw.text((10, y), f"{muscle_names[idx]}: {act_vals[idx]:.3f}",
                          fill=(255, 255, 255), font=font_sm)

            frame = np.array(img)

        frames.append(frame)

    renderer.close()

    # Save video
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fps_str = str(int(fps))
    skvideo.io.vwrite(output_path, np.asarray(frames),
                      inputdict={'-r': fps_str},
                      outputdict={'-pix_fmt': 'yuv420p', '-r': fps_str})
    print(f"   Muscle activation video saved: {output_path}")


def main():
    """Basic skeleton video (no muscle coloring)."""
    os.makedirs(os.path.dirname(OUTPUT_VIDEO) or '.', exist_ok=True)

    motion_df = read_mot_file(MOT_PATH)
    if motion_df is None:
        exit()

    mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    mj_data = mujoco.MjData(mj_model)

    model_joint_names = [mj_model.joint(j).name for j in range(mj_model.njnt)]
    motion_joint_names = [col for col in motion_df.columns if col in model_joint_names]
    print(f"Found {len(motion_joint_names)} matching joints to animate.")

    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    camera = mujoco.MjvCamera()
    camera.azimuth = CAMERA_AZIMUTH
    camera.distance = CAMERA_DISTANCE
    camera.elevation = CAMERA_ELEVATION
    camera.lookat = CAMERA_LOOKAT

    frames = []
    print("Rendering video...")
    for t in tqdm(range(len(motion_df))):
        for joint_name in motion_joint_names:
            joint_qpos_addr = mj_model.joint(joint_name).qposadr[0]
            mj_data.qpos[joint_qpos_addr] = motion_df[joint_name].iloc[t]

        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=camera)
        frame = renderer.render()
        frames.append(frame)

    renderer.close()

    time_values = motion_df['time'].values
    if len(time_values) > 1:
        avg_dt = np.mean(np.diff(time_values))
        frame_rate = int(1.0 / avg_dt)
    else:
        frame_rate = 200

    print(f"Video frame rate: {frame_rate} FPS")
    skvideo.io.vwrite(OUTPUT_VIDEO, np.asarray(frames),
                      inputdict={'-r': str(frame_rate)},
                      outputdict={'-pix_fmt': 'yuv420p', '-r': str(frame_rate)})
    print(f"Video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    # Check for --activations flag
    if '--activations' in sys.argv:
        idx = sys.argv.index('--activations')
        act_path = sys.argv[idx + 1]
        render_muscle_video(MOT_PATH, MODEL_PATH, OUTPUT_VIDEO, act_path)
    else:
        main()
