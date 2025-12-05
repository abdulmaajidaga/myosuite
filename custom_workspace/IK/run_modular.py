"""
Modular IK pipeline using ik_tools package
Processes CSV → TRC → MOT → Video using clean, maintainable modules
"""
#!/usr/bin/env python3
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import myosuite
from myosuite.physics import sim_scene
import mujoco

# Import our modular IK tools
from modular import (
    process_kinematic_data, filter_data, calculate_virtual_joints,
    write_trc_file, write_mot_file,
    solve_ik_multi_site, align_mocap_to_model,
    setup_renderer, render_frame, save_video
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_data(csv_path):
    """Load MHH CSV kinematic data with multi-level headers."""
    header_df = pd.read_csv(csv_path, header=None, nrows=2, sep=',')
    marker_names = header_df.iloc[0].str.strip().ffill()
    axis_names = header_df.iloc[1].str.strip()
    multi_index = pd.MultiIndex.from_arrays([marker_names, axis_names])
    data = pd.read_csv(csv_path, header=None, skiprows=2, names=multi_index, sep=',')
    return data


def process_markers_to_virtual_joints(raw_data, data_rate=100.0):
    """Process raw marker data to compute filtered virtual joint positions."""
    # Convert DataFrame to dict of arrays for easier processing
    marker_dict = {}
    for marker in raw_data.columns.levels[0]:
        marker_dict[marker] = raw_data[marker][['X', 'Y', 'Z']].values
    
    # Filter each marker trajectory
    filtered_markers = {}
    for name, positions in marker_dict.items():
        filtered_markers[name] = filter_data(positions, fs=data_rate, axis=0)
    
    # Calculate virtual joints
    virtual_joints = calculate_virtual_joints(filtered_markers)
    
    return virtual_joints


def run_ik_on_trajectory(sim, virtual_joints, data_rate=100.0):
    """Run IK solver on entire trajectory of virtual joint positions."""
    num_frames = len(next(iter(virtual_joints.values())))
    all_joint_names = [sim.model.joint(i).name for i in range(sim.model.njnt)]

    # === Align mocap to model using the same method as run_ik_visualise.py ===
    logger.info("Calculating optimal alignment (matching run_ik_visualise.py)...")
    model_s_pos = sim.data.site_xpos[sim.model.site('V_Shoulder').id].copy()
    model_e_pos = sim.data.site_xpos[sim.model.site('V_Elbow').id].copy()
    model_vec = model_e_pos - model_s_pos

    # Use first frame to compute mocap shoulder/elbow positions (virtual_joints in mm)
    mocap_s_raw = virtual_joints['V_Shoulder'][0]
    mocap_e_raw = virtual_joints['V_Elbow'][0]
    mocap_s = np.array([mocap_s_raw[0], -mocap_s_raw[2], mocap_s_raw[1]]) / 1000.0
    mocap_e = np.array([mocap_e_raw[0], -mocap_e_raw[2], mocap_e_raw[1]]) / 1000.0
    mocap_vec = mocap_e - mocap_s

    rotation, _ = Rotation.align_vectors(a=[model_vec], b=[mocap_vec])
    logger.info("✓ Optimal rotation calculated")

    # Build processed trajectories in model coordinates (same transform as run_ik_visualise)
    processed_trajectories = {}
    mocap_shoulder_origin = virtual_joints['V_Shoulder'][0]
    for name, arr in virtual_joints.items():
        # arr is in mm; subtract shoulder origin (mm), convert to meters and remap axes
        relative_pos_m = (arr - mocap_shoulder_origin) / 1000.0
        processed_trajectories[name] = np.array([model_s_pos + rotation.apply([p[0], -p[2], p[1]]) for p in relative_pos_m])
    logger.info("✓ Mocap data aligned to model coordinate system")

    # Run IK for each frame (first frame uses increased iterations)
    joint_trajectory = []
    logger.info("Solving inverse kinematics for all frames...")

    # First frame with higher iterations (warm start)
    first_targets = {name: processed_trajectories[name][0] for name in processed_trajectories}
    first_result = solve_ik_multi_site(sim, first_targets, max_steps=1000)
    if first_result is None:
        logger.error("Initial IK solve failed on first frame")
        # fall back to current qpos
        joint_trajectory.append(sim.data.qpos.copy())
    else:
        sim.data.qpos[:] = first_result.qpos
        mujoco.mj_forward(sim.model.ptr, sim.data.ptr)
        joint_trajectory.append(first_result.qpos)
        logger.info(f"  > Initial solve completed (Error: {first_result.err_norm*1000:.2f} mm)")

    for i in tqdm(range(1, num_frames), desc="IK Solving"):
        targets = {name: processed_trajectories[name][i] for name in processed_trajectories}
        result = solve_ik_multi_site(sim, targets)
        if result is None:
            logger.warning(f"Frame {i}: IK returned None; using previous qpos")
            joint_trajectory.append(joint_trajectory[-1])
        else:
            joint_trajectory.append(result.qpos)
            sim.data.qpos[:] = result.qpos
            mujoco.mj_forward(sim.model.ptr, sim.data.ptr)

        if i % 100 == 0 or i == num_frames - 1:
            logger.info(f"  > Solved frame {i+1}/{num_frames}")

    logger.info("✓ IK solving complete")
    return np.array(joint_trajectory), all_joint_names


def render_video_from_trajectory(sim, joint_trajectory, output_path):
    """Render video from joint angle trajectory."""
    logger.info("Setting up renderer...")
    renderer, camera = setup_renderer(
        sim.model.ptr,
        camera_cfg={
            'azimuth': 0,
            'distance': 2.5,
            'elevation': -20,
            'lookat': [-0.1, 0.0, 1.4]
        }
    )
    
    frames = []
    logger.info("Rendering frames...")
    for qpos in tqdm(joint_trajectory, desc="Rendering"):
        sim.data.qpos[:] = qpos
        mujoco.mj_forward(sim.model.ptr, sim.data.ptr)
        frame = render_frame(renderer, sim.data.ptr, camera)
        frames.append(frame)
    
    renderer.close()
    
    logger.info(f"Saving video to {output_path}...")
    # Use skvideo.io.vwrite directly to match mot_playback.py (no explicit FPS = default ~25 FPS)
    import skvideo.io
    skvideo.io.vwrite(output_path, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
    logger.info("✓ Video saved")


def main():
    # === CONFIGURATION ===
    BASE_DIR = "/home/abdul/Desktop/myosuite/custom_workspace"
    IK_DIR = os.path.join(BASE_DIR, "IK")
    
    PATHS = {
        'csv': os.path.join(BASE_DIR, "data/kinematic/Stroke/S5_12_1.csv"),
        'model': "/home/abdul/Desktop/myosuite/custom_workspace/model/myo_sim/arm/myoarm.xml",
        'output_dir': os.path.join(IK_DIR, "output")
    }
    
    SETTINGS = {
        'data_rate': 200.0,  # Hz
        'create_video': True,
        'save_trc': True,  # Optionally save intermediate TRC file
    }
    
    # Setup output directory and paths
    os.makedirs(PATHS['output_dir'], exist_ok=True)
    base_name = Path(PATHS['csv']).stem
    
    trc_path = os.path.join(PATHS['output_dir'], f"{base_name}.trc")
    mot_path = os.path.join(PATHS['output_dir'], f"{base_name}.mot")
    video_path = os.path.join(PATHS['output_dir'], f"{base_name}.mp4")
    
    # === STEP 1: Load and process CSV data ===
    logger.info("=== Step 1: Loading and processing CSV data ===")
    raw_data = load_csv_data(PATHS['csv'])
    logger.info(f"✓ Loaded {len(raw_data)} frames from CSV")
    
    # === STEP 2: Calculate virtual joints ===
    logger.info("=== Step 2: Computing virtual joint positions ===")
    virtual_joints = process_markers_to_virtual_joints(raw_data, SETTINGS['data_rate'])
    logger.info(f"✓ Computed {len(virtual_joints)} virtual joints")
    
    # Optionally save TRC file
    if SETTINGS['save_trc']:
        logger.info(f"Saving TRC file to {trc_path}...")
        write_trc_file(trc_path, virtual_joints, data_rate=SETTINGS['data_rate'])
        logger.info("✓ TRC file saved")
    
    # === STEP 3: Load MuJoCo model ===
    logger.info("=== Step 3: Loading MuJoCo model ===")
    sim_wrapper = sim_scene.SimScene.get_sim(PATHS['model'])
    sim = sim_wrapper.sim
    logger.info(f"✓ Model loaded from {PATHS['model']}")
    
    # === STEP 4: Run IK ===
    logger.info("=== Step 4: Running inverse kinematics ===")
    joint_trajectory, joint_names = run_ik_on_trajectory(
        sim, virtual_joints, SETTINGS['data_rate']
    )
    
    # === STEP 5: Save MOT file ===
    logger.info("=== Step 5: Saving MOT file ===")
    time = np.arange(len(joint_trajectory)) / SETTINGS['data_rate']
    write_mot_file(mot_path, joint_trajectory, joint_names, time)
    logger.info(f"✓ MOT file saved to {mot_path}")
    
    # === STEP 6: Create video ===
    if SETTINGS['create_video']:
        logger.info("=== Step 6: Creating video visualization ===")
        render_video_from_trajectory(sim, joint_trajectory, video_path)
    
    # === DONE ===
    logger.info("\n" + "="*60)
    logger.info("Pipeline complete! Output files:")
    if SETTINGS['save_trc']:
        logger.info(f"  - TRC:   {trc_path}")
    logger.info(f"  - MOT:   {mot_path}")
    if SETTINGS['create_video']:
        logger.info(f"  - Video: {video_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
