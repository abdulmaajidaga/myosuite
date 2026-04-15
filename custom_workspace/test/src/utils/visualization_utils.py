"""
Visualization utilities for MuJoCo rendering
"""
import os
import mujoco
import numpy as np
from typing import Optional, List, Tuple
import skvideo.io

def setup_renderer(model: mujoco.MjModel,
                  width: int = 640,
                  height: int = 480,
                  camera_cfg: Optional[dict] = None) -> Tuple[mujoco.Renderer, mujoco.MjvCamera]:
    """Setup MuJoCo renderer and camera."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    camera = mujoco.MjvCamera()
    if camera_cfg is None:
        camera_cfg = {
            'azimuth': 0,
            'distance': 2.5,
            'elevation': -20,
            'lookat': [-0.1, 0.0, 1.4]
        }
    
    for k, v in camera_cfg.items():
        setattr(camera, k, v)
        
    return renderer, camera

def render_frame(renderer: mujoco.Renderer,
                data: mujoco.MjData,
                camera: mujoco.MjvCamera) -> np.ndarray:
    """Render a single frame."""
    renderer.update_scene(data, camera=camera)
    return renderer.render()

def save_video(frames: List[np.ndarray],
               output_path: str,
               fps: int = 30,
               codec: str = 'libx264',
               pixfmt: str = 'yuv420p') -> None:
    """Save frames to video file using skvideo."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = skvideo.io.FFmpegWriter(
        output_path,
        inputdict={'-r': str(fps)},
        outputdict={
            '-vcodec': codec,
            '-pix_fmt': pixfmt,
            '-r': str(fps)
        }
    )
    
    try:
        for frame in frames:
            writer.writeFrame(frame)
    finally:
        writer.close()