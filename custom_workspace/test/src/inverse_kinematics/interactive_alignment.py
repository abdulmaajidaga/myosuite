"""
interactive_alignment.py
------------------------
Modified for VISIBILITY.
1. Shoulder marker is locked to Robot Shoulder.
2. Shoulder marker is MADE LARGER (6cm) so it is visible over the robot mesh.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation

class InteractiveAligner:
    def __init__(self, sim, trc_data, initial_rotation=None, initial_offset=None):
        self.sim = sim
        self.trc_data = trc_data
        self.marker_names = trc_data.get_marker_names()
        
        # Default to identity/zeros if nothing provided
        self.current_rot = initial_rotation if initial_rotation else Rotation.identity()
        self.pos_offset = np.zeros(3) # Forced to zero for locking
        
        # Internal tracking for Euler adjustments
        self.euler_adjustment = np.zeros(3) # Roll, Pitch, Yaw
        
        # Playback state
        self.paused = False
        self.frame_idx = 0
        self.num_frames = trc_data.get_num_frames()
        self.dt = 1.0 / trc_data.get_data_rate()
        
        # Visualization settings
        self.colors = [
            [1, 0, 0, 1.0], # Red
            [0, 1, 0, 0.7], # Green
            [0, 0, 1, 0.7], # Blue
        ]

        # Get Robot Shoulder (Anchor)
        try:
            sid = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'V_Shoulder')
            self.model_shoulder_pos = sim.data.site_xpos[sid].copy()
        except:
            print("Warning: V_Shoulder site not found in model.")
            self.model_shoulder_pos = np.zeros(3)

    def get_aligned_trajectories(self, frame_idx):
        """Calculates marker positions for a specific frame."""
        
        user_rot = Rotation.from_euler('xyz', self.euler_adjustment, degrees=True)
        total_rot = self.current_rot * user_rot
        
        # DYNAMIC ANCHOR: Get the shoulder position for THIS frame
        current_shoulder_origin = self.trc_data.get_marker_data('V_Shoulder')[frame_idx]
        
        positions = {}
        for name in self.marker_names:
            raw_pos = self.trc_data.get_marker_data(name)[frame_idx]
            
            # Center data around CURRENT shoulder
            rel_pos = (raw_pos - current_shoulder_origin) / 1000.0
            
            # Coordinate Swap (Standard Mocap Z-up -> Sim Y-up)
            vec_to_rotate = np.array([rel_pos[0], -rel_pos[2], rel_pos[1]])
            
            # Apply Rotation
            rotated_pos = total_rot.apply(vec_to_rotate)
            
            # Add to Robot Shoulder (Anchor)
            final_pos = self.model_shoulder_pos + rotated_pos
            positions[name] = final_pos
            
        return positions

    def on_key(self, keycode):
        """Handle keyboard input."""
        step_rot = 2.0     # 2 degrees
        
        # Rotation Controls
        if keycode == ord('Q'): self.euler_adjustment[0] += step_rot 
        elif keycode == ord('A'): self.euler_adjustment[0] -= step_rot 
        elif keycode == ord('W'): self.euler_adjustment[1] += step_rot 
        elif keycode == ord('S'): self.euler_adjustment[1] -= step_rot 
        elif keycode == ord('E'): self.euler_adjustment[2] += step_rot 
        elif keycode == ord('D'): self.euler_adjustment[2] -= step_rot 
        
        # Reset
        elif keycode == ord('R'): self.euler_adjustment = np.zeros(3)
        
        elif keycode == 32: # Spacebar
            self.paused = not self.paused
            
        print(f"\r[Shoulder Locked] Rot: {self.euler_adjustment}", end="")

    def run(self):
        print("========================================================")
        print(" SHOULDER-LOCKED ALIGNMENT")
        print(" The Red Ball is now LARGE (6cm) to ensure visibility.")
        print(" It acts as the pivot point inside the robot shoulder.")
        print("========================================================")

        with mujoco.viewer.launch_passive(self.sim.model.ptr, self.sim.data.ptr, key_callback=self.on_key) as viewer:
            # Init markers
            viewer.user_scn.ngeom = len(self.marker_names)
            for i, name in enumerate(self.marker_names):
                
                # SPECIAL HANDLING FOR SHOULDER VISIBILITY
                is_shoulder = 'Shoulder' in name
                
                # Make Shoulder RED, LARGE, and TRANSPARENT
                if is_shoulder:
                    color = [1, 0, 0, 0.5] # 50% transparent red
                    size = 0.04            # 6cm radius (Huge)
                else:
                    color = [0, 1, 0, 0.7] # Green standard
                    size = 0.02            # 2cm standard
                
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[size, 0, 0],
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=color
                )

            while viewer.is_running():
                if not self.paused:
                    self.frame_idx = (self.frame_idx + 1) % self.num_frames
                
                aligned_data = self.get_aligned_trajectories(self.frame_idx)
                
                for i, name in enumerate(self.marker_names):
                    viewer.user_scn.geoms[i].pos = aligned_data[name]
                
                viewer.sync()
                time.sleep(self.dt)

        final_rot = self.current_rot * Rotation.from_euler('xyz', self.euler_adjustment, degrees=True)
        return final_rot, np.zeros(3)

def run_interactive_alignment(sim, trc_data, auto_rot=None):
    aligner = InteractiveAligner(sim, trc_data, initial_rotation=auto_rot)
    return aligner.run()