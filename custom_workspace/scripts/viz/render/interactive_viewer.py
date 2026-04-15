"""
Interactive CVAE motion viewer — live generation via FMA slider.

What it does:
  Loads the trained CVAE and scaler, then renders a matplotlib animation of the
  generated 15-DOF arm trajectory in real time. A slider/text box lets you dial
  any FMA score (0–66) and immediately see the corresponding motion.

Input:
  - models/cvae/cvae_cutoff_fma_best.pth  (D_base model, config key: cvae_model_best)
  - models/cvae/scaler_cutoff_fma.pkl     (config key: cvae_scaler)
  - data/kinematic/cutoff/processed/      (healthy reference for rest-pose offset)

Output:
  - Interactive matplotlib window (no file saved)
  - Pause/Play/Reset controls; Trunk Compensation panel alongside 3D arm view

Usage:
  python scripts/viz/render/interactive_viewer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import os
import sys
import joblib
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src.utils.config import get_path

# --- Configuration ---
MODEL_PATH = get_path("cvae_model_best")
SCALER_PATH = get_path("cvae_scaler")
REFERENCE_PATH = get_path("reference_healthy_csv")

from src.generation.model import MotionCVAE, INPUT_DIM, CONDITION_DIM, HIDDEN_DIM, LATENT_DIM, NUM_HEADS, SEQ_LEN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
COLS = ARM_COLS + TRUNK_COLS


class InteractiveViewer:
    def __init__(self):
        print("Loading model...")
        self.model = MotionCVAE().to(DEVICE)

        # Try best model first
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            model_path = model_path.replace('_best', '')

        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        self.scaler = joblib.load(SCALER_PATH)
        self.ref_pose = self._load_reference()

        # State
        self.fma_score = 50
        self.guidance_scale = 2.0  # Classifier-free guidance strength
        self.motion_data = None
        self.current_frame = 0
        self.playing = True
        self.wrist_history = []

        # Setup figure first
        self.setup_figure()

        # Generate initial motion
        self.generate_motion()
        self._update_2d_plot()

        # Start animation
        self.ani = FuncAnimation(self.fig, self.update, frames=self._frame_generator,
                                 interval=30, blit=False, cache_frame_data=False)
        plt.show()

    def _load_reference(self):
        # Average starting pose across all subjects in training data
        # This converts delta outputs back to absolute positions for visualization
        return {
            'Sh_x': -77.3, 'Sh_y': 643.0, 'Sh_z': 302.7,
            'El_x': -188.2, 'El_y': 474.4, 'El_z': 41.3,
            'Wr_x': -88.7, 'Wr_y': 241.2, 'Wr_z': 41.1,
            'WrVec_x': -37.0, 'WrVec_y': 12.0, 'WrVec_z': -33.1,
            'Trunk_x': 0.0, 'Trunk_y': 0.0, 'Trunk_z': 0.0,
        }

    def generate_motion(self):
        """Generate motion for current FMA score with classifier-free guidance."""
        c = torch.FloatTensor([[self.fma_score / 66.0]]).to(DEVICE)
        gen = self.model.inference(c, SEQ_LEN, guidance_scale=self.guidance_scale).squeeze(0).cpu().numpy()
        gen = self.scaler.inverse_transform(gen)

        # Add reference pose for all columns
        for i, col in enumerate(COLS):
            if i < gen.shape[1]:
                gen[:, i] += self.ref_pose.get(col, 0)

        # Smooth
        nyq = 0.5 * 100
        b, a = signal.butter(2, min(6 / nyq, 0.99), btype='low')
        for i in range(gen.shape[1]):
            gen[:, i] = signal.filtfilt(b, a, gen[:, i])

        self.motion_data = gen
        self.current_frame = 0
        self.wrist_history = []

        # Update plot limits if figure exists
        if hasattr(self, 'ax_3d'):
            self._update_limits()
            self._update_2d_plot()

    def _update_limits(self):
        if self.motion_data is None:
            return

        all_x = np.concatenate([self.motion_data[:, 0], self.motion_data[:, 3], self.motion_data[:, 6]])
        all_y = np.concatenate([self.motion_data[:, 1], self.motion_data[:, 4], self.motion_data[:, 7]])
        all_z = np.concatenate([self.motion_data[:, 2], self.motion_data[:, 5], self.motion_data[:, 8]])

        margin = 50
        self.ax_3d.set_xlim([all_x.min() - margin, all_x.max() + margin])
        self.ax_3d.set_ylim([all_y.min() - margin, all_y.max() + margin])
        self.ax_3d.set_zlim([all_z.min() - margin, all_z.max() + margin])

    def setup_figure(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("Interactive Motion Viewer")

        # 3D animation subplot
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_3d.set_xlabel('X (mm)')
        self.ax_3d.set_ylabel('Y (mm)')
        self.ax_3d.set_zlabel('Z (mm)')

        # Arm lines
        self.line_upper, = self.ax_3d.plot([], [], [], 'b-', lw=4, marker='o', markersize=10)
        self.line_fore, = self.ax_3d.plot([], [], [], 'orange', lw=4, marker='o', markersize=10)
        self.line_wrist_vec, = self.ax_3d.plot([], [], [], 'g-', lw=2)
        self.line_trace, = self.ax_3d.plot([], [], [], 'c-', lw=1, alpha=0.5)

        # 2D trajectory subplot
        self.ax_2d = self.fig.add_subplot(122)
        self.ax_2d.set_xlabel('Y (Reach) mm')
        self.ax_2d.set_ylabel('Z (Height) mm')
        self.ax_2d.set_title('Wrist Trajectory (Y-Z plane)')
        self.ax_2d.grid(True, alpha=0.3)

        # Static trajectory line (will be updated on generate)
        self.line_traj_full, = self.ax_2d.plot([], [], 'b-', lw=1, alpha=0.3, label='Full path')
        self.line_traj_current, = self.ax_2d.plot([], [], 'r-', lw=2, label='Current')
        self.point_current, = self.ax_2d.plot([], [], 'ro', markersize=10)
        self.ax_2d.legend(loc='upper right')

        # Title
        self.title = self.ax_3d.set_title(f'FMA Score: {self.fma_score}  |  Frame: 0/{SEQ_LEN}')

        # Controls
        plt.subplots_adjust(bottom=0.25)

        # FMA Slider
        ax_slider = plt.axes([0.15, 0.12, 0.55, 0.03])
        self.slider = Slider(ax_slider, 'FMA Score', 0, 66, valinit=self.fma_score, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        # Frame slider
        ax_frame = plt.axes([0.15, 0.06, 0.55, 0.03])
        self.frame_slider = Slider(ax_frame, 'Frame', 0, SEQ_LEN-1, valinit=0, valstep=1)
        self.frame_slider.on_changed(self.on_frame_change)

        # Buttons
        ax_play = plt.axes([0.75, 0.12, 0.08, 0.04])
        self.btn_play = Button(ax_play, 'Pause')
        self.btn_play.on_clicked(self.toggle_play)

        ax_reset = plt.axes([0.85, 0.12, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)

        ax_gen = plt.axes([0.75, 0.06, 0.18, 0.04])
        self.btn_gen = Button(ax_gen, 'New Sample')
        self.btn_gen.on_clicked(self.new_sample)

        # Info text
        self.info_text = self.fig.text(0.02, 0.02, '', fontsize=9, fontfamily='monospace')

        self._update_limits()
        self._update_2d_plot()

    def _update_2d_plot(self):
        if self.motion_data is None:
            return

        wr_y = self.motion_data[:, 7]
        wr_z = self.motion_data[:, 8]

        self.line_traj_full.set_data(wr_y, wr_z)
        self.ax_2d.set_xlim([wr_y.min() - 20, wr_y.max() + 20])
        self.ax_2d.set_ylim([wr_z.min() - 20, wr_z.max() + 20])

    def _frame_generator(self):
        while True:
            yield self.current_frame

    def update(self, frame):
        if self.motion_data is None:
            return

        f = self.current_frame
        m = self.motion_data

        # Get joint positions
        sh = m[f, 0:3]
        el = m[f, 3:6]
        wr = m[f, 6:9]
        wv = m[f, 9:12]

        # Update 3D arm
        self.line_upper.set_data_3d([sh[0], el[0]], [sh[1], el[1]], [sh[2], el[2]])
        self.line_fore.set_data_3d([el[0], wr[0]], [el[1], wr[1]], [el[2], wr[2]])
        self.line_wrist_vec.set_data_3d([wr[0], wr[0]+wv[0]], [wr[1], wr[1]+wv[1]], [wr[2], wr[2]+wv[2]])

        # Update trace
        self.wrist_history.append(wr.copy())
        if len(self.wrist_history) > 1:
            hist = np.array(self.wrist_history)
            self.line_trace.set_data_3d(hist[:, 0], hist[:, 1], hist[:, 2])

        # Update 2D
        self.line_traj_current.set_data(m[:f+1, 7], m[:f+1, 8])
        self.point_current.set_data([wr[1]], [wr[2]])

        # Update title
        self.ax_3d.set_title(f'FMA Score: {self.fma_score}  |  Frame: {f}/{SEQ_LEN-1}')

        # Update frame slider (without triggering callback)
        self.frame_slider.eventson = False
        self.frame_slider.set_val(f)
        self.frame_slider.eventson = True

        # Update info (including trunk if available)
        upper_arm = np.linalg.norm(el - sh)
        forearm = np.linalg.norm(wr - el)
        info_str = f'Upper Arm: {upper_arm:.0f}mm | Forearm: {forearm:.0f}mm | Wrist Y: {wr[1]:.0f}mm Z: {wr[2]:.0f}mm'

        # Add trunk info if available
        if m.shape[1] >= 15:
            trunk = m[f, 12:15]
            trunk_disp = np.linalg.norm(trunk - m[0, 12:15])
            info_str += f' | Trunk: {trunk_disp:.1f}mm'

        self.info_text.set_text(info_str)

        # Advance frame if playing
        if self.playing:
            self.current_frame = (self.current_frame + 1) % SEQ_LEN
            if self.current_frame == 0:
                self.wrist_history = []

    def on_slider_change(self, val):
        new_fma = int(val)
        if new_fma != self.fma_score:
            self.fma_score = new_fma
            self.generate_motion()
            self._update_2d_plot()

    def on_frame_change(self, val):
        self.current_frame = int(val)
        self.wrist_history = []

    def toggle_play(self, event):
        self.playing = not self.playing
        self.btn_play.label.set_text('Play' if not self.playing else 'Pause')

    def reset(self, event):
        self.current_frame = 0
        self.wrist_history = []

    def new_sample(self, event):
        self.generate_motion()
        self._update_2d_plot()


if __name__ == "__main__":
    viewer = InteractiveViewer()
