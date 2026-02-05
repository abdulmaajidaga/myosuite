"""
Interactive Motion Viewer V2
- Input FMA score via slider or text
- Generate and visualize motion in real-time
- Pause/Play/Reset controls
- Now includes trunk compensation visualization
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
import joblib
import math

# --- Configuration ---
BASE_DIR = "/home/abdul/Desktop/myosuite/custom_workspace"
MODEL_PATH = os.path.join(BASE_DIR, "IK/cutoff/models/cvae_cutoff_fma_best.pth")
SCALER_PATH = os.path.join(BASE_DIR, "IK/cutoff/models/scaler_cutoff_fma.pkl")
REFERENCE_PATH = os.path.join(BASE_DIR, "data/kinematic/cutoff/augmented/01_12_1_FMA66.csv")

INPUT_DIM = 15       # 12 arm + 3 trunk
CONDITION_DIM = 1
HIDDEN_DIM = 256
LATENT_DIM = 32
NUM_HEADS = 4
SEQ_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
COLS = ARM_COLS + TRUNK_COLS


# --- Model (must match training exactly) ---
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.layer_norm(x + self.out_proj(context))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM + CONDITION_DIM, HIDDEN_DIM // 2,
                           num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.attention = SelfAttention(HIDDEN_DIM, NUM_HEADS)
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x, c):
        c_expanded = c.unsqueeze(1).repeat(1, x.size(1), 1)
        inputs = torch.cat([x, c_expanded], dim=2)
        lstm_out, _ = self.lstm(inputs)
        attn_out = self.attention(lstm_out)
        pooled = attn_out.mean(dim=1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.layer_norm(x + residual)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_start = nn.Linear(LATENT_DIM + CONDITION_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM // 2,
                           num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.res_block1 = ResidualBlock(HIDDEN_DIM)
        self.res_block2 = ResidualBlock(HIDDEN_DIM)
        self.fc_out = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, z, c, seq_len):
        latent_input = torch.cat([z, c], dim=1)
        hidden_start = F.gelu(self.fc_start(latent_input))
        lstm_input = hidden_start.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.lstm(lstm_input)
        output = self.res_block1(output)
        output = self.res_block2(output)
        return self.fc_out(output)


class MotionCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def inference(self, c, seq_len=SEQ_LEN, guidance_scale=2.0):
        """Generate with classifier-free guidance for stronger conditioning."""
        with torch.no_grad():
            z = torch.randn(c.size(0), LATENT_DIM).to(c.device)

            if guidance_scale == 1.0:
                return self.decoder(z, c, seq_len)

            # Classifier-free guidance
            c_null = torch.zeros_like(c)
            out_cond = self.decoder(z, c, seq_len)
            out_uncond = self.decoder(z, c_null, seq_len)

            # Guided output
            return out_uncond + guidance_scale * (out_cond - out_uncond)


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
