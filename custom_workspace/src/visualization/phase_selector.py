import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema
import os
import sys
import glob
import math
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path, get_project_root

# --- Data Loading & Processing (Matching play_kinematics.py) ---

def load_and_process_data(filepath):
    """
    Reads the CSV with 2-row header, creating a MultiIndex DataFrame.
    Matches play_kinematics.py process_kinematic_data.
    """
    try:
        header_df = pd.read_csv(filepath, header=None, nrows=2, sep=',')
        marker_names = header_df.iloc[0].str.strip().ffill()
        axis_names = header_df.iloc[1].str.strip()
        multi_index = pd.MultiIndex.from_arrays([marker_names, axis_names])
        
        # Read the data
        data = pd.read_csv(filepath, header=None, skiprows=2, names=multi_index, sep=',')
        
        # Interpolate like play_kinematics
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.interpolate(method='linear', limit_direction='both').fillna(0)
        
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def calculate_virtual_joints(df):
    """
    Calculates virtual joints for animation (Shoulder, Elbow, Wrist).
    """
    processed_df = df.copy()
    
    required_cols = ['WRA', 'WRB', 'ELB_L', 'ELB_M', 'SA_1', 'SA_2', 'SA_3']
    for col in required_cols:
        if col not in df.columns:
            pass 

    try:
        # Virtual Wrist Center: Midpoint between WRA and WRB
        processed_df[('V_Wrist', 'X')] = (processed_df['WRA']['X'] + processed_df['WRB']['X']) / 2
        processed_df[('V_Wrist', 'Y')] = (processed_df['WRA']['Y'] + processed_df['WRB']['Y']) / 2
        processed_df[('V_Wrist', 'Z')] = (processed_df['WRA']['Z'] + processed_df['WRB']['Z']) / 2
        
        # Virtual Elbow Center: Midpoint between ELB_L and ELB_M
        processed_df[('V_Elbow', 'X')] = (processed_df['ELB_L']['X'] + processed_df['ELB_M']['X']) / 2
        processed_df[('V_Elbow', 'Y')] = (processed_df['ELB_L']['Y'] + processed_df['ELB_M']['Y']) / 2
        processed_df[('V_Elbow', 'Z')] = (processed_df['ELB_L']['Z'] + processed_df['ELB_M']['Z']) / 2

        # Virtual Shoulder Center: Centroid of the three shoulder markers
        processed_df[('V_Shoulder', 'X')] = (processed_df['SA_1']['X'] + processed_df['SA_2']['X'] + processed_df['SA_3']['X']) / 3
        processed_df[('V_Shoulder', 'Y')] = (processed_df['SA_1']['Y'] + processed_df['SA_2']['Y'] + processed_df['SA_3']['Y']) / 3
        processed_df[('V_Shoulder', 'Z')] = (processed_df['SA_1']['Z'] + processed_df['SA_2']['Z'] + processed_df['SA_3']['Z']) / 3
        
    except KeyError as e:
        print(f"Missing expected columns for virtual joints calculation: {e}")
        
    return processed_df

def calculate_velocity_midpoint(df, fs=200.0):
    """
    Calculates velocity of the Wrist Midpoint.
    """
    if ('V_Wrist', 'X') not in df.columns:
        try:
            wra = df['WRA'].values
            wrb = df['WRB'].values
            midpoint = (wra + wrb) / 2.0
        except KeyError:
            return np.zeros(len(df))
    else:
        midpoint = df['V_Wrist'].values

    d_pos = np.diff(midpoint, axis=0)
    dist = np.linalg.norm(d_pos, axis=1)
    velocity = dist / (1.0 / fs)
    velocity = np.insert(velocity, 0, 0)
    return velocity

def find_minima(velocity, order=10):
    minima_indices = argrelextrema(velocity, np.less, order=order)[0]
    return minima_indices

# --- Interactive Visualizer Class ---

class PhaseSelectorApp:
    def __init__(self, df, filename, fs=200.0, saved_indices=None):
        self.df = df
        self.filename = filename
        self.fs = fs
        
        # Calculate Virtual Joints
        self.anim_df = calculate_virtual_joints(df)
        
        # Calculate Velocity
        self.velocity = calculate_velocity_midpoint(self.anim_df, fs)
        self.time = np.arange(len(self.velocity)) / fs
        
        # Animation params
        self.skip_frames = 10
        self.interval = 50
        self.anim_indices = np.arange(0, len(df), self.skip_frames)
        
        # State
        self.selected_indices = saved_indices if saved_indices else []
        self.done = False
        
        # Setup Figure
        self.fig = plt.figure(figsize=(16, 8))
        
        # Subplot 1: 3D Animation
        self.ax_anim = self.fig.add_subplot(121, projection='3d')
        self.setup_3d_plot()
        
        # Subplot 2: Velocity Graph
        self.ax_vel = self.fig.add_subplot(122)
        self.setup_velocity_plot()
        
        # Connect Events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Start Animation
        self.anim = FuncAnimation(self.fig, self.update_anim, frames=len(self.anim_indices), 
                                  interval=self.interval, blit=False)
        
        plt.suptitle(f"Processing: {filename}\nKeys: [Enter] Finish/Next, [d] Delete Last Point", fontsize=14)
        plt.show()

    def setup_3d_plot(self):
        self.ax_anim.set_xlabel('X Coordinate (mm)')
        self.ax_anim.set_ylabel('Y Coordinate (mm)')
        self.ax_anim.set_zlabel('Z Coordinate (mm)')
        
        shoulder_marker, elbow_marker, wrist_marker = 'V_Shoulder', 'V_Elbow', 'V_Wrist'
        
        if (wrist_marker, 'X') not in self.anim_df:
             return 

        all_points = self.anim_df[[shoulder_marker, elbow_marker, wrist_marker]].values.flatten()
        if len(all_points) > 0:
            min_lim, max_lim = all_points.min() - 50, all_points.max() + 50
            self.ax_anim.set_xlim([min_lim, max_lim])
            self.ax_anim.set_ylim([min_lim, max_lim])
            self.ax_anim.set_zlim([min_lim, max_lim])

        self.ax_anim.plot(self.anim_df[wrist_marker]['X'], 
                          self.anim_df[wrist_marker]['Y'], 
                          self.anim_df[wrist_marker]['Z'], 
                          color='gray', linestyle='--', alpha=0.5, label='V_Wrist Path')

        self.line_upper, = self.ax_anim.plot([], [], [], color='blue', linewidth=3, marker='o', markersize=8, label='Upper Arm')
        self.line_fore, = self.ax_anim.plot([], [], [], color='orange', linewidth=3, marker='o', markersize=8, label='Forearm')
        self.line_hand, = self.ax_anim.plot([], [], [], color='green', marker='o', markersize=10, label='Hand (V_Wrist)')
        
        self.ax_anim.legend(loc='upper left', fontsize='small')
        
        self.angle_text = self.ax_anim.text2D(0.02, 0.98, '', transform=self.ax_anim.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def setup_velocity_plot(self):
        self.ax_vel.plot(self.time, self.velocity, 'b-', label='Velocity')
        
        minima_idxs = find_minima(self.velocity, order=5)
        self.ax_vel.plot(self.time[minima_idxs], self.velocity[minima_idxs], 'ro', alpha=0.3, markersize=4)
        
        # --- CHANGED: Updated Title to 5 Points ---
        self.ax_vel.set_title("Click to select 5 Keypoints: (1)Rest -> (2)Pick -> (3)Drink -> (4)Place -> (5)Rest")
        self.ax_vel.set_xlabel("Time (s)")
        self.ax_vel.set_ylabel("Velocity (mm/s)")
        
        self.time_line = self.ax_vel.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        
        self.selection_lines = []
        for idx in self.selected_indices:
            line = self.ax_vel.axvline(x=idx/self.fs, color='red', linestyle='--', linewidth=2)
            self.selection_lines.append(line)

    def update_anim(self, frame_num):
        real_idx = self.anim_indices[frame_num]
        
        try:
            shoulder = self.anim_df['V_Shoulder'].iloc[real_idx]
            elbow = self.anim_df['V_Elbow'].iloc[real_idx]
            wrist = self.anim_df['V_Wrist'].iloc[real_idx]
            
            self.line_upper.set_data_3d(
                [shoulder['X'], elbow['X']], 
                [shoulder['Y'], elbow['Y']], 
                [shoulder['Z'], elbow['Z']]
            )
            self.line_fore.set_data_3d(
                [elbow['X'], wrist['X']], 
                [elbow['Y'], wrist['Y']], 
                [elbow['Z'], wrist['Z']]
            )
            self.line_hand.set_data_3d(
                [wrist['X']], [wrist['Y']], [wrist['Z']]
            )
            
            elev = self.ax_anim.elev
            azim = self.ax_anim.azim
            self.angle_text.set_text(f'View: elev={elev:.1f}°, azim={azim:.1f}°')
            
        except KeyError:
            pass 

        current_time = real_idx / self.fs
        self.time_line.set_xdata([current_time])
        
        return self.line_upper, self.line_fore, self.line_hand, self.time_line, self.angle_text

    def on_click(self, event):
        if event.inaxes != self.ax_vel:
            return
            
        if event.button == 1:
            # --- CHANGED: Check for 5 points instead of 3 ---
            if len(self.selected_indices) >= 5:
                print("Max 5 points selected. Press 'd' to delete or Enter to finish.")
                return
                
            click_time = event.xdata
            click_idx = int(click_time * self.fs)
            
            minima_idxs = find_minima(self.velocity, order=5)
            if len(minima_idxs) > 0:
                dists = np.abs(minima_idxs - click_idx)
                nearest = minima_idxs[np.argmin(dists)]
                if abs(nearest - click_idx) < 100: 
                    click_idx = nearest
            
            self.selected_indices.append(int(click_idx))
            self.selected_indices.sort()
            self.redraw_selections()
            
        elif event.button == 3:
            if self.selected_indices:
                self.selected_indices.pop()
                self.redraw_selections()

    def on_key(self, event):
        if event.key == 'd':
            if self.selected_indices:
                self.selected_indices.pop()
                self.redraw_selections()
        elif event.key == 'enter':
            # --- CHANGED: Check for 5 points instead of 3 ---
            if len(self.selected_indices) < 5:
                print(f"Warning: Only {len(self.selected_indices)} points selected. Expecting 5 (Rest, Pick, Drink, Place, Rest).")
            self.done = True
            plt.close(self.fig)

    def redraw_selections(self):
        for line in self.selection_lines:
            line.remove()
        self.selection_lines = []
        
        for idx in self.selected_indices:
            line = self.ax_vel.axvline(x=idx/self.fs, color='red', linestyle='--', linewidth=2)
            self.selection_lines.append(line)
        
        self.fig.canvas.draw_idle()

# --- Main Execution ---

def main():
    project_root = get_project_root()

    parser = argparse.ArgumentParser(description="Interactive Phase Selector.")
    parser.add_argument("--dataset", type=str, choices=['healthy', 'stroke'], default='stroke', help="Dataset to process.")
    args = parser.parse_args()

    if args.dataset == 'healthy':
        data_dir = get_path("data_healthy")
        visual_out_dir = os.path.join(get_path("output_analysis"), "healthy")
        if not os.path.exists(visual_out_dir):
            os.makedirs(visual_out_dir)
        json_path = os.path.join(get_path("output_phase_indices"), "manual_phase_indices.json")
        output_image_path = os.path.join(visual_out_dir, "velocity_phases_comparison_manual.png")
    else:
        data_dir = get_path("data_stroke")
        json_path = os.path.join(get_path("output_phase_indices"), "stroke_manual_phase_indices.json")
        output_image_path = os.path.join(get_path("output_analysis"), "velocity_phases_comparison_manual.png")
    
    fs_value = 200.0
    
    saved_data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                saved_data = json.load(f)
            except json.JSONDecodeError:
                pass
    
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    # csv_files = csv_files[:25] # Uncomment to limit

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    print("----------------------------------------------------------------")
    print(f"Interactive Phase Selector ({args.dataset.upper()} dataset)")
    print("----------------------------------------------------------------")
    print(f"Saving progress to: {json_path}")
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # --- CHANGED: Check for 5 points instead of 3 ---
        if filename in saved_data and len(saved_data[filename]) == 5:
             print(f"Skipping {filename} (already done).")
             continue
        
        print(f"Opening {filename}...")
        
        df = load_and_process_data(file_path)
        if df is None:
            continue
            
        current_saved = saved_data.get(filename, [])
        app = PhaseSelectorApp(df, filename, fs=fs_value, saved_indices=current_saved)
        
        if app.selected_indices:
            saved_data[filename] = app.selected_indices
            with open(json_path, 'w') as f:
                json.dump(saved_data, f, indent=4)
        
    print("\nAll files processed. Generating final summary image...")
    generate_summary_image(csv_files, data_dir, saved_data, output_image_path, fs_value)

def generate_summary_image(csv_files, original_dir, saved_data, output_path, fs):
    num_files = len(csv_files)
    cols = 5
    rows = math.ceil(num_files / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15), constrained_layout=True)
    axes = axes.flatten()
    
    for i, file_path in enumerate(csv_files):
        filename = os.path.basename(file_path)
        ax = axes[i]
        
        try:
            df_filt = load_and_process_data(file_path)
            if df_filt is not None:
                anim_df = calculate_virtual_joints(df_filt)
                vel_filt = calculate_velocity_midpoint(anim_df, fs)
            else:
                vel_filt = np.zeros(100)
            
            original_path = os.path.join(original_dir, filename)
            if os.path.exists(original_path):
                df_orig = load_and_process_data(original_path)
                if df_orig is not None:
                    anim_orig = calculate_virtual_joints(df_orig)
                    vel_orig = calculate_velocity_midpoint(anim_orig, fs)
                else:
                    vel_orig = vel_filt
            else:
                vel_orig = vel_filt
            
            time = np.arange(len(vel_filt)) / fs
            
            ax.plot(time, vel_orig, color='lightblue', linewidth=2.5, alpha=0.8)
            ax.plot(time, vel_filt, color='blue', linewidth=1.0, alpha=0.9)
            
            idxs = saved_data.get(filename, [])
            for idx in idxs:
                ax.axvline(x=idx/fs, color='red', linestyle='--', linewidth=1.5)
            
            ax.set_title(filename, fontsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, "Error", ha='center')
            
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle("Manual Phase Divisions Summary", fontsize=16)
    plt.savefig(output_path, dpi=150)
    print(f"Summary saved to {output_path}")

if __name__ == "__main__":
    main()