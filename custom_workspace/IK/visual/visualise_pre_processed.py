import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import glob
import sys

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "kinematic")

def animate_all_in_one():
    # Find files
    healthy_files = glob.glob(os.path.join(DATA_DIR, "Healthy", "processed", "*_processed.csv"))
    stroke_files = glob.glob(os.path.join(DATA_DIR, "Stroke", "processed", "*_processed.csv"))
    
    print(f"Found {len(healthy_files)} Healthy and {len(stroke_files)} Stroke files.")
    
    # Load Data
    healthy_data = []
    stroke_data = []
    
    # Load Healthy
    for f in healthy_files:
        try:
            df = pd.read_csv(f)
            healthy_data.append(df.values)
        except: pass
        
    # Load Stroke
    for f in stroke_files:
        try:
            df = pd.read_csv(f)
            stroke_data.append(df.values)
        except: pass
        
    if not healthy_data and not stroke_data:
        print("No data found.")
        return

    # Setup Plot
    fig = plt.figure(figsize=(16, 8))
    
    # Left Plot: Healthy
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"Healthy Skeletons ({len(healthy_data)})")
    
    # Right Plot: Stroke
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"Stroke Skeletons ({len(stroke_data)})")
    
    limit = 800
    
    # Setup axes properties
    for ax in [ax1, ax2]:
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        # Chest Point
        ax.scatter([0], [0], [0], color='black', s=50, label='Chest (Origin)')
    
    # Initialize Lines
    lines_h = []
    for _ in healthy_data:
        ln, = ax1.plot([], [], [], color='green', alpha=0.4, lw=1) 
        lines_h.append(ln)
        
    lines_s = []
    for _ in stroke_data:
        ln, = ax2.plot([], [], [], color='red', alpha=0.4, lw=1) 
        lines_s.append(ln)
        
    # Update Function
    def update(frame_idx):
        # Update Healthy
        for i, data in enumerate(healthy_data):
            if frame_idx >= len(data): idx = len(data) - 1
            else: idx = frame_idx
            
            row = data[idx]
            sh, el, wr = row[0:3], row[3:6], row[6:9]
            xs = [0, sh[0], el[0], wr[0]]
            ys = [0, sh[1], el[1], wr[1]]
            zs = [0, sh[2], el[2], wr[2]]
            lines_h[i].set_data_3d(xs, ys, zs)
            
        # Update Stroke
        for i, data in enumerate(stroke_data):
            if frame_idx >= len(data): idx = len(data) - 1
            else: idx = frame_idx
            
            row = data[idx]
            sh, el, wr = row[0:3], row[3:6], row[6:9]
            xs = [0, sh[0], el[0], wr[0]]
            ys = [0, sh[1], el[1], wr[1]]
            zs = [0, sh[2], el[2], wr[2]]
            lines_s[i].set_data_3d(xs, ys, zs)
            
        return lines_h + lines_s

    # Create Animation
    ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
    
    plt.show()

if __name__ == "__main__":
    animate_all_in_one()