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
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "kinematic")

def find_processed_file():
    # Look in Healthy/processed first
    pattern = os.path.join(DATA_DIR, "*", "processed", "*_processed.csv")
    files = glob.glob(pattern)
    
    if not files:
        return None
    return files[0] # Return the first one found

def animate_skeleton(file_path):
    print(f"Visualizing: {file_path}")
    df = pd.read_csv(file_path)
    data = df.values # (N, 12)
    
    # Columns: Sh_x, Sh_y, Sh_z (0-2), El (3-5), Wr (6-8), WrVec (9-11)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Processed Skeleton (Chest at Origin)")
    
    # Set limits
    # Chest is at 0,0,0. Arm length is roughly 600-800mm.
    limit = 800
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    
    # Chest Point (Fixed)
    ax.scatter([0], [0], [0], color='black', s=50, label='Chest (Origin)')
    
    # Lines
    arm_line, = ax.plot([], [], [], 'bo-', lw=4, label='Arm')
    vec_line, = ax.plot([], [], [], 'r-', lw=2, label='Wrist Vector')
    
    def update(i):
        # Frame i
        row = data[i]
        
        # Joints relative to chest
        sh = row[0:3]
        el = row[3:6]
        wr = row[6:9]
        
        # Wrist Vector (Direction, not point)
        # To visualize, we draw a line starting at Wrist and ending at Wrist + Vector
        # Scale vector for visibility? It's usually small (width of wrist).
        wr_vec = row[9:12]
        vec_end = wr + wr_vec 
        
        # Arm: Chest -> Sh -> El -> Wr
        # Wait, Chest is 0,0,0. But Sh is normalized relative to Chest.
        # So Sh coordinate IS the vector from Chest to Shoulder.
        # Skeleton chain: Chest(0) -> Sh -> El -> Wr
        
        chain_x = [0, sh[0], el[0], wr[0]]
        chain_y = [0, sh[1], el[1], wr[1]]
        chain_z = [0, sh[2], el[2], wr[2]]
        
        arm_line.set_data_3d(chain_x, chain_y, chain_z)
        
        # Vector Line
        vec_x = [wr[0], vec_end[0]]
        vec_y = [wr[1], vec_end[1]]
        vec_z = [wr[2], vec_end[2]]
        
        vec_line.set_data_3d(vec_x, vec_y, vec_z)
        
        return arm_line, vec_line
    
    ani = FuncAnimation(fig, update, frames=len(data), interval=20, blit=False)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    f = find_processed_file()
    if f is None:
        print("No processed files found. Run IK/preprocess_markers.py first.")
    else:
        animate_skeleton(f)