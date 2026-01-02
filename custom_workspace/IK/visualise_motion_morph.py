import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# --- Configuration ---
# Set the path to the CSV file you want to view
FILE_TO_VIEW = "custom_workspace/data/kinematic/augmented_smooth/S14_12_1_to_22_12_4_processed/FMA_65.csv"  # <--- CHANGE THIS PATH

def play_motion(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    print(f"Loading motion from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Setup the 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate bounds so the camera doesn't jump around
    # We look at all X, Y, Z columns to find the global min/max
    all_x = pd.concat([df['Sh_x'], df['El_x'], df['Wr_x']])
    all_y = pd.concat([df['Sh_y'], df['El_y'], df['Wr_y']])
    all_z = pd.concat([df['Sh_z'], df['El_z'], df['Wr_z']])
    
    # Add a little padding
    pad = 0.1
    x_lim = (all_x.min() - pad, all_x.max() + pad)
    y_lim = (all_y.min() - pad, all_y.max() + pad)
    z_lim = (all_z.min() - pad, all_z.max() + pad)

    # Initialize the lines and points
    # Bones: Shoulder-to-Elbow, Elbow-to-Wrist
    line_arm, = ax.plot([], [], [], 'o-', lw=4, color='blue', label='Arm Skeleton')
    
    # Wrist Trajectory Trace (shows the path history)
    line_trace, = ax.plot([], [], [], '-', lw=1, color='orange', alpha=0.5, label='Wrist Path')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Motion Viewer: {os.path.basename(csv_path)}")
    ax.legend()

    # Animation Update Function
    def update(frame):
        # Current row data
        row = df.iloc[frame]

        # Extract coordinates for this frame
        sh = [row['Sh_x'], row['Sh_y'], row['Sh_z']]
        el = [row['El_x'], row['El_y'], row['El_z']]
        wr = [row['Wr_x'], row['Wr_y'], row['Wr_z']]

        # X, Y, Z lists for the skeleton line (Sh -> El -> Wr)
        xs = [sh[0], el[0], wr[0]]
        ys = [sh[1], el[1], wr[1]]
        zs = [sh[2], el[2], wr[2]]

        # Update Skeleton
        line_arm.set_data(xs, ys)
        line_arm.set_3d_properties(zs)

        # Update Wrist Trace (History up to current frame)
        history = df.iloc[:frame+1]
        line_trace.set_data(history['Wr_x'], history['Wr_y'])
        line_trace.set_3d_properties(history['Wr_z'])

        # Keep camera fixed
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        
        return line_arm, line_trace

    # Create Animation
    # interval=20 means 20ms per frame (approx 50 fps)
    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=20, blit=False)

    plt.show()

if __name__ == "__main__":
    # You can also run this from terminal: python visualize_motion.py path/to/file.csv
    if len(sys.argv) > 1:
        play_motion(sys.argv[1])
    else:
        play_motion(FILE_TO_VIEW)