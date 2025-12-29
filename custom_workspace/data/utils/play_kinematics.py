import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt

def process_kinematic_data(filepath):
    """
    Processes the kinematic data from the MHH dataset, handling complex
    headers and whitespace.
    """
    try:
        header_df = pd.read_csv(filepath, header=None, nrows=2, sep=',')
        marker_names = header_df.iloc[0].str.strip().ffill()
        axis_names = header_df.iloc[1].str.strip()
        multi_index = pd.MultiIndex.from_arrays([marker_names, axis_names])
        data = pd.read_csv(filepath, header=None, skiprows=2, names=multi_index, sep=',')
        return data
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return None

def filter_data(df, cutoff=10, fs=100, order=4):
    """
    Applies a low-pass Butterworth filter to all marker data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered_df = df.copy()
    for col in df.columns:
        # Interpolate to fill any NaNs before filtering
        # This is crucial for markers like ELB_M which have missing values
        series = filtered_df[col].interpolate(method='linear', limit_direction='both')
        filtered_df[col] = filtfilt(b, a, series)
        
    return filtered_df

def calculate_virtual_joints(df):
    """
    Calculates the virtual joint centers based on anatomical markers,
    correctly handling the MultiIndex DataFrame structure.
    """
    processed_df = df.copy()
    
    # --- CORRECTION ---
    # Perform calculations coordinate-wise for each virtual joint
    
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
    
    return processed_df

def animate_arm_kinematics(df):
    """
    Creates and displays an animation of the refined arm movement.
    """
    shoulder_marker, elbow_marker, wrist_marker = 'V_Shoulder', 'V_Elbow', 'V_Wrist'
    skip_frames = 10
    anim_df = df.iloc[::skip_frames, :]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    all_points = df[[shoulder_marker, elbow_marker, wrist_marker]].values.flatten()
    min_lim, max_lim = all_points.min() - 50, all_points.max() + 50
    ax.set_xlim([min_lim, max_lim]); ax.set_ylim([min_lim, max_lim]); ax.set_zlim([min_lim, max_lim])

    ax.plot(df[wrist_marker]['X'], df[wrist_marker]['Y'], df[wrist_marker]['Z'], 
            color='gray', linestyle='--', alpha=0.5, label=f'{wrist_marker} Path')

    upper_arm, = ax.plot([], [], [], color='blue', linewidth=3, marker='o', markersize=8, label='Upper Arm')
    forearm, = ax.plot([], [], [], color='orange', linewidth=3, marker='o', markersize=8, label='Forearm')
    hand_marker, = ax.plot([], [], [], color='green', marker='o', markersize=10, label='Hand (V_Wrist)')
    
    ax.set_xlabel('X Coordinate (mm)'); ax.set_ylabel('Y Coordinate (mm)'); ax.set_zlabel('Z Coordinate (mm)')
    ax.legend()
    
    # Add text to display current view angles
    angle_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def update(frame):
        shoulder_pos = anim_df[shoulder_marker].iloc[frame].values
        elbow_pos = anim_df[elbow_marker].iloc[frame].values
        wrist_pos = anim_df[wrist_marker].iloc[frame].values

        upper_arm.set_data_3d([shoulder_pos[0], elbow_pos[0]], [shoulder_pos[1], elbow_pos[1]], [shoulder_pos[2], elbow_pos[2]])
        forearm.set_data_3d([elbow_pos[0], wrist_pos[0]], [elbow_pos[1], wrist_pos[1]], [elbow_pos[2], wrist_pos[2]])
        hand_marker.set_data_3d([wrist_pos[0]], [wrist_pos[1]], [wrist_pos[2]])
        ax.set_title(f'Action 12 Arm Animation')
        
        # Update angle display with current view angles
        elev = ax.elev
        azim = ax.azim
        angle_text.set_text(f'View: elev={elev:.1f}°, azim={azim:.1f}°')
        
        return upper_arm, forearm, hand_marker, angle_text

    ani = FuncAnimation(fig, update, frames=len(anim_df), blit=True, interval=50)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Use absolute path
    file_path = '/home/abdul/Desktop/myosuite/custom_workspace/data/kinematic/Stroke/S11_12_1.csv'
    # Alternative for Healthy data:
    # file_path = '/home/abdul/Desktop/myosuite/custom_workspace/data/kinematic/Healthy/01_12_1.csv'

    
    raw_kinematic_df = process_kinematic_data(file_path)

    if raw_kinematic_df is not None:
        print("1. Successfully loaded raw kinematic data.")
        
        filtered_df = filter_data(raw_kinematic_df)
        print("2. Applied Butterworth filter to smooth the data.")
        
        refined_df = calculate_virtual_joints(filtered_df)
        print("3. Calculated virtual joint centers for Shoulder, Elbow, and Wrist.")
        
        print("\nStarting animation with refined data...")
        animate_arm_kinematics(refined_df)
