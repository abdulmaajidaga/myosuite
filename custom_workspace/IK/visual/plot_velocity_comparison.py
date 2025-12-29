import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import os
import glob
import math

def calculate_velocity(df, fs=200.0):
    # Calculate velocity magnitude of the midpoint between WRA and WRB
    
    # Extract X, Y, Z for WRA (columns 0, 1, 2) and WRB (columns 3, 4, 5)
    wra = df.iloc[:, 0:3].values
    wrb = df.iloc[:, 3:6].values
    
    # Calculate midpoint
    midpoint = (wra + wrb) / 2.0
    
    # Calculate difference between consecutive points
    d_pos = np.diff(midpoint, axis=0)
    
    # Calculate Euclidean distance for each step
    dist = np.linalg.norm(d_pos, axis=1)
    
    # Velocity = distance / time_step
    dt = 1.0 / fs
    velocity = dist / dt
    
    # Pad with 0 at the beginning to match length of df
    velocity = np.insert(velocity, 0, 0)
    
    return velocity

def find_minima(velocity, order=50):
    # Find local minima
    minima_indices = argrelextrema(velocity, np.less, order=order)[0]
    
    # We expect exactly 3 main stops (phases: Reach->Stop->Lift->Stop->Place->Stop->Rest)
    # Filter minima that are likely just noise (too high velocity) if needed, 
    # but primarily we want the 3 lowest velocity points.
    
    # Get values at these indices
    minima_values = velocity[minima_indices]
    
    # Sort by value (velocity), take the 3 lowest
    if len(minima_indices) > 3:
        # Get indices of the 3 smallest values
        sorted_by_val_idx = np.argsort(minima_values)[:3]
        best_indices = minima_indices[sorted_by_val_idx]
        # Sort them back by time so lines are in order
        minima_indices = np.sort(best_indices)
    
    return minima_indices

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    original_dir = os.path.join(project_root, "data", "kinematic", "Stroke")
    filtered_dir = os.path.join(project_root, "data", "kinematic", "Stroke", "filtered")
    output_image_path = os.path.join(script_dir, "velocity_phases_comparison.png")
    
    # Note: To see a bigger difference between Original and Filtered, 
    # you might need to adjust the 'cutoff' frequency in 'IK/visual/filter_csv.py'.
    # Decreasing the cutoff (e.g., from 6.0 to 3.0) will make the filtered line smoother.
    # The 'fs' variable below is the Sampling Frequency of the data.
    fs_value = 200.0
    
    # Get list of CSV files in filtered directory (assuming they exist in original too)
    csv_files = sorted(glob.glob(os.path.join(filtered_dir, "*.csv")))
    
    # Limit to 25 files if there are more, or handle fewer
    csv_files = csv_files[:25]
    
    if not csv_files:
        print("No CSV files found.")
        return

    num_files = len(csv_files)
    cols = 5
    rows = math.ceil(num_files / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15), constrained_layout=True)
    axes = axes.flatten()
    
    for i, filtered_path in enumerate(csv_files):
        filename = os.path.basename(filtered_path)
        original_path = os.path.join(original_dir, filename)
        
        ax = axes[i]
        
        if not os.path.exists(original_path):
            ax.text(0.5, 0.5, f"Original file not found:\n{filename}", ha='center', va='center')
            continue
            
        try:
            # Read Data
            # Skip first 2 lines of header
            df_filt = pd.read_csv(filtered_path, skiprows=2, header=None)
            df_orig = pd.read_csv(original_path, skiprows=2, header=None)
            
            # Ensure numeric
            df_filt = df_filt.apply(pd.to_numeric, errors='coerce').fillna(0)
            df_orig = df_orig.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Calculate Velocity
            vel_filt = calculate_velocity(df_filt, fs=fs_value)
            vel_orig = calculate_velocity(df_orig, fs=fs_value)
            
            # Find Minima in Filtered Data
            # We want exactly 3 lines if possible
            minima_idxs = find_minima(vel_filt, order=50) 
            
            # Plot
            time = np.arange(len(vel_filt)) / fs_value
            
            # Make Original thicker and lighter to be visible behind Filtered
            ax.plot(time, vel_orig, color='lightblue', linewidth=2.5, alpha=0.8, label='Original')
            # Make Filtered thinner and darker
            ax.plot(time, vel_filt, color='blue', linewidth=1.0, alpha=0.9, label='Filtered')
            
            # Plot vertical lines for minima
            for idx in minima_idxs:
                ax.axvline(x=idx/fs_value, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
            
            ax.set_title(filename, fontsize=8)
            
            # Only add legend to the first plot to reduce clutter
            if i == 0:
                ax.legend(fontsize=6)
                
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', fontsize=6)
            print(f"Error processing {filename}: {e}")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle("Velocity Profile Comparison (Original vs Filtered) with Phase Divisions", fontsize=16)
    plt.savefig(output_image_path, dpi=150)
    print(f"Visualization saved to {output_image_path}")

if __name__ == "__main__":
    main()
