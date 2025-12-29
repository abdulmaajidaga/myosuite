import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os
import glob
import argparse

def filter_data(input_path, output_path, cutoff, fs=200, order=4):
    # Read headers to preserve them
    with open(input_path, 'r') as f:
        header_lines = [f.readline(), f.readline()]
    
    # Read data, skipping the first two header rows
    try:
        # header=None ensures we don't use the first data row as header
        df = pd.read_csv(input_path, skiprows=2, header=None)
    except pd.errors.EmptyDataError:
        print(f"Skipping empty file: {input_path}")
        return

    # Convert to numeric, coercing errors to NaN (handles ' ', string artifacts)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Interpolate missing values
    # Linear interpolation is standard for filling small gaps in kinematic data
    df_interp = df.interpolate(method='linear', axis=0, limit_direction='both')
    
    # Check for remaining NaNs (e.g., if a column is entirely empty)
    if df_interp.isnull().values.any():
        # If columns are completely empty, fill with 0 to allow filtering, 
        # though this effectively zeros out that marker's data.
        df_interp = df_interp.fillna(0)

    data = df_interp.values
    
    # Filter design (Butterworth low-pass)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Check for valid cutoff
    if normal_cutoff >= 1:
        print(f"Error: Cutoff frequency {cutoff} is >= Nyquist frequency {nyq}. Skipping {input_path}")
        return

    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply filter
    # Pad length for filtfilt should be handled automatically, but for very short files it might warn/error.
    try:
        filtered_data = filtfilt(b, a, data, axis=0)
    except Exception as e:
        print(f"Error filtering {input_path}: {e}")
        return
    
    df_filtered = pd.DataFrame(filtered_data)
    
    # Write to file
    # First write the original headers
    with open(output_path, 'w') as f:
        f.writelines(header_lines)
    
    # Then append the dataframe data
    df_filtered.to_csv(output_path, mode='a', header=False, index=False, float_format='%.6f')
    print(f"Processed: {os.path.basename(input_path)}")

def main():
    # Determine default paths relative to this script's location
    # Script is in <root>/IK/visual/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    default_input_dir = os.path.join(project_root, "data", "kinematic", "Stroke")
    default_output_dir = os.path.join(project_root, "data", "kinematic", "Stroke", "filtered")

    parser = argparse.ArgumentParser(description="Filter kinematic data (Butterworth Low-Pass).")
    parser.add_argument("--input_dir", type=str, default=default_input_dir, help="Directory containing input CSV files.")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Directory to save filtered CSV files.")
    parser.add_argument("--cutoff", type=float, default=1.5, help="Low-pass filter cutoff frequency in Hz.")
    parser.add_argument("--fs", type=float, default=200.0, help="Sampling frequency in Hz.")
    parser.add_argument("--order", type=int, default=4, help="Filter order.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            print(f"Error creating output directory: {e}")
            return

    # Find all csv files in the input directory
    csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return

    print(f"Found {len(csv_files)} files in {args.input_dir}")
    print(f"Filtering with Cutoff={args.cutoff}Hz, Sampling Rate={args.fs}Hz, Order={args.order}")
    print(f"Output directory: {args.output_dir}")
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        output_path = os.path.join(args.output_dir, filename)
        filter_data(csv_file, output_path, args.cutoff, args.fs, args.order)

    print("Done.")

if __name__ == "__main__":
    main()
