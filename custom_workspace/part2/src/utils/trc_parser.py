import numpy as np
import re

class TRCParser:
    def __init__(self, trc_filepath):
        self.filepath = trc_filepath
        self.marker_names = []
        self.data_rate = 0
        self.num_frames = 0
        self.time = []
        self.marker_data = {}
        self._parse_file()

    def _parse_file(self):
        with open(self.filepath, 'r') as f:
            # Skip first line (PathFileType)
            f.readline()

            # Read header labels and values
            header_labels = f.readline().strip().split('\t')
            header_values = f.readline().strip().split('\t')
            header_dict = dict(zip(header_labels, header_values))

            # Assign values from the parsed header
            self.data_rate = int(float(header_dict.get('DataRate', 0)))
            self.num_frames = int(float(header_dict.get('NumFrames', 0)))

            # Read marker names line
            marker_line = f.readline().strip()
            raw_names = re.split(r'\t+', marker_line)
            self.marker_names = [name for name in raw_names if name and name not in ['Frame#', 'Time']]

            # Skip coordinate line (X1, Y1, Z1...) and blank line
            f.readline()
            f.readline()

            # Read the rest of the data lines
            data_lines = f.readlines()

        try:
            # Use numpy to load the main data block
            raw_data = np.loadtxt(data_lines, delimiter='\t')
            self.time = raw_data[:, 1]
            for i, name in enumerate(self.marker_names):
                start_col = 2 + i * 3
                self.marker_data[name] = raw_data[:, start_col : start_col + 3]
        except ValueError as e:
            print(f"✗ ERROR: Could not parse data in TRC file. Check for missing values or formatting errors.")
            print(f"  > Numpy error: {e}")
            return

    def get_marker_names(self):
        return self.marker_names

    def get_num_frames(self):
        return self.num_frames

    def get_data_rate(self):
        return self.data_rate

    def get_time(self):
        return self.time

    def get_marker_data(self, marker_name):
        if marker_name not in self.marker_names:
            raise ValueError(f"Marker '{marker_name}' not found in TRC file.")
        return self.marker_data.get(marker_name)
