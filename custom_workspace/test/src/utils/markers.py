"""
Shared marker processing utilities for raw MoCap CSV → joint-center arrays.

Shared marker-processing functions for converting raw MoCap CSVs to joint-center
arrays. All raw-CSV-to-array processing should go through here.
"""
import numpy as np
import pandas as pd

# Marker groups on the body
MARKER_GROUPS = {
    'Chest': ['CS_1', 'CS_2', 'CS_3', 'CS_4'],
    'Shoulder': ['SA_1', 'SA_2', 'SA_3'],
    'Elbow': ['ELB_L', 'ELB_M'],
    'Wrist': ['WRA', 'WRB'],
}

# Standard output column names
ARM_COLS = [
    'Sh_x', 'Sh_y', 'Sh_z',
    'El_x', 'El_y', 'El_z',
    'Wr_x', 'Wr_y', 'Wr_z',
    'WrVec_x', 'WrVec_y', 'WrVec_z',
]
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']


def load_raw_csv(filepath):
    """
    Read an MHH 2-row header CSV and return (marker_map, df_data).

    The CSV layout is:
        Row 0: Marker names (e.g. " WRA,,, WRB,,, ...")
        Row 1: "X,Y,Z,X,Y,Z,..." (axis labels — ignored)
        Row 2+: numeric data

    Returns:
        marker_map: dict mapping marker_name -> column index (start of X,Y,Z triplet)
        df_data: DataFrame of numeric rows (header=None, columns are integer indices)
    """
    # Read header row to build marker map
    df_header = pd.read_csv(filepath, header=None, nrows=1)
    header_row = df_header.iloc[0].tolist()

    marker_map = {}
    for idx, val in enumerate(header_row):
        if pd.notna(val) and str(val).strip() != "":
            marker_map[str(val).strip()] = idx

    # Read data rows (skip marker-name row + XYZ row)
    df_data = pd.read_csv(filepath, header=None, skiprows=2)

    return marker_map, df_data


def extract_marker_xyz(df_data, marker_map, marker_name):
    """
    Get (N, 3) float array for a single marker from the raw data.

    Handles NaN via forward/backward fill, then fills remaining with 0.
    Returns None if the marker is not in marker_map.
    """
    if marker_name not in marker_map:
        return None
    start = marker_map[marker_name]
    data = df_data.iloc[:, start:start + 3].apply(pd.to_numeric, errors='coerce')
    data = data.ffill().bfill().fillna(0.0)
    return data.values


def compute_joint_centers(df_data, marker_map):
    """
    Compute joint-center positions by averaging each marker group.

    Returns dict {'Chest': (N,3), 'Shoulder': (N,3), 'Elbow': (N,3), 'Wrist': (N,3)}.
    Raises ValueError if any required marker is missing.
    """
    centers = {}
    for group, markers in MARKER_GROUPS.items():
        arrays = []
        for m in markers:
            xyz = extract_marker_xyz(df_data, marker_map, m)
            if xyz is None:
                raise ValueError(f"Missing marker '{m}' (group '{group}')")
            arrays.append(xyz)
        centers[group] = np.mean(arrays, axis=0)  # (N, 3)
    return centers


def compute_wrist_vector(df_data, marker_map):
    """WRB − WRA → (N, 3) direction vector."""
    wra = extract_marker_xyz(df_data, marker_map, 'WRA')
    wrb = extract_marker_xyz(df_data, marker_map, 'WRB')
    if wra is None or wrb is None:
        raise ValueError("Missing WRA or WRB markers for wrist vector")
    return wrb - wra


def normalize_by_chest(joint_centers):
    """
    Subtract the chest centroid from Shoulder, Elbow, and Wrist centres.

    Modifies the dict in-place and returns it for convenience.
    Note: Wrist *vector* is intentionally NOT normalised by position.
    """
    chest = joint_centers['Chest']
    for key in ('Shoulder', 'Elbow', 'Wrist'):
        joint_centers[key] = joint_centers[key] - chest
    return joint_centers


def compute_trunk(joint_centers):
    """
    Return the chest centroid as trunk position → (N, 3).

    Should be called *before* normalize_by_chest if you want absolute trunk.
    """
    return joint_centers['Chest'].copy()


def process_raw_to_arrays(filepath, include_trunk=False):
    """
    Full pipeline: raw MoCap CSV → numpy array of joint-center data.

    Steps:
        1. Load CSV, build marker map
        2. Compute joint centres (Shoulder, Elbow, Wrist, Chest)
        3. Compute wrist vector (WRB − WRA)
        4. Optionally capture trunk (= chest centroid) before normalisation
        5. Normalise arm joints by subtracting chest centroid
        6. Stack columns: [Sh, El, Wr, WrVec] (+ Trunk if requested)

    Args:
        filepath: path to the raw MHH CSV
        include_trunk: if True, append Trunk_x/y/z columns (15-dim output)

    Returns:
        (data, columns) where data is (N, 12) or (N, 15) and columns is the
        list of column name strings.
    """
    marker_map, df_data = load_raw_csv(filepath)

    # Check all required markers are present
    missing = []
    for group, markers in MARKER_GROUPS.items():
        for m in markers:
            if m not in marker_map:
                missing.append(m)
    if missing:
        raise ValueError(f"Missing markers in {filepath}: {missing}")

    joint_centers = compute_joint_centers(df_data, marker_map)
    wrist_vec = compute_wrist_vector(df_data, marker_map)

    # Capture trunk before normalisation (trunk = chest centroid)
    trunk = compute_trunk(joint_centers) if include_trunk else None

    # Normalise arm joints by chest
    normalize_by_chest(joint_centers)

    # Stack: Sh(3) + El(3) + Wr(3) + WrVec(3) = 12
    parts = [
        joint_centers['Shoulder'],
        joint_centers['Elbow'],
        joint_centers['Wrist'],
        wrist_vec,
    ]
    columns = list(ARM_COLS)

    if include_trunk:
        parts.append(trunk)
        columns = columns + list(TRUNK_COLS)

    data = np.concatenate(parts, axis=1)
    return data, columns
