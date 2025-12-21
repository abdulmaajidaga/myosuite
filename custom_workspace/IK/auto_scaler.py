"""
auto_scaler.py
--------------
Automatically calculates subject limb lengths from TRC data and scales 
the MuJoCo model bodies (Humerus/Ulna/Hand) to match the subject.
"""

import numpy as np
import mujoco

# ==========================================
# BODY CONFIGURATION (Edit if your XML uses different names)
# ==========================================
# The body that starts at the Elbow (defines Upper Arm length)
ELBOW_BODY_NAMES = ['ulna', 'ulna_r', 'forearm'] 
# The body that starts at the Wrist (defines Forearm length)
WRIST_BODY_NAMES = ['hand', 'hand_r', 'wrist'] 

def find_body_id(model, potential_names):
    """Helper to find the first matching body name in the model."""
    for name in potential_names:
        # mj_name2id requires the raw model pointer, not the wrapper
        id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if id != -1:
            return id, name
    return -1, None

def modify_sim_model(sim, trc_data):
    """
    1. Calculates subject arm lengths from TRC.
    2. Identifies relevant MuJoCo bodies.
    3. Updates sim.model.body_pos to match subject lengths.
    """
    print("\n[Auto-Scaler] Analyzing TRC data for subject sizing...")
    
    # 1. Get Marker Data
    try:
        # Assuming standard naming from your previous files
        s_pos = trc_data.get_marker_data('V_Shoulder')
        e_pos = trc_data.get_marker_data('V_Elbow')
        w_pos = trc_data.get_marker_data('V_Wrist')
    except KeyError as e:
        print(f"  [!] Scaling skipped: Missing marker {e}")
        return sim

    # 2. Calculate Kinematic Lengths (averaged over all frames)
    # Using Median to be robust against outliers/marker dropouts
    trc_upper_len = np.nanmedian(np.linalg.norm(e_pos - s_pos, axis=1)) / 1000.0
    trc_fore_len  = np.nanmedian(np.linalg.norm(w_pos - e_pos, axis=1)) / 1000.0

    print(f"  > Subject Upper Arm: {trc_upper_len:.4f} m")
    print(f"  > Subject Forearm:   {trc_fore_len:.4f} m")

    # 3. Locate Model Bodies
    # FIX: Handle dm_control wrapper by extracting the raw pointers
    if hasattr(sim.model, 'ptr'):
        raw_model = sim.model.ptr
        raw_data = sim.data.ptr
    else:
        raw_model = sim.model
        raw_data = sim.data
    
    # Find Elbow Body (The body whose position is relative to the shoulder/humerus)
    elbow_id, elbow_name = find_body_id(raw_model, ELBOW_BODY_NAMES)
    
    # Find Wrist/Hand Body (The body whose position is relative to the elbow)
    wrist_id, wrist_name = find_body_id(raw_model, WRIST_BODY_NAMES)

    # 4. Apply Scaling
    changes_made = False
    
    if elbow_id != -1:
        # Current offset of the elbow body from its parent (Humerus)
        current_offset = raw_model.body_pos[elbow_id]
        current_len = np.linalg.norm(current_offset)
        
        if current_len > 0.001:
            # Calculate scaling factor
            scale_factor = trc_upper_len / current_len
            new_offset = current_offset * scale_factor
            
            # UPDATE MODEL
            raw_model.body_pos[elbow_id] = new_offset
            print(f"  > Scaled Body '{elbow_name}': {current_len:.3f}m -> {trc_upper_len:.3f}m")
            changes_made = True
        else:
            print(f"  [!] Warning: '{elbow_name}' has 0 length offset. Skipping.")
    else:
        print(f"  [!] Could not find Elbow body (tried: {ELBOW_BODY_NAMES})")

    if wrist_id != -1:
        # Current offset of the hand body from its parent (Ulna/Radius)
        current_offset = raw_model.body_pos[wrist_id]
        current_len = np.linalg.norm(current_offset)
        
        if current_len > 0.001:
            # Calculate scaling factor
            scale_factor = trc_fore_len / current_len
            new_offset = current_offset * scale_factor
            
            # UPDATE MODEL
            raw_model.body_pos[wrist_id] = new_offset
            print(f"  > Scaled Body '{wrist_name}': {current_len:.3f}m -> {trc_fore_len:.3f}m")
            changes_made = True
        else:
            print(f"  [!] Warning: '{wrist_name}' has 0 length offset. Skipping.")
    else:
        print(f"  [!] Could not find Hand body (tried: {WRIST_BODY_NAMES})")

    if changes_made:
        # Important: Propagate changes through the simulation
        mujoco.mj_forward(raw_model, raw_data)
        print("  ✓ Model successfully scaled to subject dimensions.")
    
    return sim