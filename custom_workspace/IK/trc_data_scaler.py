"""
FILE: trc_data_scaler.py
"""
import numpy as np
import mujoco

def apply_retargeting(sim, trc_data):
    """
    Calculates scale factor based on V_Shoulder -> V_Elbow -> V_Wrist
    and applies it to all TRC markers in trc_data.marker_data
    """
    print("\n[trc_data_scaler] 📐 Scaling Motion Data...")

    # --- 1. Handle dm_control Wrappers (Fix for mj_name2id error) ---
    raw_model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    raw_data = sim.data.ptr if hasattr(sim.data, 'ptr') else sim.data
    # -------------------------------------------------------------

    # --- 2. Measure Robot Arm (Upper + Forearm) ---
    try:
        sid_s = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_SITE, 'V_Shoulder')
        sid_e = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_SITE, 'V_Elbow')
        sid_w = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_SITE, 'V_Wrist')

        if sid_s == -1 or sid_e == -1 or sid_w == -1:
            print("  [!] Virtual sites not found in model. Skipping scaling.")
            return trc_data

        m_s = raw_data.site_xpos[sid_s]
        m_e = raw_data.site_xpos[sid_e]
        m_w = raw_data.site_xpos[sid_w]

        len_robot = np.linalg.norm(m_e - m_s) + np.linalg.norm(m_w - m_e)
        print(f"  > Robot Arm Length: {len_robot*1000:.1f} mm")

    except Exception as e:
        print(f"  [!] Error measuring robot: {e}")
        return trc_data

    # --- 3. Measure TRC Data Arm ---
    try:
        # Detect units (if values > 50, assume mm)
        trc_s_sample = trc_data.get_marker_data('V_Shoulder') 
        scale_unit = 1000.0 if np.mean(np.abs(trc_s_sample)) > 50 else 1.0
        
        # Calculate average length over first 10 frames
        n_frames = min(10, len(trc_s_sample))
        data_lens = []
        
        # Get data arrays
        trc_s = trc_data.get_marker_data('V_Shoulder')
        trc_e = trc_data.get_marker_data('V_Elbow')
        trc_w = trc_data.get_marker_data('V_Wrist')

        for i in range(n_frames):
            s = trc_s[i] / scale_unit
            e = trc_e[i] / scale_unit
            w = trc_w[i] / scale_unit
            data_lens.append(np.linalg.norm(e - s) + np.linalg.norm(w - e))
        
        len_data = np.mean(data_lens)
        print(f"  > Data Arm Length:  {len_data*1000:.1f} mm")

    except Exception as e:
        print(f"  [!] Error measuring data: {e}")
        return trc_data

    # --- 4. Apply Scale ---
    if len_data == 0: return trc_data

    scale_factor = len_robot / len_data
    print(f"  > 💡 Scale Factor: {scale_factor:.4f}")

    # Correctly identify the storage dictionary
    # Your logs confirmed 'marker_data' exists
    data_store = getattr(trc_data, 'marker_data', None)
    
    if data_store is None:
        # Fallback attempts
        data_store = getattr(trc_data, 'markers', None) or getattr(trc_data, 'data', None)

    if data_store is None or not isinstance(data_store, dict):
        print("  [!] CRITICAL: Could not find writable marker dictionary in TRCParser.")
        return trc_data

    # Apply scaling loop
    try:
        # We need the shoulder trajectory to act as the pivot point
        ref_traj = data_store['V_Shoulder']
        
        for name in trc_data.get_marker_names():
            if name not in data_store:
                continue
                
            traj = data_store[name]
            new_traj = np.zeros_like(traj)
            
            for i in range(len(traj)):
                # Vector from shoulder to this marker
                rel_vec = traj[i] - ref_traj[i]
                # Scale the vector
                rel_vec_scaled = rel_vec * scale_factor
                # Add back to shoulder position
                new_traj[i] = ref_traj[i] + rel_vec_scaled
            
            # Overwrite the data in place
            data_store[name] = new_traj
            
        print("  > ✅ Data scaled successfully.")
        
    except Exception as e:
        print(f"  [!] Error applying scaling: {e}")

    return trc_data