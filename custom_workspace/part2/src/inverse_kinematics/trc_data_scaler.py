"""
FILE: trc_data_scaler.py
"""
import numpy as np
import mujoco

CS_MARKER_NAMES = ['CS_1', 'CS_2', 'CS_3', 'CS_4']


def apply_retargeting(sim, trc_data):
    """
    Segment-aware scaling:
      - Arm markers (V_Shoulder, V_Elbow, V_Wrist, V_Vector):
          scaled by robot_arm_length / lab_arm_length  (shoulder+elbow to wrist)
      - CS chest markers (CS_1–CS_4):
          scaled by robot_shoulder_to_CS / lab_shoulder_to_CS
          so the chest cluster sits at the correct distance from the shoulder
          regardless of the subject's arm length.

    All scaling is shoulder-pivoted: the relative vector from shoulder to each
    marker is scaled, then added back to the (unscaled) shoulder position.
    """
    print("\n[trc_data_scaler] 📐 Scaling Motion Data...")

    raw_model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    raw_data  = sim.data.ptr  if hasattr(sim.data,  'ptr') else sim.data

    # ── 1. Robot arm length ───────────────────────────────────────────────────
    sid_s = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_SITE, 'V_Shoulder')
    sid_e = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_SITE, 'V_Elbow')
    sid_w = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_SITE, 'V_Wrist')
    if sid_s == -1 or sid_e == -1 or sid_w == -1:
        print("  [!] Virtual sites not found. Skipping scaling.")
        return trc_data

    m_s = raw_data.site_xpos[sid_s].copy()
    m_e = raw_data.site_xpos[sid_e].copy()
    m_w = raw_data.site_xpos[sid_w].copy()
    len_robot = np.linalg.norm(m_e - m_s) + np.linalg.norm(m_w - m_e)
    print(f"  > Robot Arm Length: {len_robot*1000:.1f} mm")

    # ── 2. Robot shoulder-to-CS-centroid distance ─────────────────────────────
    cs_sids = [mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_SITE, n)
               for n in CS_MARKER_NAMES]
    cs_sids = [s for s in cs_sids if s != -1]
    if cs_sids:
        robot_cs_centroid = np.mean([raw_data.site_xpos[s] for s in cs_sids], axis=0)
        robot_cs_dist = np.linalg.norm(robot_cs_centroid - m_s)
        print(f"  > Robot Shoulder→CS centroid: {robot_cs_dist*1000:.1f} mm")
    else:
        robot_cs_dist = None

    # ── 3. Lab arm length and CS distance ────────────────────────────────────
    trc_s_sample = trc_data.get_marker_data('V_Shoulder')
    scale_unit   = 1000.0 if np.mean(np.abs(trc_s_sample)) > 50 else 1.0
    n_ref        = min(10, len(trc_s_sample))

    trc_s = trc_data.get_marker_data('V_Shoulder')
    trc_e = trc_data.get_marker_data('V_Elbow')
    trc_w = trc_data.get_marker_data('V_Wrist')

    arm_lens = []
    for i in range(n_ref):
        s = trc_s[i] / scale_unit
        e = trc_e[i] / scale_unit
        w = trc_w[i] / scale_unit
        arm_lens.append(np.linalg.norm(e - s) + np.linalg.norm(w - e))
    len_data = np.mean(arm_lens)
    if len_data == 0:
        print("  [!] Zero arm length in data. Skipping scaling.")
        return trc_data
    arm_scale = len_robot / len_data
    print(f"  > Data Arm Length:  {len_data*1000:.1f} mm")
    print(f"  > Arm scale factor: {arm_scale:.4f}")

    # CS-specific scale factor
    lab_cs_names = [n for n in CS_MARKER_NAMES
                    if n in trc_data.get_marker_names()]
    cs_scale = arm_scale  # fallback: use arm scale if no CS markers
    if robot_cs_dist is not None and lab_cs_names:
        lab_cs_dists = []
        for i in range(n_ref):
            lab_cs_pts = np.array([trc_data.get_marker_data(n)[i] / scale_unit
                                   for n in lab_cs_names])
            lab_centroid = lab_cs_pts.mean(axis=0)
            lab_cs_dists.append(np.linalg.norm(lab_centroid - trc_s[i] / scale_unit))
        lab_cs_dist = np.mean(lab_cs_dists)
        if lab_cs_dist > 1e-6:
            cs_scale = robot_cs_dist / lab_cs_dist
            print(f"  > Lab Shoulder→CS: {lab_cs_dist*1000:.1f} mm  "
                  f"→ CS scale: {cs_scale:.4f}")

    # ── 4. Apply per-marker scale ─────────────────────────────────────────────
    data_store = (getattr(trc_data, 'marker_data', None)
                  or getattr(trc_data, 'markers', None)
                  or getattr(trc_data, 'data', None))
    if data_store is None or not isinstance(data_store, dict):
        print("  [!] CRITICAL: Could not find marker dictionary.")
        return trc_data

    ref_traj = data_store['V_Shoulder']
    for name in trc_data.get_marker_names():
        if name not in data_store:
            continue
        sf = cs_scale if name in CS_MARKER_NAMES else arm_scale
        traj     = data_store[name]
        new_traj = np.zeros_like(traj)
        for i in range(len(traj)):
            new_traj[i] = ref_traj[i] + (traj[i] - ref_traj[i]) * sf
        data_store[name] = new_traj

    print("  > ✅ Data scaled successfully.")
    return trc_data