"""
Pure-IK trunk recovery: CS markers only, no injection.

trunk_forward_lean is solved by the IK optimizer from CS_1–CS_4 site positions.
Targets use absolute lab-frame delta from frame 0 (no shoulder pivot) to avoid
contamination from arm motion.

Usage:
  python scripts/test_cs_trunk_ik.py [trial_stem]
  Default trial: S5_12_1

Validation: Pearson r between recovered trunk_forward_lean and CS plate normal
(from inject_cs_trunk.cs_lean_rad).
"""
import os
import sys
import collections
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import myosuite
from myosuite.physics import sim_scene
try:
    from myosuite.utils.trc_parser import TRCParser
except ImportError:
    from src.utils.trc_parser import TRCParser

from src.utils.config import get_path, get, get_project_root
from scripts.inject_cs_trunk import load_csv, wrist_fwd_axis, cs_lean_rad

ROOT       = get_project_root()
MODEL_PATH = get_path("mujoco_arm_model").replace("myoarm.xml", "myoarm_trunk.xml")

TRIAL_STEM = sys.argv[1] if len(sys.argv) > 1 else "S5_12_1"

# Resolve TRC and CSV paths
if TRIAL_STEM.startswith("S"):
    GROUP = "stroke"
else:
    GROUP = "healthy"

TRC_PATH = os.path.join(ROOT, "output", "originals", GROUP, "trc", TRIAL_STEM + ".trc")
CSV_PATH = None
for candidate in [
    os.path.join(ROOT, "data", "kinematic", GROUP, TRIAL_STEM + ".csv"),
    os.path.join(ROOT, "data", GROUP, TRIAL_STEM + ".csv"),
]:
    if os.path.exists(candidate):
        CSV_PATH = candidate
        break

OUT_DIR = os.path.join(ROOT, "output", "cs_trunk_ik_test")
os.makedirs(OUT_DIR, exist_ok=True)

IKResult = collections.namedtuple('IKResult', ['qpos', 'err_norm', 'steps', 'success'])

CS_MARKERS     = ['CS_1', 'CS_2', 'CS_3', 'CS_4']
TRUNK_JOINT    = 'trunk_forward_lean'
PHANTOM_JOINTS = [
    'unrotscap_r2', 'unrotscap_r3',
    'acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3',
    'unrothum_r1', 'unrothum_r2', 'unrothum_r3',
    'shoulder1_r2',
    'trunk_lateral_lean',
]

DAMPING     = 1e-4   # very small — only 1 active DOF
MAX_ITERS   = 200
CONV_THRESH = 1e-5   # tight for 1-DOF


def _unwrap(sim):
    model = sim.model.ptr if hasattr(sim.model, 'ptr') else sim.model
    data  = sim.data.ptr  if hasattr(sim.data,  'ptr') else sim.data
    return model, data


def _jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _sid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def solve_trunk_only(model, data, target_vec, site_ids, trunk_dof,
                     phantom_qpos_idxs, limit_qidxs, limit_lo, limit_hi):
    """1-DOF IK: only trunk_forward_lean moves, minimises CS site error."""
    nv = model.nv
    damping = np.zeros(nv)
    damping[trunk_dof] = DAMPING

    # Build a mask: only trunk DOF active
    mask = np.zeros(nv)
    mask[trunk_dof] = 1.0

    err_norm = np.inf
    for step in range(MAX_ITERS):
        data.qpos[phantom_qpos_idxs] = 0.0
        mujoco.mj_forward(model, data)

        curr     = np.concatenate([data.site_xpos[sid] for sid in site_ids])
        err      = target_vec - curr
        err_norm = np.linalg.norm(err)
        if err_norm < CONV_THRESH:
            return IKResult(data.qpos.copy(), err_norm, step, True)

        jac = np.zeros((3 * len(site_ids), nv))
        for i, sid in enumerate(site_ids):
            mujoco.mj_jacSite(model, data, jac[3*i:3*i+3], None, sid)

        # Only use the trunk column
        j_trunk = jac[:, trunk_dof]                  # (3*n_sites,)
        dq_trunk = float(
            (j_trunk @ err) / (j_trunk @ j_trunk + DAMPING)
        )
        dq_trunk = np.clip(dq_trunk, -0.1, 0.1)
        data.qpos[trunk_dof] += dq_trunk
        data.qpos[limit_qidxs] = np.clip(data.qpos[limit_qidxs], limit_lo, limit_hi)

    return IKResult(data.qpos.copy(), err_norm, MAX_ITERS, False)


def main():
    print(f"\n=== CS-Only Trunk IK: {TRIAL_STEM} ===")
    print(f"  TRC: {TRC_PATH}")
    print(f"  CSV: {CSV_PATH}")

    if not os.path.exists(TRC_PATH):
        print(f"[ERROR] TRC not found: {TRC_PATH}")
        sys.exit(1)

    sim = sim_scene.SimScene.get_sim(MODEL_PATH).sim
    model, data = _unwrap(sim)

    trc = TRCParser(TRC_PATH)
    # No retargeting: use raw lab mm coordinates for absolute delta
    n = trc.get_num_frames()
    rate = trc.get_data_rate()
    print(f"  Frames: {n}  Rate: {rate} Hz")

    # ── Trunk DOF index ───────────────────────────────────────────────────────
    jid = _jid(model, TRUNK_JOINT)
    if jid == -1:
        print(f"[ERROR] Joint '{TRUNK_JOINT}' not found in model")
        sys.exit(1)
    trunk_dof  = model.jnt_dofadr[jid]
    trunk_qpos = model.jnt_qposadr[jid]
    print(f"  trunk_forward_lean: joint id={jid}, dof={trunk_dof}, qpos={trunk_qpos}")

    # ── Phantom qpos indices ──────────────────────────────────────────────────
    phantom_qpos = []
    for name in PHANTOM_JOINTS:
        j = _jid(model, name)
        if j != -1:
            phantom_qpos.append(model.jnt_qposadr[j])
    phantom_qpos_idxs = np.array(phantom_qpos, dtype=int)

    # ── Joint limits ──────────────────────────────────────────────────────────
    qidxs, lows, highs = [], [], []
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            qidxs.append(model.jnt_qposadr[j])
            lo, hi = model.jnt_range[j]
            lows.append(lo); highs.append(hi)
    limit_qidxs = np.array(qidxs, dtype=int)
    limit_lo    = np.array(lows)
    limit_hi    = np.array(highs)

    # ── CS site ids ───────────────────────────────────────────────────────────
    site_ids = []
    for name in CS_MARKERS:
        sid = _sid(model, name)
        if sid == -1:
            print(f"[ERROR] Site '{name}' not found in model")
            sys.exit(1)
        site_ids.append(sid)

    # ── Model default CS world positions (trunk=0) ────────────────────────────
    data.qpos[:] = 0.0
    data.qpos[phantom_qpos_idxs] = 0.0
    mujoco.mj_forward(model, data)
    model_cs0 = np.array([data.site_xpos[sid].copy() for sid in site_ids])
    print(f"  Model CS centroid (trunk=0): {model_cs0.mean(axis=0)}")

    # ── Lab CS trajectories (raw mm) ──────────────────────────────────────────
    missing = [m for m in CS_MARKERS if m not in trc.get_marker_names()]
    if missing:
        print(f"[ERROR] CS markers missing from TRC: {missing}")
        sys.exit(1)
    lab_cs = {m: trc.get_marker_data(m) for m in CS_MARKERS}   # (n, 3) in mm
    lab_cs0 = {m: lab_cs[m][0].copy() for m in CS_MARKERS}     # frame 0 baseline

    # ── CS scale factor: robot_cs_dist / lab_cs_dist ──────────────────────────
    # Mirrors what trc_data_scaler does but applied to absolute delta, not shoulder-pivot.
    sid_shoulder = _sid(model, 'V_Shoulder')
    robot_shoulder = data.site_xpos[sid_shoulder].copy()
    robot_cs_dist  = np.linalg.norm(model_cs0.mean(axis=0) - robot_shoulder)

    lab_shoulder_traj = trc.get_marker_data('V_Shoulder')  # (n, 3) mm
    n_ref = min(10, n)
    lab_cs_dists = []
    for i in range(n_ref):
        lab_centroid = np.mean([lab_cs[m][i] for m in CS_MARKERS], axis=0)
        lab_cs_dists.append(np.linalg.norm((lab_centroid - lab_shoulder_traj[i]) / 1000.0))
    lab_cs_dist = np.mean(lab_cs_dists)
    cs_scale = robot_cs_dist / lab_cs_dist
    print(f"  Robot shoulder→CS: {robot_cs_dist*1000:.1f}mm  "
          f"Lab shoulder→CS: {lab_cs_dist*1000:.1f}mm  "
          f"CS scale: {cs_scale:.4f}")

    # Lab → model coordinate transform:  [lab_X, lab_Y, lab_Z] → [lab_X, -lab_Z, lab_Y]
    def lab_to_model(v_mm):
        v = v_mm / 1000.0 * cs_scale
        return np.array([v[0], -v[2], v[1]])

    # ── Build target vectors ──────────────────────────────────────────────────
    target_vecs = []
    for i in range(n):
        targets = []
        for k, name in enumerate(CS_MARKERS):
            delta_lab  = lab_cs[name][i] - lab_cs0[name]   # mm change from frame 0
            delta_model = lab_to_model(delta_lab)
            targets.append(model_cs0[k] + delta_model)
        target_vecs.append(np.concatenate(targets))

    # ── Run IK frame by frame ─────────────────────────────────────────────────
    recovered_trunk = []
    errs = []
    for i, tv in enumerate(target_vecs):
        if i == 0:
            data.qpos[trunk_qpos] = 0.0   # start at neutral
        mujoco.mj_forward(model, data)
        res = solve_trunk_only(model, data, tv, site_ids, trunk_dof,
                               phantom_qpos_idxs, limit_qidxs, limit_lo, limit_hi)
        recovered_trunk.append(data.qpos[trunk_qpos])
        errs.append(res.err_norm)

    recovered_trunk = np.array(recovered_trunk)
    errs_mm = np.array(errs) * 1000
    print(f"\n  IK error: mean={errs_mm.mean():.2f}mm  p95={np.percentile(errs_mm,95):.2f}mm  "
          f"max={errs_mm.max():.2f}mm")
    print(f"  Trunk range: [{np.degrees(recovered_trunk.min()):.1f}, "
          f"{np.degrees(recovered_trunk.max()):.1f}]°")

    # ── CS plate normal reference ─────────────────────────────────────────────
    if CSV_PATH is None:
        print("\n[WARN] No CSV found — skipping plate normal comparison")
        cs_ref = None
    else:
        df   = load_csv(CSV_PATH)
        fwd  = wrist_fwd_axis(df)
        tilt = cs_lean_rad(df, fwd)
        if tilt is None:
            cs_ref = None
        else:
            baseline = float(np.nanmean(tilt[:5]))
            cs_ref   = tilt - baseline
            nf       = min(len(cs_ref), n)
            cs_ref   = cs_ref[:nf]
            r_trunk  = recovered_trunk[:nf]
            r, p     = pearsonr(r_trunk, cs_ref)
            print(f"\n  Pearson r (IK vs CS plate normal): r={r:.4f}  p={p:.6f}  n={nf}")
            rmse = np.sqrt(np.mean((r_trunk - cs_ref)**2))
            print(f"  RMSE: {np.degrees(rmse):.2f}°")

    # ── Save plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Time series
    t = np.arange(n) / rate
    axes[0].plot(t, np.degrees(recovered_trunk), label='IK trunk_forward_lean', lw=1.5)
    if cs_ref is not None:
        tc = np.arange(len(cs_ref)) / rate
        axes[0].plot(tc, np.degrees(cs_ref), '--', label='CS plate normal (reference)', lw=1.2)
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Angle (°)")
    axes[0].set_title(f"CS-Only Trunk IK — {TRIAL_STEM}")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Scatter
    if cs_ref is not None:
        nf = min(len(cs_ref), n)
        axes[1].scatter(np.degrees(cs_ref), np.degrees(recovered_trunk[:nf]),
                        alpha=0.4, s=8, c=np.arange(nf), cmap='viridis')
        mn = min(np.degrees(cs_ref).min(), np.degrees(recovered_trunk[:nf]).min()) - 0.5
        mx = max(np.degrees(cs_ref).max(), np.degrees(recovered_trunk[:nf]).max()) + 0.5
        axes[1].plot([mn, mx], [mn, mx], 'k--', lw=0.8)
        axes[1].set_xlabel("CS plate normal (°)"); axes[1].set_ylabel("IK trunk_forward_lean (°)")
        axes[1].set_title(f"r={r:.4f}  RMSE={np.degrees(rmse):.2f}°  n={nf}")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{TRIAL_STEM}_cs_trunk_ik.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
