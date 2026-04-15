"""
Inverse Dynamics: MOT -> joint torques + muscle activations.

Supports single-file mode (via module-level vars or CLI) and batch mode
(directory of .mot files).

Approach:
- Disables MuJoCo equality constraints (scapulohumeral rhythm) before
  mj_inverse() so qfrc_inverse contains TRUE physical torques (~12 Nm
  shoulder, ~3-4 Nm elbow) instead of constraint solver artifacts.
- Manually applies scapulohumeral rhythm (linear joint coupling) so
  scapulothoracic joints track shoulder_elv, giving correct muscle
  tendon wrapping and moment arms.
- Anatomical muscle-joint mask removes spurious cross-joint moment arms
  from the phantom body kinematic chain.
- Post-hoc spasticity-driven co-contraction adds activation-proportional
  antagonist co-activation scaled by FMA impairment level, producing
  clinically realistic CCI that increases with impairment.
- No calibration needed — raw qfrc_inverse is in real Nm.
- 5-DOF solving: shoulder (3) + elbow + pro_sup
- Phase detection: auto-detects Pick/Drink/Place from elbow velocity
- External bottle load: gravitational force at lunate for ALL phases
- Static optimization solver (L-BFGS-B) with Fmax scaling and arm-only muscles
- NNMF synergy extraction post-processing
- Per-FMA effort metrics (TRR with actual ROM, ATI, CCI)
"""
import os
import re
import sys
import glob
import json
import mujoco
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, argrelextrema
from scipy.optimize import minimize
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path, get

# =============================================================================
# CONFIGURATION
# =============================================================================

# Single-file defaults (can be monkey-patched by run_generated_pipeline.py)
MOT_FILE_PATH = get_path("output_dir") + "/S5_12_1.mot"
MODEL_XML_PATH = get_path("mujoco_arm_model")
OUTPUT_DIRECTORY = get_path("output_originals_id")

# Batch defaults
INPUT_PATH = get_path("output_originals_mot")
OUTPUT_ROOT = get_path("output_originals_id")

# Toggles
GENERATE_VIDEO = False

# Load config
_id_cfg = get("inverse_dynamics")
SHOULDER_JOINTS = _id_cfg.get("shoulder_joints", ['elv_angle', 'shoulder_elv', 'shoulder_rot'])
ELBOW_WRIST_JOINTS = _id_cfg.get("elbow_wrist_joints", ['elbow_flexion', 'pro_sup', 'deviation', 'flexion'])
# Solve joints: only DOFs fed to the muscle solver (excludes wrist/pro_sup noise)
SOLVE_JOINTS = _id_cfg.get("solve_joints",
    ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion'])

BOTTLE_MASS = _id_cfg.get("bottle_mass", 0.5)
LOAD_BODY = _id_cfg.get("load_body", "lunate")

PHASE_DETECTION = _id_cfg.get("phase_detection", True)
PHASE_VELOCITY_ORDER = _id_cfg.get("phase_velocity_order", 10)

SYNERGY_ENABLED = _id_cfg.get("synergy_enabled", True)
N_SYNERGIES = _id_cfg.get("n_synergies", 4)

LOW_PASS_CUTOFF = _id_cfg.get("low_pass_cutoff", 1.5)

# ATI calibration
ATI_CALIBRATION_ENABLED = _id_cfg.get("ati_calibration_enabled", False)
ATI_HEALTHY_BASELINE = _id_cfg.get("ati_healthy_baseline", 0.276)

# Triceps Fmax boost
TRICEPS_FMAX_BOOST = _id_cfg.get("triceps_fmax_boost", 1.0)

# Synergy noise injection
SYNERGY_NOISE_STD = _id_cfg.get("synergy_noise_std", 0.0)

# Solver config
SOLVER_WEIGHT = _id_cfg.get("solver_weight", 10000.0)
ARM_MUSCLE_COUNT = _id_cfg.get("arm_muscle_count", 32)
USE_FMAX_SCALING = _id_cfg.get("use_fmax_scaling", True)
OUTPUT_JOINTS = _id_cfg.get("output_joints",
    ['elv_angle', 'shoulder_elv', 'shoulder_rot',
     'elbow_flexion', 'pro_sup', 'deviation', 'flexion'])

# =============================================================================
# UTILITIES
# =============================================================================

def read_mot_file(filepath):
    """Read a .mot file and return a pandas DataFrame."""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return None
    skiprows = 0
    with open(filepath, "r") as f:
        for line in f:
            if "endheader" in line:
                break
            skiprows += 1
    return pd.read_csv(filepath, sep=r'\s+', skiprows=skiprows + 1)


def compute_derivatives(data, dt):
    """Compute velocity and acceleration using np.gradient."""
    vel = np.gradient(data, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def apply_lowpass_filter(data, cutoff, fs, order=4):
    """Apply a low-pass Butterworth filter column-wise."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if data.shape[0] < 15:
        return data
    if data.ndim == 1:
        return filtfilt(b, a, data)
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])
    return filtered


# =============================================================================
# PHASE DETECTION
# =============================================================================

def detect_phases(qpos_elbow, time, fs, order=None):
    """
    Detect Pick/Drink/Place phases from elbow_flexion angular velocity.

    Finds the 2 most prominent velocity minima to split the trajectory
    into 3 phases.

    Returns phase_indices: [0, min1, min2, n_frames]
    """
    if order is None:
        order = PHASE_VELOCITY_ORDER

    n_frames = len(qpos_elbow)

    # Compute angular velocity of elbow_flexion
    dt = 1.0 / fs
    vel = np.gradient(qpos_elbow, dt)
    vel_smooth = apply_lowpass_filter(vel, LOW_PASS_CUTOFF, fs)

    # Compute speed (absolute velocity)
    speed = np.abs(vel_smooth)

    # Find local minima in speed (movement pauses)
    # Adjust order for short sequences
    adjusted_order = min(order, max(2, n_frames // 6))
    minima = argrelextrema(speed, np.less, order=adjusted_order)[0]

    # Filter out minima too close to boundaries (each phase must be >= 15%)
    min_frac = 0.15
    lo = int(n_frames * min_frac)
    hi = int(n_frames * (1 - min_frac))
    valid = minima[(minima >= lo) & (minima <= hi)]

    if len(valid) >= 2:
        # Pick the 2 valid minima with smallest speed, ensure separation
        min_speeds = speed[valid]
        ranked = valid[np.argsort(min_speeds)]
        best_two = [ranked[0]]
        min_gap = int(n_frames * min_frac)
        for candidate in ranked[1:]:
            if abs(int(candidate) - int(best_two[0])) >= min_gap:
                best_two.append(candidate)
                break
        if len(best_two) == 2:
            best_two = np.sort(best_two)
            phase_indices = [0, int(best_two[0]), int(best_two[1]), n_frames]
        else:
            # Only found 1 valid minimum after separation check
            mid = int(best_two[0])
            phase_indices = [0, mid, (mid + n_frames) // 2, n_frames]
    elif len(valid) == 1:
        mid = int(valid[0])
        second = (mid + n_frames) // 2
        if second > hi:
            second = hi
        phase_indices = [0, mid, second, n_frames]
    else:
        # No valid minima: split into 3 equal parts
        third = n_frames // 3
        phase_indices = [0, third, 2 * third, n_frames]

    return phase_indices


def make_phase_labels(phase_indices, n_frames):
    """Create per-frame phase label array."""
    labels = np.empty(n_frames, dtype='U6')
    labels[phase_indices[0]:phase_indices[1]] = 'Pick'
    labels[phase_indices[1]:phase_indices[2]] = 'Drink'
    labels[phase_indices[2]:phase_indices[3]] = 'Place'
    return labels


# =============================================================================
# DOF INDEX MAPPING
# =============================================================================

def map_dof_indices(model, joint_names):
    """Map a list of joint names to their DOF addresses in the model."""
    indices = []
    for jname in joint_names:
        try:
            jid = model.joint(jname).id
            indices.append(model.jnt_dofadr[jid])
        except KeyError:
            pass
    return indices


# =============================================================================
# SCAPULOHUMERAL RHYTHM
# =============================================================================

# The model has 11 equality constraints implementing scapulohumeral rhythm:
#   scapulothoracic_joint = coefficient * shoulder_elv  (or elv_angle)
# We disable the MuJoCo constraint solver (to avoid massive constraint forces
# contaminating qfrc_inverse), but we still need the scapula to move correctly
# so that muscle tendon wrapping produces correct moment arms.
# Solution: manually set scapulothoracic joints using the linear coefficients.

SCAP_RHYTHM = {
    # joint_name: (driver_joint_name, coefficient)
    'sternoclavicular_r2': ('shoulder_elv', -0.242),
    'sternoclavicular_r3': ('shoulder_elv',  0.1025),
    'unrotscap_r2':        ('shoulder_elv',  0.242),
    'unrotscap_r3':        ('shoulder_elv', -0.1025),
    'acromioclavicular_r1':('shoulder_elv',  0.178),
    'acromioclavicular_r2':('shoulder_elv', -0.049),
    'acromioclavicular_r3':('shoulder_elv',  0.396),
    'unrothum_r1':         ('shoulder_elv', -0.178),
    'unrothum_r2':         ('shoulder_elv',  0.049),
    'unrothum_r3':         ('shoulder_elv', -0.396),
    'shoulder1_r2':        ('elv_angle',    -1.0),
}


def apply_scapulohumeral_rhythm(model, data):
    """
    Set scapulothoracic joints from their driving joints using the
    linear scapulohumeral rhythm coefficients.

    Must be called AFTER setting the prescribed joint angles (shoulder_elv,
    elv_angle) and BEFORE computing moment arms.
    """
    for scap_joint, (driver_joint, coeff) in SCAP_RHYTHM.items():
        try:
            scap_id = model.joint(scap_joint).id
            driver_id = model.joint(driver_joint).id
            driver_val = data.qpos[model.jnt_qposadr[driver_id]]
            scap_val = coeff * driver_val
            # Clamp to joint range if limited
            if model.jnt_limited[scap_id]:
                lo, hi = model.jnt_range[scap_id]
                scap_val = np.clip(scap_val, lo, hi)
            data.qpos[model.jnt_qposadr[scap_id]] = scap_val
        except KeyError:
            pass  # Joint not found in model


def _build_coupling_map(model):
    """
    Build a mapping from solve DOF index -> list of (scap_dof_index, coefficient).

    This captures which scapulothoracic DOFs are driven by each solve DOF,
    needed for computing effective moment arms and torques.
    """
    # Group scap joints by their driver joint
    driver_groups = {}  # driver_joint_name -> [(scap_joint_name, coeff), ...]
    for scap_joint, (driver_joint, coeff) in SCAP_RHYTHM.items():
        if driver_joint not in driver_groups:
            driver_groups[driver_joint] = []
        driver_groups[driver_joint].append((scap_joint, coeff))

    # Map to DOF indices
    coupling = {}  # driver_dof_idx -> [(scap_dof_idx, coeff), ...]
    for driver_name, deps in driver_groups.items():
        try:
            driver_dof = model.jnt_dofadr[model.joint(driver_name).id]
        except KeyError:
            continue
        dof_deps = []
        for scap_name, coeff in deps:
            try:
                scap_dof = model.jnt_dofadr[model.joint(scap_name).id]
                dof_deps.append((scap_dof, coeff))
            except KeyError:
                pass
        if dof_deps:
            coupling[driver_dof] = dof_deps
    return coupling


# =============================================================================
# ANATOMICAL MUSCLE-JOINT MASK
# =============================================================================
# MuJoCo's spatial tendon moment arm computation produces spurious cross-joint
# entries (e.g., BRA showing 65mm moment arm at acromioclavicular_r3 despite
# only crossing the elbow). This is an artifact of the phantom body kinematic
# chain interacting with wrapping geometry. We mask the moment arm matrix to
# only allow each muscle to produce force at joints it anatomically crosses.

MUSCLE_JOINT_MAP = {
    # Shoulder-only muscles (scapula/clavicle/thorax -> humerus)
    'DELT1':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'DELT2':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'DELT3':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'SUPSP':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'INFSP':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'SUBSC':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'TMIN':    ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'TMAJ':    ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'PECM1':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'PECM2':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'PECM3':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'LAT1':    ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'LAT2':    ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'LAT3':    ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'CORB':    ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    # Biarticular muscles (scapula -> radius/ulna, crossing shoulder + elbow)
    'BIClong': ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion', 'pro_sup'],
    'BICshort':['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion', 'pro_sup'],
    'TRIlong': ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion'],
    'TRIlat':  ['elbow_flexion'],  # lateral head: humerus -> ulna
    'TRImed':  ['elbow_flexion'],  # medial head: humerus -> ulna
    'ANC':     ['elbow_flexion'],  # humerus -> ulna
    # Elbow/forearm muscles
    'BRA':     ['elbow_flexion'],           # humerus -> ulna
    'BRD':     ['elbow_flexion', 'pro_sup'],# humerus -> radius
    'SUP':     ['elbow_flexion', 'pro_sup'],# humerus/ulna -> radius
    # Forearm muscles (cross elbow and/or wrist)
    'ECRL':    ['elbow_flexion', 'pro_sup', 'deviation', 'flexion'],
    'ECRB':    ['pro_sup', 'deviation', 'flexion'],
    'ECU':     ['pro_sup', 'deviation', 'flexion'],
    'FCR':     ['pro_sup', 'deviation', 'flexion'],
    'FCU':     ['elbow_flexion', 'pro_sup', 'deviation', 'flexion'],
    'PL':      ['pro_sup', 'deviation', 'flexion'],
    'PT':      ['elbow_flexion', 'pro_sup'],   # humerus -> radius
    'PQ':      ['pro_sup'],                     # radius -> ulna
}


def build_anatomical_mask(model, solve_joints, arm_muscle_count=32):
    """
    Build a binary mask (n_solve, n_arm) where mask[j,m] = 1 if muscle m
    anatomically crosses solve joint j.

    Zeros out spurious MuJoCo moment arms from the phantom body chain.
    """
    n_arm = min(arm_muscle_count, model.nu)
    n_solve = len(solve_joints)
    mask = np.zeros((n_solve, n_arm), dtype=np.float64)

    for mi in range(n_arm):
        mname = model.actuator(mi).name
        allowed = MUSCLE_JOINT_MAP.get(mname, [])
        for i, jname in enumerate(solve_joints):
            if jname in allowed:
                mask[i, mi] = 1.0
    return mask


# =============================================================================
# SOLVER
# =============================================================================

def compute_coupled_moment_arms(model, data, solve_dof_indices, n_arm, eps=1e-4):
    """
    Compute moment arms via finite differences with scapulohumeral rhythm.

    For each solve DOF, perturbs q ± eps, applies scapulohumeral rhythm at
    the perturbed pose, runs mj_forward, and measures actuator length change.
    This gives the TRUE coupled moment arm that accounts for the scapula
    moving with the shoulder, unlike MuJoCo's analytic partial derivatives.

    Returns: M (n_solve, n_arm) moment arm matrix.
    """
    M = np.zeros((len(solve_dof_indices), n_arm))
    qpos_save = data.qpos.copy()

    for i, dof in enumerate(solve_dof_indices):
        # Forward: q + eps
        data.qpos[:] = qpos_save
        data.qpos[dof] += eps
        apply_scapulohumeral_rhythm(model, data)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        L_plus = data.actuator_length[:n_arm].copy()

        # Backward: q - eps
        data.qpos[:] = qpos_save
        data.qpos[dof] -= eps
        apply_scapulohumeral_rhythm(model, data)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        L_minus = data.actuator_length[:n_arm].copy()

        # Moment arm = -dL/dq (negative because shorter tendon = positive moment)
        M[i, :] = -(L_plus - L_minus) / (2 * eps)

    # Restore original pose
    data.qpos[:] = qpos_save
    apply_scapulohumeral_rhythm(model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    return M


def solve_muscle_activations_static_opt(model, data, tau_scaled, solve_dof_indices,
                                         fmax, prev_a=None,
                                         arm_muscle_count=32, weight=10000.0,
                                         anat_mask=None, coupling_map=None):
    """
    Solve for muscle activations using static optimization (L-BFGS-B).

    Only solves for arm muscles (0..arm_muscle_count-1). Finger muscles stay at 0.
    Uses anatomical mask to remove spurious moment arms, then optionally applies
    scapulohumeral coupling correction for shoulder muscles to boost effective
    moment arms (especially DELT1).
    """
    n_arm = min(arm_muscle_count, model.nu)
    n_solve = len(solve_dof_indices)
    n_total = model.nu

    if n_solve == 0:
        return np.zeros(n_total)

    # Apply scapulohumeral rhythm so tendon wrapping reflects correct scapula position
    apply_scapulohumeral_rhythm(model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    moment_arm = data.actuator_moment.reshape(model.nv, n_total)

    # Extract moment arms for arm muscles only at solve DOFs
    M_geom = moment_arm[solve_dof_indices, :n_arm].copy()  # (n_solve, n_arm)
    tau = tau_scaled[solve_dof_indices].copy()               # (n_solve,)

    # Apply anatomical mask: zero out moment arms for joints a muscle doesn't cross
    if anat_mask is not None:
        M_geom *= anat_mask

    # Selective coupling correction: for shoulder muscles, add the scapulothoracic
    # coupling contribution to their effective moment arms. Only applied to
    # muscle-joint entries that survived anatomical masking, preventing spurious
    # amplification of forearm muscles at shoulder DOFs.
    if coupling_map is not None and anat_mask is not None:
        full_M = moment_arm[:, :n_arm]  # full nv × n_arm matrix
        for i, solve_dof in enumerate(solve_dof_indices):
            if solve_dof in coupling_map:
                for scap_dof, coeff in coupling_map[solve_dof]:
                    for m in range(n_arm):
                        if anat_mask[i, m] > 0:  # only shoulder muscles
                            M_geom[i, m] += coeff * full_M[scap_dof, m]

    # Fmax scaling: M_eff[i,j] = M_geom[i,j] * fmax[j]
    fmax_arm = fmax[:n_arm]
    M_eff = M_geom * fmax_arm[np.newaxis, :]  # (n_solve, n_arm)

    # Precompute for objective and gradient
    MtM = M_eff.T @ M_eff      # (n_arm, n_arm)
    Mt_tau = M_eff.T @ tau      # (n_arm,)
    w = weight

    def objective(a):
        residual = M_eff @ a - tau
        return np.sum(a**2) + w * np.sum(residual**2)

    def gradient(a):
        return 2.0 * a + 2.0 * w * (MtM @ a - Mt_tau)

    # Initial guess: warm-start from previous frame or zeros
    if prev_a is not None:
        a0 = prev_a[:n_arm].copy()
    else:
        a0 = np.zeros(n_arm)

    bounds = [(0.0, 1.0)] * n_arm

    result = minimize(objective, a0, jac=gradient, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})

    a_full = np.zeros(n_total)
    a_full[:n_arm] = np.clip(result.x, 0, 1)
    return a_full


# =============================================================================
# SYNERGY EXTRACTION
# =============================================================================

def extract_synergies(activations, n_synergies=4):
    """
    Extract muscle synergies via Non-Negative Matrix Factorization.

    activations: (n_frames, n_muscles) non-negative matrix
    Returns: synergy_weights, synergy_coeffs, reconstructed, var_explained
    """
    from sklearn.decomposition import NMF

    # Ensure non-negative input
    act_nn = np.clip(activations, 0, None)

    # Skip if all zeros
    if act_nn.max() < 1e-10:
        n_muscles = activations.shape[1]
        return (np.zeros((n_synergies, n_muscles)),
                np.zeros((activations.shape[0], n_synergies)),
                np.zeros_like(activations), 0.0)

    nmf = NMF(n_components=n_synergies, init='nndsvda', max_iter=500, random_state=42)
    coeffs = nmf.fit_transform(act_nn)      # (n_frames, n_synergies)
    weights = nmf.components_                # (n_synergies, n_muscles)
    reconstructed = coeffs @ weights         # (n_frames, n_muscles)

    ss_total = np.sum(act_nn ** 2)
    ss_resid = np.sum((act_nn - reconstructed) ** 2)
    var_explained = 1 - (ss_resid / ss_total) if ss_total > 0 else 0.0

    return weights, coeffs, np.clip(reconstructed, 0, 1), var_explained


def sweep_synergies(activations, max_n=10, vaf_threshold=0.90):
    """
    Data-driven synergy count selection.

    Sweeps from 1 to max_n synergies and returns:
      - optimal_n: smallest n where VAF >= vaf_threshold
      - vaf_curve: dict {n: vaf} for all tested values

    The standard approach in the literature (Tresch et al. 2006, Ting & Chvatal 2010)
    is to increment n until global VAF crosses 90%, with an optional check that
    adding one more synergy gives <5% incremental improvement.
    """
    from sklearn.decomposition import NMF

    act_nn = np.clip(activations, 0, None)
    if act_nn.max() < 1e-10:
        return 1, {i: 0.0 for i in range(1, max_n + 1)}

    ss_total = np.sum(act_nn ** 2)
    vaf_curve = {}
    optimal_n = max_n

    for n in range(1, max_n + 1):
        nmf = NMF(n_components=n, init='nndsvda', max_iter=1000, random_state=42)
        coeffs = nmf.fit_transform(act_nn)
        recon = coeffs @ nmf.components_
        ss_resid = np.sum((act_nn - recon) ** 2)
        vaf = 1 - (ss_resid / ss_total)
        vaf_curve[n] = vaf

        if vaf >= vaf_threshold and n < optimal_n:
            optimal_n = n

    return optimal_n, vaf_curve


# =============================================================================
# EFFORT METRICS
# =============================================================================

def compute_effort_metrics(tau_history, act_history, model, output_dof_indices,
                           output_joint_names, arm_muscle_count,
                           qpos=None, valid_cols=None):
    """
    Compute per-motion effort metrics for FMA differentiation.

    Returns dict with:
      - TRR: Torque-ROM Ratio per joint (effort per unit movement)
      - ATI: Activation-Time Integral (total muscular effort)
      - CCI: Co-Contraction Index (biceps vs triceps overlap)
      - per_joint_mean_torque: mean |torque| per output joint
      - per_joint_rom: ROM per output joint (radians)
    """
    n_arm = min(arm_muscle_count, act_history.shape[1])
    arm_act = act_history[:, :n_arm]

    # ATI: mean sum of squared activations across time
    ati = float(np.mean(np.sum(arm_act**2, axis=1)))

    # CCI: biceps vs triceps co-contraction
    muscle_names = [model.actuator(i).name for i in range(model.nu)]
    bic_indices = [i for i, n in enumerate(muscle_names) if n in ('BIClong', 'BICshort') and i < n_arm]
    tri_indices = [i for i, n in enumerate(muscle_names) if n in ('TRIlong', 'TRIlat', 'TRImed') and i < n_arm]

    if bic_indices and tri_indices:
        biceps = np.mean(arm_act[:, bic_indices], axis=1)
        triceps = np.mean(arm_act[:, tri_indices], axis=1)
        numer = 2.0 * np.minimum(biceps, triceps)
        denom = biceps + triceps + 1e-8
        cci = float(np.mean(numer / denom))
    else:
        cci = 0.0

    # TRR per output joint — uses actual qpos ROM (radians) when available
    tau_output = tau_history[:, output_dof_indices]
    trr = {}
    per_joint_mean_torque = {}
    per_joint_rom = {}
    for j, jname in enumerate(output_joint_names):
        tau_j = tau_output[:, j]
        mean_tau = float(np.mean(np.abs(tau_j)))
        # Use actual joint ROM from qpos if available
        if qpos is not None and valid_cols is not None and jname in valid_cols:
            col_idx = list(valid_cols).index(jname)
            q_j = qpos[:, col_idx]
            rom_j = float(np.max(q_j) - np.min(q_j))  # radians
        else:
            rom_j = float(np.max(tau_j) - np.min(tau_j))  # fallback: torque range
        per_joint_mean_torque[jname] = mean_tau
        per_joint_rom[jname] = rom_j
        # TRR only meaningful for joints with real movement (>1 deg)
        if rom_j > 0.017:  # ~1 degree in radians
            trr[jname] = mean_tau / rom_j
        else:
            trr[jname] = 0.0

    return {
        'ATI': ati,
        'CCI': cci,
        'TRR': trr,
        'per_joint_mean_torque': per_joint_mean_torque,
        'per_joint_rom': per_joint_rom,
    }


# =============================================================================
# FMA SCORE EXTRACTION
# =============================================================================

def extract_fma_score(file_id):
    """
    Extract FMA score from filename.
    FMA_18 -> 18, FMA_66 -> 66, anything else -> 66 (healthy).
    """
    m = re.match(r'FMA_(\d+)', file_id)
    if m:
        return int(m.group(1))
    return 66  # healthy default


# =============================================================================
# SPASTICITY-DRIVEN CO-CONTRACTION
# =============================================================================

# Antagonist muscle pairs for co-contraction modelling.
# Each pair defines the agonist/antagonist groups at a joint.
# Spasticity causes the antagonist to fire when the agonist is active.
ANTAGONIST_PAIRS = [
    {   # Elbow flexion/extension
        'group_a': ['BIClong', 'BICshort', 'BRA', 'BRD'],   # flexors
        'group_b': ['TRIlong', 'TRIlat', 'TRImed', 'ANC'],  # extensors
    },
    {   # Shoulder elevation
        'group_a': ['DELT1', 'DELT2', 'SUPSP'],              # elevators
        'group_b': ['LAT1', 'LAT2', 'PECM2', 'PECM3'],      # depressors
    },
    {   # Shoulder rotation
        'group_a': ['INFSP', 'TMIN'],                        # external rotators
        'group_b': ['SUBSC', 'PECM1', 'LAT1'],               # internal rotators
    },
    {   # Pronation/supination
        'group_a': ['SUP', 'BIClong', 'BICshort'],           # supinators
        'group_b': ['PT', 'PQ'],                              # pronators
    },
]

# Spasticity model parameters — calibrated to produce CCI in published ranges:
#   Healthy: 0.30-0.39, Mild (FMA>40): 0.35-0.45,
#   Moderate (FMA 25-40): 0.41-0.50, Severe (FMA<25): 0.50-0.70
SPASTICITY_BASE_GAIN = 0.6    # agonist-proportional co-contraction strength
SPASTICITY_EXPONENT = 1.3     # nonlinear: more effect at lower FMA


def apply_spasticity_cocontraction(act_history, qvel, valid_cols, fma_score, model):
    """
    Add activation-proportional antagonist co-contraction scaled by impairment.

    Physiological basis: In stroke, spasticity causes involuntary antagonist
    activation whenever the agonist is active. The effect is proportional to:
      - Agonist activation level (more effort → more co-contraction)
      - Impairment level (lower FMA → more spasticity)

    For each antagonist pair, the weaker group gets boosted proportionally
    to the stronger group's activation, creating clinically realistic
    co-contraction that scales monotonically with impairment.

    Applied post-hoc to solver activations. Healthy subjects (FMA>=66) get
    no modification.
    """
    if fma_score >= 66:
        return act_history  # healthy, no spasticity

    impairment = 1.0 - fma_score / 66.0  # 0..1
    gain = SPASTICITY_BASE_GAIN * impairment ** SPASTICITY_EXPONENT

    # Map muscle names to indices
    muscle_names = [model.actuator(i).name for i in range(model.nu)]
    name_to_idx = {n: i for i, n in enumerate(muscle_names)}

    act_out = act_history.copy()
    n_frames = act_history.shape[0]

    for pair in ANTAGONIST_PAIRS:
        a_indices = [name_to_idx[n] for n in pair['group_a'] if n in name_to_idx]
        b_indices = [name_to_idx[n] for n in pair['group_b'] if n in name_to_idx]

        if not a_indices or not b_indices:
            continue

        for t in range(n_frames):
            # Mean activation of each group
            a_act = np.mean(act_out[t, a_indices])
            b_act = np.mean(act_out[t, b_indices])

            if a_act > b_act:
                # Group A is agonist → boost group B (antagonist)
                boost = gain * a_act
                for mi in b_indices:
                    act_out[t, mi] = min(1.0, act_out[t, mi] + boost)
            elif b_act > a_act:
                # Group B is agonist → boost group A
                boost = gain * b_act
                for mi in a_indices:
                    act_out[t, mi] = min(1.0, act_out[t, mi] + boost)

    return act_out


# =============================================================================
# ATI CALIBRATION
# =============================================================================

def calibrate_ati(act_history, arm_muscle_count, target_ati):
    """
    Scale activations so that ATI matches a target baseline.

    Uses sqrt(target/current) scaling since ATI = mean(sum(a^2)):
    if we scale all activations by k, ATI scales by k^2.
    Applied BEFORE spasticity co-contraction so spasticity adds
    impairment elevation on top of calibrated baseline.
    """
    n_arm = min(arm_muscle_count, act_history.shape[1])
    arm_act = act_history[:, :n_arm]
    current_ati = float(np.mean(np.sum(arm_act**2, axis=1)))

    if current_ati <= 0 or target_ati <= 0:
        return act_history

    scale = np.sqrt(target_ati / current_ati)
    act_out = act_history.copy()
    act_out[:, :n_arm] = np.clip(arm_act * scale, 0, 1)
    return act_out


# =============================================================================
# SYNERGY NOISE INJECTION
# =============================================================================

def inject_synergy_noise(act_history, arm_muscle_count, fma_score, noise_std):
    """
    Add FMA-scaled Gaussian noise to activations before NNMF extraction.

    Healthy subjects have more motor variability (noise), severe patients
    exhibit stereotyped (low-variability) patterns. This produces more
    realistic synergy VAF separation between groups.

    Formula: effective_noise = std * (0.3 + 0.7 * fma/66)
    """
    if noise_std <= 0:
        return act_history

    n_arm = min(arm_muscle_count, act_history.shape[1])
    effective_std = noise_std * (0.3 + 0.7 * fma_score / 66.0)
    rng = np.random.default_rng(seed=42 + fma_score)
    noise = rng.normal(0, effective_std, size=(act_history.shape[0], n_arm))

    act_out = act_history.copy()
    act_out[:, :n_arm] = np.clip(act_out[:, :n_arm] + noise, 0, 1)
    return act_out


# =============================================================================
# CORE PROCESSING
# =============================================================================

def process_file(mot_path, model_path, output_dir):
    """
    Run full inverse dynamics on a single MOT file.

    Saves torques (7 output joints), activations (arm muscles only),
    synergy data, phase labels, and effort metrics to output_dir.
    """
    file_id = os.path.basename(mot_path).replace('.mot', '')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing: {file_id}")

    df = read_mot_file(mot_path)
    if df is None:
        return
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    model_joints = [model.joint(j).name for j in range(model.njnt)]
    valid_cols = [c for c in df.columns if c in model_joints]

    # 1. Disable equality constraints AND contacts for inverse dynamics.
    # The model has scapulohumeral rhythm constraints (11 joint equalities)
    # and self-collision contacts that produce massive constraint forces
    # (~1000+ Nm) contaminating qfrc_inverse. Disabling both yields true
    # physical torques (~5-10 Nm shoulder, ~3-5 Nm elbow) — pure gravity
    # compensation + inertial forces, no constraint artifacts.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
    print(f"   Disabled contacts + {model.neq} equality constraints")

    # 2. Map DOF indices
    # Solve joints: the DOFs fed to the muscle solver (5 DOFs: 3 shoulder + elbow + pro_sup).
    # Excludes deviation/flexion (wrist DOFs with tiny torques that add noise).
    # Scapulohumeral rhythm is manually applied so moment arms are correct.
    solve_dof_indices = map_dof_indices(model, SOLVE_JOINTS)
    solve_joint_names = [j for j in SOLVE_JOINTS if j in model_joints]

    # Output DOF indices: all 7 joints still saved to torques.csv
    output_dof_indices = map_dof_indices(model, OUTPUT_JOINTS)
    output_joint_names = [j for j in OUTPUT_JOINTS if j in model_joints]

    # Build anatomical mask to zero out spurious cross-joint moment arms
    anat_mask = build_anatomical_mask(model, SOLVE_JOINTS, ARM_MUSCLE_COUNT)
    n_allowed = int(anat_mask.sum())
    n_possible = anat_mask.size
    print(f"   Anatomical mask: {n_allowed}/{n_possible} muscle-joint entries allowed")

    # Extract FMA score for spasticity model
    fma_score = extract_fma_score(file_id)

    print(f"   Solve DOFs: {len(solve_dof_indices)} {solve_joint_names}")
    print(f"   Output DOFs: {len(output_dof_indices)} {output_joint_names}")
    print(f"   FMA score: {fma_score} | Arm muscles: {ARM_MUSCLE_COUNT} | "
          f"Fmax scaling: {USE_FMAX_SCALING}")

    # 2. Kinematics
    time = df['time'].values
    dt = np.mean(np.diff(time))
    if dt == 0:
        dt = 0.005
    fs = 1.0 / dt

    qpos = df[valid_cols].values

    # Clamp joint angles to model limits — CVAE-generated motions can
    # exceed model ranges, producing non-physical torques at extreme poses.
    n_clamped = 0
    for c, jname in enumerate(valid_cols):
        jid = model.joint(jname).id
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            before = qpos[:, c].copy()
            qpos[:, c] = np.clip(qpos[:, c], lo, hi)
            n_violations = np.sum(before != qpos[:, c])
            if n_violations > 0:
                n_clamped += n_violations
                print(f"      Clamped {jname}: {n_violations} frames to [{np.degrees(lo):.1f}, {np.degrees(hi):.1f}] deg")
    if n_clamped > 0:
        print(f"   Total: {n_clamped} joint-limit violations clamped")

    qvel, qacc = compute_derivatives(qpos, dt)
    qvel = apply_lowpass_filter(qvel, LOW_PASS_CUTOFF, fs)
    qacc = apply_lowpass_filter(qacc, LOW_PASS_CUTOFF, fs)

    n_frames = len(time)

    # 3. Phase detection
    if PHASE_DETECTION and 'elbow_flexion' in valid_cols:
        ef_col_idx = valid_cols.index('elbow_flexion')
        phase_indices = detect_phases(qpos[:, ef_col_idx], time, fs)
        phase_labels = make_phase_labels(phase_indices, n_frames)
        print(f"   Phases: Pick[0:{phase_indices[1]}] "
              f"Drink[{phase_indices[1]}:{phase_indices[2]}] "
              f"Place[{phase_indices[2]}:{phase_indices[3]}]")
    else:
        phase_indices = [0, n_frames // 3, 2 * n_frames // 3, n_frames]
        phase_labels = make_phase_labels(phase_indices, n_frames)
        print("   Phase detection disabled or elbow_flexion not found; using equal splits")

    # 4. Resolve bottle load body
    bottle_force = BOTTLE_MASS * 9.81
    try:
        load_body_id = model.body(LOAD_BODY).id
    except KeyError:
        print(f"   WARNING: Body '{LOAD_BODY}' not found. Bottle load disabled.")
        load_body_id = None

    # 5. Extract Fmax array for static optimization
    fmax = np.ones(model.nu)
    if USE_FMAX_SCALING:
        for i in range(model.nu):
            # gainprm[2] = peak isometric force for muscle actuators
            f = abs(model.actuator_gainprm[i, 2])
            if f > 0:
                fmax[i] = f

    # 5b. Triceps Fmax boost — encourages solver to recruit triceps during extension
    if TRICEPS_FMAX_BOOST > 1.0:
        muscle_names_list = [model.actuator(i).name for i in range(model.nu)]
        for mi, mname in enumerate(muscle_names_list):
            if mname in ('TRIlat', 'TRImed'):
                fmax[mi] *= TRICEPS_FMAX_BOOST
        print(f"   Triceps Fmax boost: {TRICEPS_FMAX_BOOST}x for TRIlat, TRImed")

    # 6. Dynamics loop — no calibration, raw torques in real Nm
    act_history = np.zeros((n_frames, model.nu))
    tau_history = np.zeros((n_frames, model.nv))
    prev_a = None

    print("   -> Running Dynamics (no calibration, constraints disabled)...")
    for i in tqdm(range(n_frames), leave=False):
        for jname, val, vel, acc in zip(valid_cols, qpos[i], qvel[i], qacc[i]):
            jid = model.joint(jname).id
            data.qpos[model.jnt_qposadr[jid]] = val
            data.qvel[model.jnt_dofadr[jid]] = vel
            data.qacc[model.jnt_dofadr[jid]] = acc

        # Apply scapulohumeral rhythm — set scapulothoracic joints from
        # shoulder_elv/elv_angle so that muscle tendon wrapping is correct.
        # This must happen AFTER setting prescribed joints but BEFORE mj_inverse.
        apply_scapulohumeral_rhythm(model, data)

        # Apply bottle load for ALL phases (cup is in hand throughout)
        data.xfrc_applied[:] = 0
        if load_body_id is not None:
            data.xfrc_applied[load_body_id, 2] = -bottle_force  # Z-down

        mujoco.mj_inverse(model, data)
        tau_raw = data.qfrc_inverse.copy()

        # Use raw torques directly — they're already in real Nm
        tau_history[i] = tau_raw

        # Solve muscle activations (with anatomical mask)
        a = solve_muscle_activations_static_opt(
            model, data, tau_raw, solve_dof_indices,
            fmax=fmax, prev_a=prev_a,
            arm_muscle_count=ARM_MUSCLE_COUNT, weight=SOLVER_WEIGHT,
            anat_mask=anat_mask)
        prev_a = a

        act_history[i] = a

    # Report raw torque ranges (should be physiological: ~5-20 Nm shoulder, ~2-5 Nm elbow)
    for jname, dof_idx in zip(OUTPUT_JOINTS, output_dof_indices):
        peak = np.max(np.abs(tau_history[:, dof_idx]))
        print(f"      {jname}: peak {peak:.2f} Nm")

    # 6b. ATI calibration — scale activations to match healthy baseline BEFORE spasticity
    if ATI_CALIBRATION_ENABLED:
        _n_arm = min(ARM_MUSCLE_COUNT, act_history.shape[1])
        pre_ati = float(np.mean(np.sum(act_history[:, :_n_arm]**2, axis=1)))
        act_history = calibrate_ati(act_history, ARM_MUSCLE_COUNT, ATI_HEALTHY_BASELINE)
        post_ati = float(np.mean(np.sum(act_history[:, :_n_arm]**2, axis=1)))
        print(f"   -> ATI calibration: {pre_ati:.4f} -> {post_ati:.4f} (target={ATI_HEALTHY_BASELINE})")

    # 7. Apply spasticity-driven co-contraction (before filtering)
    if fma_score < 66:
        print(f"   -> Applying spasticity co-contraction (FMA={fma_score}, "
              f"gain={SPASTICITY_BASE_GAIN * (1-fma_score/66)**SPASTICITY_EXPONENT:.3f})...")
        act_history = apply_spasticity_cocontraction(
            act_history, qvel, valid_cols, fma_score, model)

    # 8. Filter and clip activations
    act_history = apply_lowpass_filter(act_history, LOW_PASS_CUTOFF, fs)
    act_history = np.clip(act_history, 0, 1)

    n_arm = min(ARM_MUSCLE_COUNT, model.nu)
    arm_act = act_history[:, :n_arm]
    active_count = np.sum(np.mean(arm_act, axis=0) > 0.05)
    avg_act = np.mean(arm_act)
    print(f"   -> Result: Avg Arm Act = {avg_act:.3f}, Active muscles (>0.05): {active_count}/{n_arm}")

    # 9. Effort metrics (with actual qpos ROM for TRR)
    metrics = compute_effort_metrics(
        tau_history, act_history, model, output_dof_indices,
        output_joint_names, ARM_MUSCLE_COUNT,
        qpos=qpos, valid_cols=valid_cols)
    print(f"   -> Effort: ATI={metrics['ATI']:.4f}, CCI={metrics['CCI']:.3f}")

    # 10. Synergy extraction (arm muscles only)
    if SYNERGY_ENABLED:
        # Inject FMA-scaled noise before NNMF for realistic VAF separation
        if SYNERGY_NOISE_STD > 0:
            act_history = inject_synergy_noise(
                act_history, ARM_MUSCLE_COUNT, fma_score, SYNERGY_NOISE_STD)
            arm_act = act_history[:, :n_arm]  # refresh reference
            eff_std = SYNERGY_NOISE_STD * (0.3 + 0.7 * fma_score / 66.0)
            print(f"   -> Synergy noise: std={eff_std:.4f} (FMA={fma_score})")

        print(f"   -> Extracting {N_SYNERGIES} synergies (NNMF)...")
        syn_weights, syn_coeffs, act_synergy, var_explained = extract_synergies(
            arm_act, N_SYNERGIES)
        print(f"      VAF (variance accounted for): {var_explained:.3f}")

        # Data-driven synergy sweep: find optimal n where VAF >= 90%
        print(f"   -> Synergy sweep (1-8, VAF >= 0.90 threshold)...")
        optimal_n, vaf_curve = sweep_synergies(arm_act, max_n=8, vaf_threshold=0.90)
        print(f"      Optimal synergies: {optimal_n} (VAF curve: "
              + ", ".join(f"{n}:{v:.3f}" for n, v in vaf_curve.items()) + ")")
    else:
        syn_weights = syn_coeffs = act_synergy = None
        var_explained = None
        optimal_n = None
        vaf_curve = None

    # 11. Save outputs — filtered to meaningful DOFs and arm muscles
    all_muscle_names = [model.actuator(i).name for i in range(model.nu)]
    arm_muscle_names = all_muscle_names[:n_arm]

    # Torques — only output joints (7 DOFs)
    tau_output = tau_history[:, output_dof_indices]
    tau_df = pd.DataFrame(tau_output, columns=output_joint_names)
    tau_df.insert(0, 'time', time)
    tau_df.to_csv(os.path.join(output_dir, 'torques.csv'), index=False)

    # Activations — only arm muscles (32)
    act_df = pd.DataFrame(arm_act, columns=arm_muscle_names)
    act_df.insert(0, 'time', time)
    act_df.to_csv(os.path.join(output_dir, 'activations.csv'), index=False)

    # Phase labels
    pd.DataFrame({'time': time, 'phase': phase_labels}).to_csv(
        os.path.join(output_dir, 'phase_labels.csv'), index=False)

    # Effort metrics (include synergy sweep results)
    if optimal_n is not None:
        metrics['optimal_synergies'] = optimal_n
        metrics['vaf_curve'] = {str(k): round(v, 4) for k, v in vaf_curve.items()}
    if var_explained is not None:
        metrics['synergy_vaf_4'] = round(var_explained, 4)
    with open(os.path.join(output_dir, 'effort_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Synergy outputs (arm muscles only)
    if SYNERGY_ENABLED and syn_weights is not None:
        pd.DataFrame(syn_weights, columns=arm_muscle_names,
                     index=[f'Synergy_{i+1}' for i in range(N_SYNERGIES)]).to_csv(
            os.path.join(output_dir, 'synergy_weights.csv'))

        syn_coeff_df = pd.DataFrame(
            syn_coeffs,
            columns=[f'Synergy_{i+1}' for i in range(N_SYNERGIES)])
        syn_coeff_df.insert(0, 'time', time)
        syn_coeff_df.to_csv(os.path.join(output_dir, 'synergy_coefficients.csv'), index=False)

    print(f"   -> Saved to {output_dir}")


# =============================================================================
# BACKWARD-COMPAT WRAPPERS (used by run_generated_pipeline.py)
# =============================================================================

def run_inverse_dynamics():
    """Process the single file specified by module-level MOT_FILE_PATH."""
    process_file(MOT_FILE_PATH, MODEL_XML_PATH, OUTPUT_DIRECTORY)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    if os.path.isdir(INPUT_PATH):
        files = sorted(glob.glob(os.path.join(INPUT_PATH, "*.mot")))
    else:
        files = [INPUT_PATH]

    print(f"Found {len(files)} MOT file(s)")

    for f in tqdm(files, desc="Total"):
        try:
            file_id = os.path.basename(f).replace('.mot', '')
            session_dir = os.path.join(OUTPUT_ROOT, file_id)
            process_file(f, MODEL_XML_PATH, session_dir)
        except Exception as e:
            print(f"Error {f}: {e}")

    # Generate cross-file comparison plots
    from scripts.viz.figures.plot_id_comparison import generate_all
    generate_all(id_base_dir=OUTPUT_ROOT, output_dir=os.path.join(OUTPUT_ROOT, '..', 'plots'))

    print("\nDONE.")
