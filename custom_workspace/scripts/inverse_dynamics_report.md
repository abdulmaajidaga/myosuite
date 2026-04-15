# Inverse Dynamics Pipeline: Model Analysis and Solver Improvements

## Overview

This report documents the investigation into the MuJoCo musculoskeletal model's muscle routing, the discovery of spurious cross-joint moment arms, and the implementation of two corrections — scapulohumeral rhythm and anatomical masking — that improved EMG validation from 6/30 PASS to 17/30 PASS.

---

## 1. Problem Statement

The inverse dynamics pipeline decomposes joint torques into muscle activations using static optimization. The original results showed wrong dominant muscles:

- **Before**: PQ, DELT3, PT, PL, FCU dominated all phases
- **Expected (literature)**: DELT1, DELT2, BIClong for Reach; BIClong, BICshort, BRA for Drink; TRIlong, TRIlat for Place

EMG validation scored **6 PASS / 6 WARN / 18 FAIL** out of 30 checks across 6 FMA-scored sessions.

## 2. Model Analysis

### 2.1 Model Architecture

The MuJoCo arm model (`models/model/myo_sim/arm/myoarm.xml`) is based on the Stanford/Wu ISB shoulder model. Key features:

- **63 muscle actuators** as spatial tendons with wrapping geometries (ellipsoids, cylinders, tori)
- **Two actuator classes**: `general` (shoulder, full Hill-type muscle dynamics) and `muscle` (forearm, simpler force model)
- **Phantom body kinematic chain** for Euler angle decomposition of shoulder rotation:

```
thorax
  └─ clavicle (sternoclavicular_r2, r3)
       └─ clavphant (unrotscap_r2, r3)
           └─ scapula (acromioclavicular_r1, r2, r3)
               └─ scapphant (unrothum_r1, r2, r3)
                   └─ humphant (elv_angle)
                       └─ humphant1 (shoulder_elv, shoulder1_r2)
                           └─ humerus (shoulder_rot)
                               └─ ulna (elbow_flexion)
                                   └─ radius (pro_sup)
                                       └─ hand (deviation, flexion)
```

- **11 equality constraints** implementing scapulohumeral rhythm — linear coupling of scapulothoracic joints to `shoulder_elv` and `elv_angle`

### 2.2 Muscle Routing Verification

Tendon attachment sites were traced to their parent bodies to verify anatomical correctness:

| Muscle | Origin Body | Insertion Body | Wrapping | Anatomically Correct? |
|--------|------------|----------------|----------|----------------------|
| DELT1 (anterior) | clavicle (P4), scapula (P3) | humerus (P1, P2) | humerus ellipsoid | Yes |
| DELT2 (middle) | scapula (P3, P4) | humerus (P1, P2) | humerus cylinder, scapula ellipsoid | Yes |
| DELT3 (posterior) | scapula (P1, P2) | humerus (P3) | humerus ellipsoid | Yes |
| BIClong | scapula (P1, P2) | radius (P10, P11) | humerus via-points, elbow ellipsoid | Yes |
| BRA | humerus (P1) | ulna (P4) | humerus TRI_cylinder | Yes |
| PQ | radius (P1) | ulna (P2) | ulna wrapping | Yes |

**Conclusion**: The model's muscle routing is anatomically correct. The issue lies in how MuJoCo computes moment arms through the phantom body chain.

### 2.3 Scapulohumeral Rhythm Constraints

All 11 equality constraints are linear (`polycoef = [0, k, 0, 0, 0]` means `joint1 = k * joint2`):

| Scapulothoracic Joint | Driver | Coefficient |
|------------------------|--------|-------------|
| sternoclavicular_r2 | shoulder_elv | -0.242 |
| sternoclavicular_r3 | shoulder_elv | +0.1025 |
| unrotscap_r2 | shoulder_elv | +0.242 |
| unrotscap_r3 | shoulder_elv | -0.1025 |
| acromioclavicular_r1 | shoulder_elv | +0.178 |
| acromioclavicular_r2 | shoulder_elv | -0.049 |
| acromioclavicular_r3 | shoulder_elv | +0.396 |
| unrothum_r1 | shoulder_elv | -0.178 |
| unrothum_r2 | shoulder_elv | +0.049 |
| unrothum_r3 | shoulder_elv | -0.396 |
| shoulder1_r2 | elv_angle | -1.0 |

## 3. Root Cause Analysis

### 3.1 Issue 1: Disabled Scapulohumeral Rhythm

Equality constraints were disabled to prevent massive constraint forces (~1000+ Nm) from contaminating `qfrc_inverse`. However, this left scapulothoracic joints at zero regardless of shoulder position. Since muscles like DELT1 route from clavicle/scapula to humerus, their tendon paths (and moment arms) depend on correct scapula positioning.

**Impact**: DELT1's direct moment arm at `shoulder_elv` was only 0.004m (4mm) with a frozen scapula. With correct scapula positioning, it should be ~0.019m (19mm).

### 3.2 Issue 2: Spurious Cross-Joint Moment Arms

MuJoCo's `data.actuator_moment` (computed as `∂tendon_length/∂joint_angle`) produces non-zero entries at joints a muscle does not anatomically cross:

| Muscle | Spurious Joint | Moment Arm | Anatomically Possible? |
|--------|---------------|------------|----------------------|
| BRA (brachialis) | acromioclavicular_r3 | 65 mm | No — BRA only crosses elbow |
| INFSP (infraspinatus) | elbow_flexion | 14 mm | No — INFSP only crosses shoulder |
| INFSP | pro_sup | 3 mm | No |
| DELT1 | flexion (wrist) | 4 mm | No |
| DELT1 | cmc_abduction | 3 mm | No |

These are artifacts of the phantom body kinematic chain interacting with the spatial tendon wrapping geometry computation. They are not numerical noise — BRA at 65mm is a substantial value that would dominate the muscle solver.

## 4. Implemented Solutions

### 4.1 Scapulohumeral Rhythm Application

**File**: `src/inverse_dynamics/calc_mot2invdyn.py`

Before computing torques (`mj_inverse`) and moment arms (`mj_forward`), the scapulothoracic joints are set from their driving joints using the linear coefficients:

```python
SCAP_RHYTHM = {
    'sternoclavicular_r2': ('shoulder_elv', -0.242),
    'sternoclavicular_r3': ('shoulder_elv',  0.1025),
    # ... (11 entries total)
}

def apply_scapulohumeral_rhythm(model, data):
    for scap_joint, (driver_joint, coeff) in SCAP_RHYTHM.items():
        scap_val = coeff * data.qpos[driver_qpos_addr]
        scap_val = np.clip(scap_val, joint_range_lo, joint_range_hi)
        data.qpos[scap_qpos_addr] = scap_val
```

This ensures correct scapula positioning for tendon wrapping without the constraint solver forces.

### 4.2 Anatomical Muscle-Joint Mask

**File**: `src/inverse_dynamics/calc_mot2invdyn.py`

A binary mask `(n_solve_dofs, n_arm_muscles)` zeros out moment arms for joints a muscle does not anatomically cross:

```python
MUSCLE_JOINT_MAP = {
    # Shoulder-only: scapula/clavicle/thorax -> humerus
    'DELT1':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    'DELT2':   ['elv_angle', 'shoulder_elv', 'shoulder_rot'],
    # ... 15 shoulder muscles

    # Biarticular: scapula -> radius/ulna
    'BIClong': ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion', 'pro_sup'],
    'TRIlong': ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion'],
    'TRIlat':  ['elbow_flexion'],
    # ... 5 biarticular muscles

    # Elbow/forearm
    'BRA':     ['elbow_flexion'],
    'PT':      ['elbow_flexion', 'pro_sup'],
    'PQ':      ['pro_sup'],
    # ... 12 forearm muscles
}
```

Applied in the solver before optimization: `M_geom *= anat_mask`

Result: 78 out of 160 muscle-joint entries are allowed (49%). The remaining 51% are zeroed out as anatomically impossible.

### 4.3 Pro_sup Restored to Solve DOFs

**File**: `config/settings.yaml`

Pronation/supination was added back to the solve DOFs (previously removed as a workaround). With the anatomical mask preventing forearm muscles from spuriously activating at shoulder DOFs, pro_sup can be included without contaminating the results.

```yaml
solve_joints: ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion', 'pro_sup']
```

### 4.4 Approaches Attempted but Rejected

#### 4.4.1 Scapulohumeral Coupling Correction (Effective Moment Arms)

Attempted to compute "effective" moment arms that account for the coupled scapulothoracic joints:

```
M_eff[shoulder_elv, muscle] = M[shoulder_elv, muscle] + sum(coeff_k * M[scap_k, muscle])
```

**Result**: LAT1-3 and PECM1-3 dominated because the coupling amplified their already-spurious cross-joint entries. Rejected.

#### 4.4.2 Finite-Difference Moment Arms

Computed moment arms by perturbing each solve DOF and measuring tendon length change, with scapulohumeral rhythm applied at each perturbed pose:

```python
M[j, m] = -(L(q_j + eps) - L(q_j - eps)) / (2 * eps)
```

**Result**: Coupled moment arms were so large that activations dropped to near-zero (avg 0.016). The torques (computed without coupling) were too small relative to the muscles' coupled force-production capacity. Rejected due to inconsistency between torques and moment arms.

## 5. Results

### 5.1 EMG Validation

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| PASS | 6 | 17 | +183% |
| WARN | 6 | 10 | +67% |
| FAIL | 18 | 3 | -83% |

#### Per-Phase Muscle Accuracy (Overlap with Published Literature)

| Phase | Before (avg) | After (avg) | Expected Muscles |
|-------|-------------|-------------|-----------------|
| Reach/Pick | 0-20% | **80%** | DELT1, DELT2, BIClong, BICshort, SUPSP |
| Drink | 0-25% | **75%** | BIClong, BICshort, BRA, DELT1 |
| Place | 25% | **25-50%** | TRIlong, TRIlat, DELT2, DELT3 |

**Reach and Drink phases** now correctly identify deltoids, biceps, and supraspinatus as the dominant muscles. **Place phase** is the remaining weakness — TRIlong/TRIlat have small direct moment arms at elbow_flexion in this model.

### 5.2 Effort Metrics

| Session | ATI | CCI | Active Muscles (>0.05) |
|---------|-----|-----|----------------------|
| FMA_18 (severe) | 0.788 | 0.359 | 21/32 |
| FMA_20 (severe) | 0.861 | 0.244 | 20/32 |
| FMA_30 (moderate) | 0.591 | 0.301 | 22/32 |
| FMA_40 (moderate) | 0.666 | 0.254 | 21/32 |
| FMA_50 (mild) | 0.756 | 0.375 | 21/32 |
| FMA_66 (near-healthy) | 0.682 | 0.298 | 18/32 |
| **01_12_1 (healthy)** | **0.199** | **0.379** | **17/32** |

**Key differentiation**: Healthy subject (ATI=0.199) uses 3-4x less muscular effort than generated FMA motions (ATI=0.59-0.86). This is consistent with the clinical expectation that stroke patients require more co-contraction and compensatory muscle recruitment.

### 5.3 Joint Torques

All torques are in physiological range (no calibration needed):

| Joint | Healthy Peak (Nm) | Generated Range (Nm) |
|-------|-------------------|---------------------|
| elv_angle | 11.1 | 6.6 - 16.2 |
| shoulder_elv | 1.3 | 2.7 - 4.6 |
| shoulder_rot | 0.4 | 2.6 - 3.4 |
| elbow_flexion | 4.6 | 3.6 - 4.1 |
| pro_sup | 0.5 | 1.3 - 1.6 |

Generated motions show higher shoulder torques than healthy, consistent with compensatory movement strategies in stroke.

## 6. Remaining Limitations

1. **Place phase triceps**: TRIlat and TRImed have small direct moment arms at elbow_flexion in this model. This is a model geometry limitation — the spatial tendon wrapping produces correct but small values for these muscles at the elbow.

2. **CCI not monotonic with FMA**: Co-Contraction Index ranges 0.24-0.38 across FMA scores without a clear impairment gradient. The static optimization solver minimizes total activation (L2 norm), which inherently reduces co-contraction. A forward dynamics approach would better capture spasticity-driven co-contraction.

3. **DELT1 underrepresented**: Anterior deltoid has a smaller direct moment arm at shoulder_elv (4mm) than anatomically expected (~15-25mm). The phantom body chain distributes DELT1's torque contribution across scapulothoracic DOFs that are not included in the solve set.

4. **Model designed for forward dynamics**: The MyoSuite model is optimized for RL-based control (forward dynamics), where the constraint solver handles scapulohumeral rhythm naturally. Inverse dynamics requires manual workarounds (rhythm application, anatomical masking) to produce meaningful results.

## 7. Files Modified

| File | Changes |
|------|---------|
| `src/inverse_dynamics/calc_mot2invdyn.py` | Added `SCAP_RHYTHM`, `apply_scapulohumeral_rhythm()`, `MUSCLE_JOINT_MAP`, `build_anatomical_mask()`, `compute_coupled_moment_arms()`, `_build_coupling_map()`. Updated solver to use anatomical mask. |
| `config/settings.yaml` | Changed `solve_joints` to include `pro_sup`. Added comments about scapulohumeral rhythm. |

## 8. How to Reproduce

```bash
cd /home/abdul/Desktop/myosuite/custom_workspace

# Process all generated FMA motions through IK + ID
python scripts/run_generated_pipeline.py

# Process a single file
python scripts/run_generated_pipeline.py FMA_50.csv

# Run batch ID on originals
python src/inverse_dynamics/calc_mot2invdyn.py

# Run EMG validation
python scripts/viz/validate_emg.py

# Generate comparison plots
python -c "
from src.visualization.plot_id_comparison import generate_all
generate_all()
"
```

## 9. References

- Alt Murphy M et al. (2011). Kinematic analysis of the upper extremity in the drinking task.
- Roh J et al. (2013). Alterations in upper limb muscle synergy structure in chronic stroke survivors.
- Seth A et al. (2010). A biomechanical model of the scapulothoracic joint. Stanford University.
- Saul KR et al. (2015). Benchmarking of dynamic simulation predictions in two software platforms using an upper limb musculoskeletal model.
