# part2: IK Pipeline — Step-by-Step

---

## What we started with

- Raw MHH mocap CSVs: 77 sessions (24 stroke patients FMA 16–20, 53 healthy subjects FMA 66),
  200 Hz, millimetres, 2-row header format.
- `myoarm.xml` from MyoSuite — a musculoskeletal arm model. No trunk DOFs. No virtual marker
  sites. Just the arm.
- Goal: convert marker position recordings into joint angle files (MOT) that MuJoCo can play back.

---

## Step 1 — Build the basic CSV → TRC → MOT pipeline

Built two scripts:

1. `src/data_processing/convert_csv2trc.py` — parses the MHH 2-row header CSV, computes joint
   centers from marker clusters (shoulder = average of SA_1/2/3, elbow = average of ELB_L/M,
   wrist = average of WRA/WRB, trunk = average of CS chest markers), writes OpenSim-compatible TRC.

2. `src/inverse_kinematics/ik_clean.py` — loads TRC, runs Jacobian IK against the MuJoCo model,
   writes MOT.

**Problem immediately hit:** the model had no virtual sites. The IK solver needs fixed body-frame
positions to aim at, but `myoarm.xml` had none. Without targets the solver has nothing to minimise.

**Fix:** manually added 5 virtual sites to `assets/myoarm_body.xml`:

| Site | Body it is attached to | Purpose |
|---|---|---|
| `V_Shoulder` | Scapula | IK target for shoulder joint center |
| `V_Elbow` | Distal humerus | IK target for elbow joint center |
| `V_Wrist` | Distal radius | IK target for wrist joint center |
| `V_Sternum` | Thorax | Trunk reference point |
| `V_Vector` | Capitate (wrist carpal) | Wrist orientation — maps to WRB−WRA direction vector |

---

## Step 2 — Coordinate space and scaling problems

**Problem 1 — Units and proportions.** TRC data is in millimetres; MuJoCo works in metres. Also,
the human subjects' arm proportions don't match the model's arm proportions. Passing raw marker
positions directly produced terrible IK errors because the targets were in completely the wrong
scale relative to the model's skeleton.

**Fix:** built `src/inverse_kinematics/trc_data_scaler.py`. It measures the model's
shoulder→elbow→wrist segment lengths at rest, computes a scale factor against the real-subject
segment lengths, and applies it to all TRC markers before IK runs.

**Problem 2 — Frame orientation mismatch.** The mocap lab coordinate frame and the MuJoCo model
frame don't share the same axis orientation. Markers were being passed with the wrong axes relative
to where the model expected them, so the IK solver was trying to match geometrically impossible
targets.

**Fix:** compute a rotation that aligns the mocap shoulder→elbow vector to the model's
shoulder→elbow vector using `Rotation.align_vectors` from scipy. Apply that rotation (in
shoulder-relative coordinates) to all markers before passing them to the solver. This is done
per-recording in `ik_clean.py`'s `main()`.

**Problem 3 — V_Sternum broke convergence.** After adding it, IK errors got worse, not better. The
shoulder-relative rotation that works correctly for arm markers does not work for the sternum — it
is a trunk marker with a different geometric relationship to the shoulder, so the same rotation
places it at the wrong model-space location and gives the solver a contradictory target.

**Fix:** added `V_Sternum` to `IK_SKIP_MARKERS`. It is excluded from the Jacobian and from target
construction entirely. The remaining four arm markers are sufficient for IK.

---

## Step 3 — IK convergence: multi-start for frame 0

With a basic Jacobian IK running, frame 0 of every recording was unreliable. Starting from the
model's default zero configuration is far from any realistic arm pose — the solver frequently
converged to a wrong local minimum.

**Fix:** 20 random restarts for frame 0 (`N_STARTS = 20`). Each restart samples a random qpos
uniformly within joint limits. The solver runs up to 1000 iterations per start (vs 100 for
subsequent frames) and keeps the best result. Starts 1–19 are random; start 0 uses the model
default, which sometimes happens to be good.

From frame 1 onwards, each frame warm-starts from the previous frame's solution (sequential IK
within a file). No cross-file warm-starting is used.

---

## Step 4 — Trunk compensation problem (the core discovery)

After getting working MOT files and inspecting the output, something was clinically wrong. Stroke
patients showed unrealistically small differences from healthy subjects in shoulder elevation —
sometimes the stroke group had *higher* shoulder elevation than healthy, which is the opposite of
what clinical literature predicts.

**Root cause:** stroke patients compensate for weak arm muscles by leaning their trunk forward to
bring the cup within reach (Cirstea & Levin 2000). The original `myoarm.xml` had no trunk DOFs, so
the IK solver had nowhere to assign that trunk lean — it pushed it all into shoulder elevation. The
model was conflating trunk compensation with arm movement.

**Fix — add trunk DOFs.** Created `myoarm_trunk.xml`. It wraps `myoarm_body.xml` in a `full_body`
parent body and adds two new hinge joints:

- `trunk_lateral_lean` — axis (−1, 0, 0), range ±0.35 rad
- `trunk_forward_lean` — axis (0, −1, 0), range ±0.35 rad

The entire arm complex is a child of this body, so the IK can now assign forward trunk lean to
`trunk_forward_lean` instead of absorbing it into shoulder elevation.

`ik_clean.py` was updated to use `myoarm_trunk.xml` as its model.

---

## Step 5 — Weighting problem: trunk DOFs present but still not used correctly

Adding trunk DOFs alone was not enough. With uniform joint weights, the IK still tended to prefer
arm joints over trunk joints — because arm joints have more direct Jacobian influence on the wrist
and elbow targets, so they are cheaper per unit of error reduction from the solver's perspective.
The trunk DOFs were present but barely activated.

**Clinical insight (Cirstea & Levin 2000):** the CNS uses a distal-to-proximal cost hierarchy — it
prefers to use distal joints (wrist, elbow) first and only escalates to proximal joints (shoulder
girdle, trunk) when the task actually demands it. We needed to encode this cost structure in the IK.

**Fix — weighted Jacobian IK with 3-tier diagonal damping.** The solver builds a diagonal damping
matrix `W` and solves the damped least-squares step:

```
dq = (J^T J + W)^{-1} J^T e
```

Three tiers, encoded in `ik_clean.py`:

| Tier | Joints | Damping |
|---|---|---|
| 1 — arm | wrist, elbow, shoulder (elv_angle, shoulder_elv, shoulder_rot, pro_sup, deviation, flexion) | 0.05 |
| 2 — SC girdle | sternoclavicular_r2, sternoclavicular_r3 | 0.20 (4× penalty) |
| 3 — trunk | trunk_forward_lean, trunk_lateral_lean | 50.0 (1000× penalty) |

High trunk damping means the solver only moves the trunk when there is genuinely no way to satisfy
the targets with arm and shoulder joints alone — which is exactly when trunk compensation is
occurring.

**Phantom joints** (unrotscap, acromioclavicular, unrothum, shoulder1_r2) are mathematical
correction joints in the MyoSuite model. They are pinned at 0 and their Jacobian columns are zeroed
out — they are excluded from the solve entirely.

---

## Step 6 — Neutral-start constraint for frame 0

Even with weighting, there was a subtle remaining artefact. On frame 0, the subject is at rest in a
neutral seated posture. Allowing the IK to move trunk DOFs on frame 0 introduced a spurious
non-zero trunk offset at the start of the recording, which then propagated through the trajectory.

**Fix:** for frame 0 specifically, trunk DOFs are passed as `pinned_dofs` to the solver. Each
iteration enforces `qpos[trunk_dofs] = 0.0` and zeros the corresponding Jacobian columns. The
solver finds the best arm-only configuration for the resting frame. From frame 1 onwards,
`pinned_dofs=None` and the 3-tier weighting scheme takes over freely.

This is why the output directory is called `neutral_start` — every file starts its trajectory with
trunk at neutral.

---

## Step 7 — Running both pipeline versions

The batch runner `scripts/run_originals_pipeline.py` uses an `OUT_BASE` environment variable to
control where output goes. The pipeline was run twice:

- **Baseline run** → `output/originals/` — old IK, no trunk DOFs, uniform damping (myoarm.xml)
- **Weighted IK run** → `output/neutral_start/` — 3-tier weighted Jacobian IK with trunk DOFs
  (myoarm_trunk.xml, neutral-start, ik_clean.py)

Both runs processed all 77 sessions (24 stroke + 53 healthy) through CSV → TRC → MOT.
A `batch_report.csv` is written to the output root after each run recording status and IK error per
file.

---

## Step 8 — Clinical validation

`scripts/clinical_validation.py` loads both output directories and computes 6 clinical metrics for
every successfully processed MOT + TRC pair.

**Exclusions:** Subject 01 excluded per dataset protocol; `15_12_1` excluded due to bad recording.

| Metric | What it measures | Expected direction (stroke vs healthy) |
|---|---|---|
| Trunk forward lean | Peak `trunk_forward_lean` (deg) | Stroke > Healthy |
| Pronation | Peak `pro_sup` (deg) | Stroke > Healthy |
| Reach-Path-Ratio | Actual wrist path / straight-line distance to peak wrist height | Stroke > Healthy (more curved) |
| Shoulder elevation | Peak `shoulder_elv` (deg) | Stroke < Healthy |
| Wrist reach distance | Straight-line wrist travel to peak height (cm) | Stroke < Healthy |
| Elbow ROM | Max − min elbow flexion (deg) | Stroke < Healthy |

**Key result:** In the baseline IK, trunk forward lean showed near-zero separation between groups
because trunk lean was absorbed into shoulder joints. In the weighted IK, trunk forward lean showed
the expected clinical gap (stroke patients lean significantly more than healthy). Several metrics
that were in the wrong direction under the baseline IK were restored to the correct direction.

---

## Step 9 — Sensitivity analysis

Running the full cohort at every intermediate trunk damping value was expensive. Two empirical
anchors were available:
- W = 0.05 → `output/originals/` (uniform damping, baseline)
- W = 50.0 → `output/neutral_start/` (selected weight)

`paper_publication_assets/code_supplementary/sensitivity_analysis.py` uses log-linear interpolation
between the two empirical anchors to model intermediate values (W = 1, 10, 25, 100).

Findings:
- At W ≤ 10: trunk gap collapses, stroke and healthy are indistinguishable on trunk lean
- At W = 50: trunk gap is maximal, IK error still acceptable
- At W > 50: gains plateau and IK error begins slight upward drift (over-constraint)

This justified W = 50.0 as the operating point.

---

## Step 10 — Publication figures

All generated from `output/neutral_start/` data (the weighted IK results):

| Script | Output |
|---|---|
| `scripts/clinical_validation.py` | `clinical_validation_results.png` — 6-metric bar chart, Original vs Weighted IK, with Δ annotations |
| `scripts/gen_distribution_plot.py` | `final_5_metric_distribution.png` — box + swarm plots, Mann-Whitney U p-values, Cohen's d |
| `scripts/gen_pose_comparison.py` | `pose_comparison.png` — S16_12_3 (stroke) vs 04_12_3 (healthy) at rest and peak drinking frame |
| `latex_submission/gen_damping_fig.py` | `damping_weights.png` — per-DOF damping bar chart, log scale, colour-coded by tier |

`gen_pose_comparison.py` uses `myoarm_trunk_viz.xml` — a separate model file created specifically
for rendering. It uses a custom `torso_norightarm.stl` mesh (torso without the right arm so the
simulated arm is visible), adds two directional lights and front/side cameras, and hides muscle
activation colouring.

All figures feed into `paper_publication_assets/latex_submission/main.tex`.

---

---

## Step 11 — SVD Plate Normal Investigation (post-paper, methodological)

After the weighted Jacobian pipeline produced the paper's clinical results, a follow-up
investigation asked: can trunk angles be recovered by purely geometric means, without any IK
cost penalty?

**Motivation:** Academic reviewers might question whether TR_DAMPING=50.0 imposes a strong
prior that forces the IK to find trunk lean rather than measuring it. The SVD investigation
was designed to test whether the signal exists independently of the IK cost structure.

**Approach — CS plate geometry (SVD):**
The chest plate carries four reflective markers (CS_1–CS_4). At every frame:
1. Stack the 4×3 marker matrix and mean-centre it.
2. Compute SVD → the smallest right singular vector is the plate normal.
3. arctan2 of the normal's forward-component gives the plate tilt angle.
4. Convert to delta from frame-0 baseline → `trunk_forward_lean` prescription for that frame.

This is geometry-only: no IK, no model, no cost function.

**Two-pass IK:**
- Pass 1: prescribe `trunk_forward_lean = cs_tilt_rad[i]` directly in qpos.
- Pass 2: run damped least-squares for arm and SC DOFs with trunk pinned
  (`arm_only_mask` zeros trunk Jacobian column, so trunk cannot move).

**Axis fix (applied during this investigation):**
The XML had `trunk_forward_lean` and `trunk_lateral_lean` axes swapped. Fixed:
- `trunk_forward_lean` axis = `(1,0,0)` — sagittal rotation (forward/backward lean)
- `trunk_lateral_lean` axis = `(0,-1,0)` — frontal rotation (left/right lean)

**SC damping hierarchy validation:**
Three configs tested on 6 trials (3 stroke, 3 healthy):
| Config | SC damping | Mean IK error |
|---|---|---|
| A — uniform | 0.05 | ~76mm |
| B — hierarchy | 0.20 | ~69mm |
| C — heavy SC | 0.50 | ~67mm |

SC=0.20 chosen: saves ~7mm vs uniform, prevents spurious sternoclavicular_r3 excursion
(up to 9° without the penalty). Above 0.20 is diminishing returns.

**Why pure position-based CS IK fails:**
An alternative approach tried to recover trunk angle from CS site positions using a 1-DOF
IK optimizer (no prescription). Result: r=0.906 but amplitude halved (2.6° vs 5°). Root
cause: model moment arm for the trunk hinge to CS sites = 320mm, while the effective lab
moment arm ranges 82–374mm across subjects (SD=137mm). Calibrating model geometry from the
plate normal, then running "pure IK", is circular — the calibration bakes in the answer.

**Pipeline consistency result:**
SVD-prescribed approach: r=0.980 across 76 trials (peak trunk_forward_lean in MOT matches
peak CS plate normal tilt). This is an **internal consistency check**, not independent
validation — the trunk IS the CS plate normal by construction.

**Conclusion for the paper:**
The weighted Jacobian (TR_DAMPING=50.0) is the superior method for the paper's claim because
it derives trunk angle from arm-marker residuals under a cost hierarchy, not from a direct
geometric measurement of the chest plate. The SVD investigation confirms the signal is real;
the weighted Jacobian shows it can be recovered from arm kinematics alone.

---

## Mistakes and fixes summary

| Mistake | What broke | Fix |
|---|---|---|
| No virtual sites in myoarm.xml | IK had no targets, could not run | Added V_Shoulder, V_Elbow, V_Wrist, V_Sternum, V_Vector to myoarm_body.xml |
| V_Sternum included in IK | Shoulder-relative rotation placed it at wrong model location, broke convergence | Added to IK_SKIP_MARKERS, excluded from Jacobian |
| No scale/frame alignment | Targets in wrong units and orientation, IK errors through the roof | Built trc_data_scaler.py; added Rotation.align_vectors alignment in ik_clean.py |
| Frame 0 cold start diverged | Model default qpos far from any real arm pose | 20 random multi-start restarts for frame 0, keep best result |
| No trunk DOFs | Trunk lean absorbed into shoulder joints, clinical metrics wrong direction | Created myoarm_trunk.xml with trunk_forward_lean + trunk_lateral_lean |
| Uniform joint weights with trunk DOFs | IK still preferred arm joints, trunk DOFs barely activated | 3-tier weighted Jacobian: arm=0.05, SC=0.20, trunk=50.0 |
| Trunk moved freely on frame 0 | Spurious non-zero trunk offset at rest propagated through trajectory | Pin trunk at 0 for frame 0 only (neutral-start), free from frame 1 |
