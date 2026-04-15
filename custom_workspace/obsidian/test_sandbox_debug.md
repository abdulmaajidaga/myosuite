# Test Sandbox Debug: IK Pipeline Fix

**Date:** 2026-04-08
**Status:** Resolved — videos confirmed visually correct.

---

## Background

Gemini created a `test/` sandbox inside `custom_workspace/` to systematically compare all 12 CVAE configurations (Stages 0–3 × DTW/SMOTE/Linear). The sandbox mirrored the full pipeline (generate → IK → video → ID) but all output motions were biomechanically broken — skeletons in wrong positions, IK errors through the roof.

---

## Root Causes

### 1. Missing Reference Pose (Primary — Killed IK Entirely)

**What happened:** `test/src/generation/cvae_generate.py` simplified the generation function and dropped the reference pose step. It did:

```python
recon_unnorm = scaler.inverse_transform(recon_np)
df = pd.DataFrame(recon_unnorm, columns=COLS)  # saved as-is
```

**Why this breaks everything:** The training data stores positions as *deltas from the resting start pose* (frame 0 is all zeros). `inverse_transform` brings you back to that delta-mm space — values like `Sh_y ≈ 0`, `El_y ≈ 5`. The IK solver expects absolute millimeter positions in body space (e.g. `Sh_y ≈ 643mm`, `El_y ≈ 474mm`). Passing near-zero values puts every marker at the model's origin — geometrically impossible for a skeleton — so IK errors exploded to >200mm and joint angles were nonsense.

**The fix:** Add `load_reference_pose()` back and apply it after `inverse_transform`:

```python
ref_pose = load_reference_pose()        # avg absolute starting positions (mm)
data_delta = scaler.inverse_transform(recon_np)
data_abs = data_delta.copy()
for col_idx, col in enumerate(COLS):
    data_abs[:, col_idx] += ref_pose.get(col, 0.0)  # delta → absolute mm
```

Reference pose values (average across all training subjects, absolute body frame mm):

| Joint | x | y | z |
|-------|---|---|---|
| Shoulder | -77.3 | 643.0 | 302.7 |
| Elbow | -188.2 | 474.4 | 41.3 |
| Wrist | -88.7 | 241.2 | 41.1 |
| WrVec | -37.0 | 12.0 | -33.1 |
| Trunk | 0.0 | 0.0 | 0.0 |

Trunk is 0 because training data is chest-normalized (chest = origin).

---

### 2. Missing Reference MOT for IK Initialization

**What happened:** The test config pointed to `test/output/originals/mot/S5_12_1.mot` as the IK initial pose reference, but that directory didn't exist in the sandbox.

**Why this matters:** Without a reference MOT, the IK solver starts each trajectory from an uninitialised joint configuration — much harder to converge, especially for the first frame which anchors the rest.

**The fix:** Symlinked the workspace reference MOT into the expected test location:

```bash
mkdir -p test/output/originals/mot/
ln -sf .../output/originals/mot/S5_12_1.mot test/output/originals/mot/S5_12_1.mot
```

---

### 3. Videos Hardcoded as Skipped in Comparison Suite

**What happened:** `test/run_comparison_suite.py` passes `--skip-video` on every pipeline call:

```python
pipeline_cmd = [..., "--skip-video"]   # line 103, hardcoded
```

This is intentional for batch speed (51 FMA scores × 12 configs = 612 runs). Videos were never deleted — they were just never created. The existing workspace videos came from a prior separate pipeline run.

**Resolution:** No change needed for batch runs (speed is correct). Generate individual verification videos by calling `run_generated_pipeline.py` directly without `--skip-video`.

---

## IK Error After Fix

| | Workspace (v2/v3 models) | Test (stage0 model) |
|---|---|---|
| Originals | ~18mm | — |
| Generated | ~24–30mm | **54.8mm** |
| Threshold (retry) | 100mm | 100mm |
| Threshold (fail) | 200mm | 200mm |

The 54.8mm is higher than the workspace baseline but well within acceptable range. The gap is architectural — stage-0 is the simplest model (baseline LSTM CVAE, no CFG). Expect stage-3 + SMOTE to converge closer to 24–30mm once that sweep runs.

---

## Files Changed

| File | Change |
|------|--------|
| `test/src/generation/cvae_generate.py` | Added `load_reference_pose()`, `smooth_trajectory()`, applied delta→absolute conversion before saving |
| `test/output/originals/mot/S5_12_1.mot` | Symlink created → workspace reference MOT |

---

## Verification

```bash
# Generate one file
/home/abdul/miniconda3/envs/MyoSuite/bin/python test/src/generation/cvae_generate.py \
  --fma 50 --stage 0 --data-source dtw \
  --output-dir test/output/verification/csv

# Run full pipeline with video
/home/abdul/miniconda3/envs/MyoSuite/bin/python test/scripts/run_generated_pipeline.py \
  FMA_50.csv \
  --input-dir /path/to/test/output/verification/csv \
  --output-dir /path/to/test/output/verification

# Video output: test/output/verification/videos/FMA_50.mp4
```

> Note: `--input-dir` and `--output-dir` must be **absolute paths** when calling `test/scripts/run_generated_pipeline.py` from the `custom_workspace/` directory. The script prepends its own project root (`test/`) to any relative paths, causing a double-prefix bug.
