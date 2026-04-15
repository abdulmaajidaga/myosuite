# Known Limitations

## 1. Temporal Scaling — No Ground Truth Beyond FMA 20

**What it is:**
The pipeline uses `src/data_processing/temporal_scaling.py` to stretch the CVAE's fixed 100-frame output to a realistic real-world duration. It learns an FMA → duration mapping from the actual MHH dataset.

**The problem:**
The dataset only contains stroke patients in the **FMA 16–20 range** (9 patients). There are no recordings of patients with FMA 21–65. So:

| FMA Range | Duration Source |
|-----------|----------------|
| 16 | 4.21s — 1 patient, measured |
| 17 | 6.12s — 3 trials, measured |
| 18 | 4.16s — 3 trials, measured |
| 19 | 2.97s — 12 trials, measured |
| 20 | 2.62s — 4 trials, measured |
| **21–65** | **Interpolated** between FMA 20 (2.62s) and FMA 66 (3.40s) |
| 66 | 3.40s — 53 healthy trials, measured |

The interpolation range for FMA 21–65 is only **0.78 seconds** (2.62s → 3.40s), so every generated motion in that range lands at approximately **3 seconds** regardless of impairment severity.

**Implication for the paper:**
- Video durations for FMA 16–20 vary meaningfully and reflect real patient data
- Video durations for FMA 21–66 are effectively constant (~3s) and should not be interpreted as a validated temporal model
- The kinematic content (joint angles, trajectories) is still meaningful — only the duration is affected
- This is a direct consequence of the narrow FMA range in the MHH cohort (all stroke patients were mild–moderate, FMA 16–20)

**Where to address in the paper:**
- §3.1 (Dataset) — note the FMA range of the MHH stroke cohort
- §4.x (Temporal Scaling) — state that temporal extrapolation beyond FMA 20 is linear interpolation, not data-driven
- §6.3 (Limitations) — flag as a direction for future work requiring a wider-severity cohort

---

## 2. FMA Augmentation Beyond Observed Range

**What it is:**
The CVAE generates motions for FMA scores 21–65 which do not exist in the training data. The augmentation methods (DTW, SMOTE, Linear) create synthetic training samples by interpolating between FMA 16–20 and FMA 66 (healthy). The model is therefore extrapolating into unseen clinical territory.

**Implication:**
Generated motions for mid-range FMA (21–65) are plausible interpolations but have no ground-truth validation. Biomechanical validity (CCI, literature comparison) is the only available proxy for correctness in this range.
