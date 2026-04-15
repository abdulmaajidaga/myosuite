# Dataset Statistics — Reference

*Last updated: 2026-04-10*

---

## 1. MHH Real Motion Capture Dataset

### Recording protocol
- Task: Drinking task (Pick up cup → Drink → Return cup to table)
- Marker set: Shoulder, Elbow, Wrist, WristVector, Trunk (5 sites × 3 axes = 15 channels)
- Sampling rate: 200 Hz
- Resampled to: 100 frames (fixed-length) for CVAE training

### Subject breakdown

| Group | Sessions | FMA range | FMA mean |
|---|---|---|---|
| Stroke patients | 24 sessions (S-prefix files) | 16–20 | ~18.6 |
| Healthy subjects | 53 sessions | 66 (all healthy) | 66 |
| **Total** | **77 sessions** | **16–20, 66** | — |

### FMA distribution (stroke patients only)

| FMA score | Sessions |
|---|---|
| 16 | 2 |
| 17 | 3 |
| 18 | 3 |
| 19 | 12 |
| 20 | 4 |
| **Total stroke** | **24** |

**Critical gap:** No patients with FMA 21–65 (moderate/mild severity). The model extrapolates across the entire moderate-severity range from two endpoint groups only (severe: FMA 16–20, healthy: FMA 66).

---

## 2. Augmented Training Datasets

All augmentation methods interpolate between stroke (FMA 16–20) and healthy (FMA 66) sessions to synthesise training samples for the full FMA 16–66 range.

| Method | Files | Files per FMA level | Notes |
|---|---|---|---|
| SMOTE | 58,303 | ~1,140 | SMOTE-NC: new minority-class samples via k-NN interpolation |
| DTW | 59,172 | ~1,160 | Dynamic Time Warping alignment before linear interp |
| Linear | 56,574 | ~1,109 | Direct linear interpolation between mean stroke/healthy |

**Training cap applied during test/ experiments:** 15,000 files per run (random subsample), to keep training time manageable (~14 min/experiment on RTX 4070 Laptop).

---

## 3. Generated Output

- **FMA levels generated:** 51 (integer scores 16–66 inclusive)
- **Sessions per FMA level:** 1 (default), 10 (n=10 averaging for litval)
- **Format:** 100 frames × 15 channels CSV (pre-IK), then MOT (post-IK), then activations/torques (post-ID)

---

## 4. Temporal Scaling

Duration of real MHH recordings by FMA group (used for temporal scaling of generated motions):

| FMA | Mean duration | N trials | Source |
|---|---|---|---|
| 16 | 4.21s | ~few | Measured (real patients) |
| 17 | 6.12s | 3 | Measured |
| 18 | 4.16s | 3 | Measured |
| 19 | 2.97s | 12 | Measured |
| 20 | 2.62s | 4 | Measured |
| 21–65 | ~2.62–3.40s | N/A | **Linearly interpolated** (no ground truth) |
| 66 | 3.40s | 53 | Measured (healthy subjects) |

**Implication:** All generated motions for FMA 21–65 have nearly identical duration (~3s). Temporal variation is only meaningful at the endpoints (FMA 16–20 and 66).

---

## 5. IK Convergence (typical errors)

| Data source | Typical mean IK error | Notes |
|---|---|---|
| Real MoCAP (originals) | ~18mm | Reference quality |
| Generated (CVAE output) | ~24–30mm | Slightly higher due to generation noise |
| Failed frames (>200mm) | Flagged and excluded | Very rare in processed data |

Target: <30mm mean. >200mm = bad sample, excluded.
