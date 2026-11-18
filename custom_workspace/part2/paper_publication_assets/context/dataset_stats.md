# Dataset Statistics — Reference

## MHH Real Motion Capture Dataset

### Recording protocol
- Task: Drinking task (pick up cup → drink → return cup to table)
- Marker set: Shoulder, Elbow, Wrist, WristVector, Trunk (5 sites × 3 axes = 15 channels)
- Sampling rate: 200 Hz

### Subject breakdown (raw recordings)

| Group | Sessions | FMA range | FMA mean |
|---|---|---|---|
| Stroke patients | 24 sessions (S-prefix files) | 16–20 | ~18.6 |
| Healthy subjects | 53 sessions | 66 (all healthy) | 66 |
| **Total** | **77 sessions** | — | — |

### Analytic sample (after exclusions)

- Subject 01 excluded per protocol (data quality)
- Recording `15_12_1` excluded (bad ELB_M marker)
- **Final: 24 stroke, 49 healthy**

### FMA distribution (stroke patients only)

| FMA score | Sessions |
|---|---|
| 16 | 2 |
| 17 | 3 |
| 18 | 3 |
| 19 | 12 |
| 20 | 4 |
| **Total** | **24** |

Note: No patients with FMA 21–65 exist in the dataset. The cohort covers only severe stroke
(FMA 16–20) and healthy (FMA 66).

### IK convergence

| Pipeline | Group | Mean error (mm) |
|---|---|---|
| Original IK (uniform damping) | Stroke | 40.18 |
| Original IK (uniform damping) | Healthy | 27.07 |
| Weighted IK (proposed) | Stroke | 35.61 |
| Weighted IK (proposed) | Healthy | 22.15 |

Warn threshold: 30mm. Fail threshold: 200mm (frame excluded).
