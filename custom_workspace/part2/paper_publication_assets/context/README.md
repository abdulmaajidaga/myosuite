# Context Guide for AI-Assisted LaTeX Editing

This folder contains everything needed to understand and edit `main.tex`.

---

## What the paper is about

Stroke patients performing a drinking task compensate for weak arm muscles by leaning their
trunk forward. Standard inverse kinematics (IK) — which uses a uniform joint cost — absorbs
this trunk lean into shoulder joints, producing clinically wrong joint angle distributions.

This paper introduces a **weighted Jacobian IK** that encodes the CNS distal-to-proximal cost
hierarchy (Cirstea & Levin 2000) via a 3-tier diagonal damping matrix. Arm DOFs are cheap
(w=0.05), the sternoclavicular girdle is penalised 4× (w=0.20), and trunk DOFs are penalised
1000× (w=50.0). A neutral-start constraint pins the trunk at 0° for frame 0 of every recording
to prevent a spurious offset. The model is MyoSuite/MuJoCo with two trunk DOFs
(`trunk_forward_lean`, `trunk_lateral_lean`) added to the original `myoarm.xml`.

The method is validated on 24 stroke patients (FMA 16–20) and 49 healthy subjects performing
a standardised cup-drinking task recorded with a 5-site reflective marker set at 200 Hz.

---

## Key numbers (use these exactly — do not paraphrase)

### Cohort
- Stroke: n = 24 (FMA 16–20, mean ~18.6)
- Healthy: n = 49 (FMA 66; Subject 01 and recording 15_12_1 excluded per protocol)

### Damping tiers
| Tier | Joints | w |
|---|---|---|
| 1 — Arm | deviation, flexion, pro_sup, elbow_flexion, elv_angle, shoulder_elv, shoulder_rot (7 DOFs) | 0.05 |
| 2 — SC girdle | sternoclavicular_r2, sternoclavicular_r3 | 0.20 |
| 3 — Trunk | trunk_forward_lean, trunk_lateral_lean | 50.0 |

### IK reconstruction error
| Group | Original IK (mm) | Weighted IK (mm) | Change |
|---|---|---|---|
| Stroke | 40.18 | 35.61 | −11% |
| Healthy | 27.07 | 22.15 | −18% |

### Clinical metrics (Weighted IK, peak values, mean across subjects)
| Metric | Stroke | Healthy | Gap (S−H) | Expected |
|---|---|---|---|---|
| Trunk forward lean (°) | 11.45 | 6.32 | +5.13 | Stroke > Healthy |
| Peak pronation (°) | 74.27 | 65.19 | +9.08 | Stroke > Healthy |
| Reach-path-ratio | 1.70 | 1.37 | +0.33 | Stroke > Healthy |
| Peak shoulder elevation (°) | 22.85 | 28.66 | −5.81 | Stroke < Healthy |
| Wrist reach distance (cm) | 35.37 | 44.14 | −8.77 | Stroke < Healthy |
| Elbow ROM (°) | 71.86 | 80.61 | −8.75 | Stroke < Healthy |

Original IK trunk lean gap was only 0.09° (20.05° stroke vs 19.96° healthy). Weighted IK
expanded it to 5.13° — a ~57× amplification. All 6 biological trends are maintained.

---

## File map

### This folder (`context/`)
| File | What it is | Use for |
|---|---|---|
| `README.md` | This file | Orientation |
| `steps.md` | Full methodology narrative — every step, every mistake and fix | Understanding the Methods section |
| `final_results_table.md` | Complete quantitative results tables | Checking/updating numbers in Results |
| `clinical_validation_report.md` | Per-metric detailed breakdown with all raw values | Detailed Results / Discussion |
| `dataset_stats.md` | MHH cohort breakdown, exclusions, FMA distribution, IK error by pipeline | Methods / Dataset subsection |
| `limitations.md` | Known limitations specific to this IK method | Discussion / Limitations section |

### `../latex_submission/`
| File | What it is |
|---|---|
| `main.tex` | **The paper — edit this** |
| `references.bib` | Bibliography (11 entries) |
| `IEEEtran.cls`, `IEEEtran.bst`, `ieeetr.bst` | IEEE template files — do not edit |
| `main.pdf` | Last compiled output — for visual reference |
| `damping_weights.png` | Fig 1 — per-DOF damping bar chart (log scale, 3-tier colour-coded) |
| `pose_comparison.png` | Fig 2 — S16_12_3 (stroke) vs 04_12_3 (healthy) at rest + peak drink |
| `joint_trajectories_original.png` | Fig 3 — full joint angle time series, Original IK |
| `joint_trajectories_weighted.png` | Fig 4 — full joint angle time series, Weighted IK |
| `clinical_validation_results.png` | Fig 5 — 6-metric bar chart, Original vs Weighted IK |
| `final_5_metric_distribution.png` | Fig 6 — box+swarm plots, Mann-Whitney U, Cohen's d |

### `../code_supplementary/`
| File | What it is |
|---|---|
| `ik_clean_primary_weighted.py` | **PRIMARY** — Weighted Jacobian IK (TR_DAMPING=50.0). This is the method the paper describes and from which all clinical results were produced. |
| `ik_clean_secondary_svd_verification.py` | **SECONDARY / verification only** — SVD plate normal two-pass IK. Used as a pipeline consistency check (r=0.980). Not the paper's contribution. |
| `sensitivity_analysis.py` | Trunk damping sensitivity analysis (2 empirical anchors + log-linear model) |
| `gen_damping_fig.py` | Script that generates `damping_weights.png` |

### `../figures_and_tables/`
Archive copies of all figures. The versions in `latex_submission/` are the ones actually
included in the compiled PDF.

---

## `main.tex` structure (1234 lines)

| Section | Lines | Content |
|---|---|---|
| Abstract | ~1–60 | Core contribution summary |
| §1 Introduction | ~62–162 | Kinematic redundancy problem, stroke compensation |
| §1.1 Related Work | ~98–162 | OpenSim / MuJoCo / MyoSuite context |
| §2 Methods | ~163–624 | Dataset, model, preprocessing, baseline IK, weighted IK, metrics |
| §3 Results | ~625–998 | IK error, solver-independent metrics, IK-dependent metrics, trajectories |
| §4 Discussion | ~999–1234 | Why it works, V_Sternum failure, clinical implications |

Key `\label` references in the text:
- `\ref{sec:weighted}` — Weighted Jacobian IK method section
- `\ref{sec:ik_validation}` — IK-dependent metrics results section
- `\ref{sec:sternum}` — V_Sternum appendix / discussion

---

---

## Part 2 Investigation — SVD Plate Normal (secondary context, not the paper's contribution)

After the weighted Jacobian pipeline was complete, a follow-up investigation explored whether
trunk angles could be recovered by a **pure geometric approach** — directly prescribing
`trunk_forward_lean` from the SVD plate normal of the CS_1–CS_4 chest markers, rather than
letting the IK optimizer derive it under a high-cost penalty.

### What it is
- **Two-pass IK**: Pass 1 = compute CS plate normal via SVD of the 4×3 marker matrix →
  smallest right singular vector → arctan2 → forward tilt angle. Prescribe this directly to
  `trunk_forward_lean` qpos. Pass 2 = damped least-squares for arm and SC DOFs with trunk
  pinned (`arm_only_mask` zeros the trunk Jacobian column).
- **SC damping hierarchy**: SC=0.20 (4×) tested empirically against uniform=0.05 and
  heavy=0.50 on 6 representative trials. SC=0.20 saves ~7mm mean IK error vs uniform;
  above 0.20 is diminishing returns.
- **Axis fix applied**: `trunk_forward_lean` axis corrected to `(1,0,0)` (sagittal rotation);
  `trunk_lateral_lean` to `(0,-1,0)`. Earlier XML had these swapped.

### What we found
- **r=0.980** pipeline consistency check across 76 trials (peak trunk_forward_lean in MOT
  matches peak CS plate normal tilt from same CS markers). This is a **verification of
  internal consistency**, not an independent validation — the trunk angle *is* the CS plate
  normal by construction.
- Pure position-based CS IK (without prescription) gives r=0.906 but amplitude halved (2.6°
  vs 5°) due to model moment arm mismatch (320mm model vs 82–374mm effective lab range,
  SD=137mm). No single geometry calibration fixes this across subjects.
- Calibrating geometry from plate normal then running "pure IK" is circular (tautological).

### Key terminology rule
Use **"pipeline verification"** (not "validation") for the r=0.980 result throughout the
paper. The trunk angle was prescribed from the CS data — this shows consistency, not
that the model independently recovers something independently measured.

### What this means for the paper
The weighted Jacobian (TR_DAMPING=50.0) remains the primary contribution. It avoids the
strong-prior problem: trunk angle emerges from IK pressure alone, with no direct injection
from CS markers. The SVD investigation reinforces that the CS plate normal geometry IS the
forward lean signal — and that the weighted IK captures it without prescription.

Reference files:
- `trunk_ik_development_log.md` (this folder) — full 11-phase development journal
- `../figures_and_tables/damping_comparison.png` — SC damping config comparison
- `../figures_and_tables/ik_vs_cs_correlation.png` — pipeline consistency scatter (r=0.980)
- `../code_supplementary/ik_clean_secondary_svd_verification.py` — SVD two-pass IK code

---

## What NOT to change

- Do not alter the damping values (0.05 / 0.20 / 50.0) — these are empirically validated.
- Do not change cohort sizes (n=24 stroke, n=49 healthy) — these reflect the exact exclusion protocol.
- Do not add citations that are not in `references.bib` without also adding the bib entry.
- Figure filenames in `\includegraphics{}` must match exactly what is in `latex_submission/`.
