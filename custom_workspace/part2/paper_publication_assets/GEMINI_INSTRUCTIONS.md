# Instructions for Gemini: Write main.tex for Journal of Biomechanics (Elsevier)

---

## Your task

Rewrite `latex_submission/main.tex` as a complete, publication-ready paper for the
**Journal of Biomechanics** (Elsevier). Use `elsarticle.cls` with the `\documentclass[5p,times,twocolumn]{elsarticle}` option (two-column, 5p layout — standard for JBiomech full papers).
Use all files in `context/` for facts and methodology. Use `figures_and_tables/` for figure
filenames. **Do not invent numbers** — every statistic must appear verbatim in a `context/` file.

### Journal of Biomechanics formatting requirements
- **Template**: `elsarticle.cls` (Elsevier). The existing `IEEEtran.cls` in the folder is from
  the old draft — do not use it.
- **Bibliography style**: `\bibliographystyle{elsarticle-num}` (numbered, Elsevier style).
- **Section structure**: Abstract → Highlights → Introduction → Methods → Results →
  Discussion → Conflict of Interest → Acknowledgements → References
- **Highlights**: 3–5 bullet points (85 characters max each), placed before Introduction,
  using `\begin{highlights}...\end{highlights}` or as a commented block labelled
  `% Highlights` for Gemini to populate.
- **Keywords**: 4–6 keywords, `\begin{keyword}...\end{keyword}`.
- **Word limit**: ~5,000–6,000 words for the body (Abstract excluded).
- **Figures**: `\begin{figure}[!t]...\end{figure}`, captions below figures.
- **Tables**: `\begin{table}...\end{table}`, captions above tables, `\toprule/\midrule/\bottomrule` (booktabs).
- **Equations**: numbered with `\begin{equation}`, referenced as Eq.~(\ref{eq:X}).
- **Citations**: numbered superscript style — `\cite{key}` renders as [1], [2], etc.

---

## Primary contribution (what the paper is about)

The paper's claim is: **Weighted Jacobian IK with a 3-tier diagonal damping matrix recovers
clinically meaningful trunk compensation in stroke patients from a standard 5-marker arm set.**

The method:
- Model: `myoarm_trunk.xml` (MyoSuite arm + 2 manual trunk DOFs: `trunk_forward_lean`,
  `trunk_lateral_lean`)
- IK formulation: damped least-squares `dq = (J^T J + W)^{-1} J^T e`
- 3-tier damping matrix W (diagonal):

| Tier | Joints | w |
|---|---|---|
| 1 — Arm | deviation, flexion, pro_sup, elbow_flexion, elv_angle, shoulder_elv, shoulder_rot | 0.05 |
| 2 — SC girdle | sternoclavicular_r2, sternoclavicular_r3 | 0.20 |
| 3 — Trunk | trunk_forward_lean, trunk_lateral_lean | 50.0 |

- Neutral-start: trunk pinned at 0 for frame 0 of every recording; free from frame 1
- Phantom joints (unrotscap, acromioclavicular, unrothum, shoulder1_r2) pinned at 0 and
  excluded from Jacobian
- Pipeline: CSV → TRC → MOT (200 Hz, MHH dataset, 24 stroke + 49 healthy)

All clinical numbers come from `output/neutral_start/` (Weighted IK run).

---

## Context files — read these for facts

| File | What it contains |
|---|---|
| `context/README.md` | Key numbers table, file map, what NOT to change |
| `context/steps.md` | Full methodology narrative (Steps 1–11) — every design decision |
| `context/clinical_validation_report.md` | All clinical metrics with exact values |
| `context/final_results_table.md` | Complete quantitative results tables |
| `context/dataset_stats.md` | Cohort breakdown, exclusions, FMA distribution, IK error |
| `context/limitations.md` | Known limitations 1–7 — use for Discussion section |
| `context/trunk_ik_development_log.md` | Development journal (Steps 1–11 in narrative form) |

---

## Figures — exact filenames for `\includegraphics{}`

All figures are in `figures_and_tables/`. Before compiling, copy them into `latex_submission/`.

| Filename | Caption guidance | Section |
|---|---|---|
| `damping_weights.png` | Per-DOF damping bar chart, log scale, 3-tier colour-coded | §2 Methods |
| `pose_comparison.png` | S16_12_3 (stroke) vs 04_12_3 (healthy) at rest and peak drink | §2 or §3 |
| `joint_trajectories_original.png` | Full joint angle time series, Original IK | §3 Results |
| `joint_trajectories_weighted.png` | Full joint angle time series, Weighted IK | §3 Results |
| `clinical_validation_results.png` | 6-metric bar chart Original vs Weighted IK with Δ annotations | §3 Results |
| `final_5_metric_distribution.png` | Box+swarm plots, Mann-Whitney U p-values, Cohen's d | §3 Results |
| `damping_comparison.png` | SC damping config comparison (A/B/C) across 6 trials | §2 Methods or Appendix |
| `ik_vs_cs_correlation.png` | Pipeline consistency: IK trunk lean vs CS plate normal, r=0.980 | §2 Methods or Appendix |

---

## Secondary context — SVD plate normal investigation

`context/steps.md` Step 11 and `context/trunk_ik_development_log.md` document a follow-up
investigation using SVD plate normal geometry. **This is NOT the primary contribution.**

How to use it in the paper:
- In §2 Methods, add a brief paragraph after the weighted IK description: something like
  "To verify that the CS plate geometry and the weighted IK recover the same signal, we
  additionally computed trunk angles directly from the SVD plate normal of CS_1–CS_4
  markers. This pipeline verification yielded r=0.980 across 76 trials (see Fig. X)."
- Always say **"pipeline verification"**, never "validation" for the r=0.980 result.
- Reference `ik_vs_cs_correlation.png` for this result.
- The Discussion section should note that the weighted Jacobian is preferred over SVD
  prescription because it derives trunk angle from arm kinematics alone (no direct chest
  plate measurement), making it epistemically stronger.

---

## Supplementary code files

| File | Description |
|---|---|
| `code_supplementary/ik_clean_primary_weighted.py` | **PRIMARY** — Weighted Jacobian IK (TR_DAMPING=50.0). This is the paper's method. All clinical results were produced by this solver. |
| `code_supplementary/ik_clean_secondary_svd_verification.py` | **SECONDARY / verification only** — SVD plate normal two-pass IK. Pipeline consistency check only (r=0.980). Not the paper's contribution. |
| `code_supplementary/sensitivity_analysis.py` | Trunk damping sensitivity (2 empirical anchors + log-linear model) |
| `code_supplementary/gen_damping_fig.py` | Script that generates damping_weights.png |

---

## Key numbers — do not change these

See `context/README.md` §"Key numbers" for the complete authoritative list. Summary:

- Cohort: n=24 stroke (FMA 16–20), n=49 healthy (FMA 66)
- IK error: Stroke 40.18mm → 35.61mm; Healthy 27.07mm → 22.15mm
- Trunk lean gap: 0.09° (original) → 5.13° (weighted), ~57× amplification
- All 6 biological trends maintained under weighted IK

---

## Target structure for the rewritten main.tex (Journal of Biomechanics)

| Section | Content |
|---|---|
| Abstract | 250 words max. Problem, method, key result (5.13° gap, ~57×), conclusion. |
| Highlights | 3–5 bullets, ≤85 chars each. |
| Keywords | 4–6 terms (e.g. inverse kinematics, stroke rehabilitation, trunk compensation, MuJoCo, motion capture) |
| 1. Introduction | Kinematic redundancy problem; stroke trunk compensation (Cirstea & Levin 2000); gap in current IK tools; paper contribution. |
| 2. Methods | 2.1 Participants & data; 2.2 Musculoskeletal model; 2.3 Baseline IK; 2.4 Weighted Jacobian IK (primary); 2.5 Pipeline verification (SVD, one paragraph); 2.6 Clinical metrics |
| 3. Results | 3.1 IK reconstruction error; 3.2 Trunk forward lean; 3.3 All 6 clinical metrics; 3.4 Joint trajectories |
| 4. Discussion | Why the weighted hierarchy works; V_Sternum failure; SVD as epistemic justification; limitations (use context/limitations.md 1–7) |
| Conflict of Interest | "The authors declare no conflict of interest." |
| Acknowledgements | Leave a placeholder: [Acknowledgements TBD] |
| References | `\bibliography{references}` with `\bibliographystyle{elsarticle-num}` |

Suggested `\label` names to use:
- `\label{sec:weighted}` — Weighted Jacobian IK subsection
- `\label{sec:pipeline_verification}` — SVD pipeline verification paragraph
- `\label{sec:sternum}` — V_Sternum discussion
- `\label{eq:dls}` — damped least-squares equation

---

## Execution Protocol: Room to Think

Before you write any LaTeX code, you MUST output a `<thinking>` block. Use this block to:

1. Map out the full structure of the paper (Abstract, §1–§4, figures, tables).
2. Confirm the exact numbers you will use for the trunk forward lean gap, the multiplier, and the cohort sizes — cross-check them against `context/README.md` §"Key numbers".
3. Explicitly state which method is the paper's primary contribution (Weighted Jacobian, TR_DAMPING=50.0) and which is secondary verification only (SVD plate normal, r=0.980).
4. Confirm the terminology rule: the r=0.980 result is **"pipeline verification"**, never "validation".
5. Plan where each figure will be inserted and which section it belongs to.

Only after completing this `<thinking>` block should you output the final `main.tex` code.

---

## Hard constraints — do not violate

- Do not alter damping values (0.05 / 0.20 / 50.0)
- Do not change cohort sizes (n=24 stroke, n=49 healthy)
- Do not add citations not already in `references.bib` without also adding the bib entry
- Figure filenames in `\includegraphics{}` must exactly match the filenames in `figures_and_tables/`
- Use `elsarticle.cls` — do not use or reference `IEEEtran.cls`, `IEEEtran.bst`, or `ieeetr.bst`
- Use "pipeline verification" not "validation" for the r=0.980 CS plate normal result
- Trunk lean gap is **5.13°** (~57× amplification) — do not write 5.14° or 58×
- Healthy trunk lean is **6.32°** — do not write 6.31°
