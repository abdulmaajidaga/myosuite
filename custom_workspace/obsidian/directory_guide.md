---
name: Directory Guide
description: Full directory structure, what every folder contains, and which files matter
type: reference
---

# Project Directory Guide

Root: `/home/abdul/Desktop/myosuite/custom_workspace/`

---

## Top-level files

| File | Purpose |
|---|---|
| `config/settings.yaml` | Centralised config — all paths, hyperparameters, pipeline settings |
| `CLAUDE.md` | Instructions for Claude Code when working in this repo |
| `requirements.txt` | Python dependencies |
| `GEMINI.md` | Legacy file — ignore |

---

## `latex/`
All LaTeX source and build output.

| File | Purpose |
|---|---|
| `latex/main.tex` | The report source — compile from inside `latex/` |
| `latex/main.pdf` | Compiled output |
| `latex/main.aux`, `*.toc`, `*.lof` etc. | Build artefacts (auto-generated, safe to delete and rebuild) |

**To compile:**
```bash
cd /home/abdul/Desktop/myosuite/custom_workspace/latex
pdflatex main.tex        # single pass (fast, may have stale refs)
pdflatex main.tex        # run twice to resolve cross-references
# or
latexmk -pdf main.tex    # fully automatic (recommended)
```

`\graphicspath{{../figures/}}` — LaTeX resolves figures from `custom_workspace/figures/` automatically.

---

## `figures/`
All PNG figures referenced by `main.tex`. LaTeX finds them automatically via `\graphicspath{{figures/}}`. **Never put figures anywhere else** — always save to here.

See `obsidian/visuals_guide.md` for which script generates each figure.

---

## `config/`
- `settings.yaml` — single source of truth for all paths and hyperparameters. Loaded in Python via `src/utils/config.py`.

---

## `data/kinematic/`

| Subfolder | Contents |
|---|---|
| `healthy/` | Raw CSV files from healthy subjects (53 recordings, 18 subjects) |
| `stroke/` | Raw CSV files from stroke patients (24 recordings, 9 patients, FMA 16–20) |
| `cutoff/processed/` | 77 preprocessed cutoff segments (100 frames, chest-relative, no outlier) |
| `cutoff/original/` | Original full-length recordings before cutoff extraction |
| `cutoff/augmented_smote/` | ~58,303 SMOTE-augmented files (≈1,140 per FMA level, FMA 16–66) |
| `cutoff/augmented_dtw/` | ~59,172 DTW-morphed files |
| `cutoff/augmented_linear/` | ~56,574 linearly interpolated files |

Filename conventions:
- Stroke: `S{patient}_{session}_{trial}.csv` (e.g. `S3_12_1.csv`)
- Healthy: `{subject}_{session}_{trial}.csv` (e.g. `01_12_1.csv`)
- Augmented SMOTE: `smote_{idx:05d}_FMA{score}.csv`
- Augmented DTW/Linear: `{subject}_{session}_{trial}_FMA{score}.csv`

---

## `models/`

| Path | Contents |
|---|---|
| `models/cvae/cvae_cutoff_fma.pth` | Root workspace CVAE weights (83% litval) |
| `models/cvae/cvae_cutoff_fma_best.pth` | Best checkpoint from workspace training |
| `models/cvae/scaler_cutoff_fma.pkl` | StandardScaler — must match training data |
| `models/model/myo_sim/arm/myoarm.xml` | MyoArm MuJoCo musculoskeletal model (27 DOF, 63 muscles) |

---

## `src/`

| Module | Purpose |
|---|---|
| `src/data_processing/` | CSV→TRC conversion, preprocessing, temporal scaling, SMOTE augmentation |
| `src/inverse_kinematics/convert_trc2mot.py` | TRC→MOT (IK solver, Levenberg-Marquardt, adaptive damping) |
| `src/inverse_dynamics/calc_mot2invdyn.py` | MOT→torques/activations (static optimisation, synergy extraction) |
| `src/generation/model.py` | Workspace CVAE architecture (FiLM + CFG + Residual, BiLSTM encoder) |
| `src/generation/model_v2.py` | Architecture V2 (FiLM + Bahdanau attention + TCB) |
| `src/generation/model_v3.py` | Architecture V3 (dual-branch convolutional decoder) |
| `src/generation/cvae_train.py` | Training script for workspace model |
| `src/generation/cvae_generate.py` | Generation script for workspace model |
| `src/generation/generate_augmented_smote.py` | SMOTE dataset generator |
| `src/generation/generate_augmented_fma.py` | DTW/linear augmentation generator |
| `src/utils/config.py` | Config loader (`get_path()`, `get()`, `get_project_root()`) |
| `src/utils/trc_parser.py` | TRC file parser |
| `src/visualization/convert_mot2video.py` | MOT→MP4 video renderer (MuJoCo) |
| `src/visualization/plot_publication_figures.py` | Publication-quality figure generator |

---

## `scripts/`

| Script | Purpose |
|---|---|
| `run_pipeline.py` | Batch IK pipeline (CSV→TRC→MOT for all originals) |
| `run_modular_pipeline.py` | Single-file modular pipeline |
| `run_generated_pipeline.py` | CVAE output→IK→ID pipeline (`--input-dir`, `--output-dir`, `--skip-id`) |
| `literature_validation.py` | 18-check automated literature validation (`--gen-id-dir`, `--output-dir`) |
| `publication_stats.py` | Group statistics, correlations, Cohen's d |
| `cvae_validation.py` | KS test + Wasserstein: generated vs real distribution |
| `viz/pca_dataset_comparison.py` | 2×2 PCA figure: original gap + 3 augmentation methods |
| `viz/visualize_id_findings.py` | ID results dashboard (torques, activations, synergies) |
| `viz/visualize_smote.py` | SMOTE quality: PCA + trajectory plots |
| `viz/validate_emg.py` | EMG validation against published muscle patterns |
| `viz/verify_midline.py` | Sagittal plane midline verification |
| `viz/animate_v1_v2_v3.py` | Side-by-side animation of V1/V2/V3 architectures |

---

## `test/`
Self-contained experimental sandbox for ablation studies and architecture development. Does not depend on `src/` at runtime.

| Path | Contents |
|---|---|
| `test/models.py` | MotionCVAE with config dict (FiLM+CFG+Residual+TCB) |
| `test/train.py` | Training loop (ReduceLROnPlateau, CFG dropout) |
| `test/generate.py` | Generation with n-sample averaging + guidance |
| `test/evaluate.py` | wrist_rho, trunk_rho, sag_dev, segment_std metrics |
| `test/experiments.py` | All experiment configs (A0–D3, E, F, G, H, I phases) |
| `test/run_experiment.py` | CLI runner: `python test/run_experiment.py --experiment D1_stage3_sag` |
| `test/make_paper_figures.py` | Generates all Ch4 figures → `figures/` |
| `test/output/` | One subfolder per experiment (model weights, CSV, eval_summary.json) |
| `test/output/litval_I1/` | Clean ID run on I1_tcb_no_sag (honest 14/18 litval, no post-processing) |

### Key experiment outputs in `test/output/`

| Experiment | Description | wrist_rho (n=10) | litval |
|---|---|---|---|
| `D1_stage3_sag/` | FiLM+CFG+Residual+SMOTE+w_sag=5 | 0.915 | 15/18 (83%) |
| `G1_standard_split_300/` | D1 arch + 300 epochs | 0.550 | — |
| `I1_tcb_no_sag/` | +TCB, no sagittal constraint | 0.860 | 14/18 honest / 17/18 with post-proc |
| `litval_I1/` | ID output for I1 honest litval run | — | 14/18 (78%) |

---

## `output/`

| Path | Contents |
|---|---|
| `output/originals/trc/` | TRC files from real MoCap sessions |
| `output/originals/mot/` | MOT files (joint angles) from real sessions |
| `output/originals/id/` | ID results from real sessions (torques, activations) |
| `output/compressed/` | 100-frame resampled versions of all originals |
| `output/generated/csv/` | CVAE-generated CSV files (workspace model) |
| `output/generated/plots/` | Plots from literature_validation, publication_stats, cvae_validation |
| `output/analysis/` | General analysis figures (PCA, trajectory analysis) |
| `output/phase_indices/` | JSON files mapping trial names to Pick/Drink/Place frame indices |
| `output/scores.csv` | FMA scores for each original session (24 stroke + 1 healthy entry) |

---

## `obsidian/`
Research notes for the paper. All notes in Markdown.

| File | Contents |
|---|---|
| `findings_summary.md` | Plain-language synthesis of all experimental findings |
| `results_all_models.md` | Complete numeric results: all experiments + litval per model |
| `dataset_stats.md` | MHH dataset breakdown, augmentation counts, temporal scaling |
| `litval_critique.md` | Per-check analysis of 18 litval checks with literature backing |
| `test2_ablation_results.md` | Full Phase A–D table with explanations |
| `architectural_citations.md` | Citations for all model components + alternatives from literature |
| `limitations.md` | Known limitations: temporal scaling, FMA gap, pro/sup bias |
| `visuals_guide.md` | How to regenerate every figure in the report |
| `directory_guide.md` | This file |
| `session_2026_04_09.md` | Session notes: D1/E/F/G phases, metric rationale |
| `session_2026_04_10.md` | Session notes: Phase I (TCB), litval scrutiny, honest scores |

---

## `docs/`
- `docs/info/` — Patient documentation, data release notes for MHH dataset

---

## `report/`
Legacy report folder — superseded by `main.tex` at root.

---

## Conda environment
Python: `/home/abdul/miniconda3/envs/MyoSuite/bin/python`  
Version: 3.9 (has dm_control, mujoco, myosuite)

Always activate or use the full path — base Python does not have the required packages.
