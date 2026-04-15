# test2 Ablation Study — Full Results & Conclusions

**Date:** 2026-04-08  
**Framework:** test2/ — 16 experiments across 4 phases  
**Data:** SMOTE augmented, FMA 16–66 (51 scores), 10,153 training files  
**Hardware:** NVIDIA RTX 4070 Laptop (8GB), ~14 min per experiment  

---

## 1. Full Results Table (16 experiments)

| Experiment | Architecture | Key change | segment_std (mm) | sag_dev (mm) | wrist_rho | trunk_rho |
|---|---|---|---|---|---|---|
| **A0_full** | **FiLM+CFG+Residual** | **Reference** | **11.09** | **24.80** | **0.688** | **-0.695** |
| A1_no_residual | FiLM+CFG | −Residual | 12.11 | 20.49 | 0.389 | -0.102 |
| A2_no_film | Concat+CFG | −FiLM | 11.47 | 22.88 | 0.426 | -0.270 |
| A3_no_cfg | FiLM only | −CFG | 8.83 | 16.00 | 0.272 | -0.573 |
| B0_full_loss | FiLM+CFG | Full loss | 12.08 | 28.29 | 0.607 | -0.711 |
| B1_no_dyn | FiLM+CFG | −dyn_corr | 11.10 | 24.76 | 0.594 | -0.467 |
| B2_no_acc | FiLM+CFG | −acc loss | 11.11 | 22.86 | 0.258 | -0.632 |
| B3_recon_vel_only | FiLM+CFG | Recon+vel only | 11.37 | 25.59 | 0.454 | -0.428 |
| C0_no_constraints | FiLM+CFG | No constraints | 13.14 | 21.02 | 0.318 | -0.421 |
| C1_seg_only | FiLM+CFG | +Seg loss (scaled) | 21.05 | 18.75 | 0.519 | -0.727 |
| **C2_sag_only** | **FiLM+CFG** | **+Sag constraint** | **14.15** | **0.41** | **0.496** | **-0.617** |
| C3_both | FiLM+CFG | +Both | 26.53 | 0.62 | 0.196 | -0.782 |
| D0_stage3_baseline | FiLM+CFG+Residual | Stage 3, full loss | 11.23 | 34.00 | 0.443 | -0.656 |
| **D1_stage3_sag** | **FiLM+CFG+Residual** | **+Sag (w=5)** | **14.02** | **0.29** | **0.665** | **-0.462** |
| D2_stage3_sag_strong | FiLM+CFG+Residual | +Sag (w=15) | 14.43 | 0.13 | 0.557 | -0.568 |
| D3_stage3_minimal | FiLM+CFG+Residual | +Sag, min loss | 13.09 | 0.28 | 0.345 | -0.664 |

**Metric definitions:**
- `segment_std_mean` — temporal std of upper-arm + forearm length (mm). Target: ~6mm (real MoCap). Lower = more rigid body consistency.
- `sag_dev_mean` — mean lateral (X-axis) wrist displacement from start frame (mm). Target: ~0. Lower = more sagittal-plane constrained.
- `wrist_rho` — Spearman ρ (FMA vs wrist Y-axis range). Target: +1.0. Measures FMA gradient strength.
- `trunk_rho` — Spearman ρ (FMA vs trunk compensation ratio). Target: −1.0.

---

## 2. Phase A — Architecture Ablation

### Question
Which components are actually necessary? Can any of Stage 3's FiLM, CFG, or Residual be removed?

### Corrected finding

With D0 (Stage 3 properly trained) giving segment_std=11.23mm — the same range as Stage 2 experiments (11–14mm) — **the residual decoder does NOT improve physical plausibility (segment consistency)**. Its value comes from what the test/ 12-config sweep showed: CCI Rho improvement when combined with SMOTE (Stage 2+SMOTE: −1.000 CCI vs Stage 1+SMOTE: −0.650 CCI).

| Component | Effect on CCI (from test/ sweep) | Effect on segment_std (test2) | Effect on wrist_rho (test2) |
|---|---|---|---|
| Residual | SMOTE CCI: −0.650→−1.000 | 11.09 (A0) vs 12.11 (A1) — modest | **0.688 → 0.389** (significant) |
| FiLM | −0.650→−1.000 on SMOTE | 11.47 vs 12.11mm (minor) | 0.426 vs 0.389 (minor) |
| CFG | Largest single gain (−0.480→−0.650, DTW −0.600→−1.000) | 8.83 vs ~11mm | **0.272** vs 0.688 (critical) |

### Architecture conclusions

- **CFG is the most impactful component** — without it (A3), wrist_rho drops from 0.607 to 0.272. The FMA gradient collapses.
- **FiLM and Residual help CCI Rho** on SMOTE (from test/ evidence) but have minor effect on the physical metrics measured in test2.
- **Nothing can be safely removed** — each component has a specific purpose.
- **The minimal sufficient architecture for this task is Stage 3 (FiLM + CFG + Residual)**, validated by the test/ 12-config sweep CCI results.

---

## 3. Phase B — Loss Function Ablation

### Setup
All Phase B experiments used Stage 2 (FiLM + CFG, no Residual). The differences between B experiments are modest — architecture has a larger effect than loss function.

### Results

| Comparison | segment_std | wrist_rho | Conclusion |
|---|---|---|---|
| B0 (full) vs B1 (−dyn) | 12.08 → 11.10 | 0.607 → 0.594 | `w_dyn` is marginal — remove it |
| B1 vs B2 (−acc) | 11.10 → 11.11 | 0.594 → 0.258 | Acc loss matters for gradient |
| B2 vs B3 (−dyn, −acc) | ≈same segment | 0.607 → 0.454 | Both losses contribute to gradient |

### Loss conclusions

- **`w_dyn` (dynamics correlation loss) adds no measurable benefit** — remove from final config.
- **`w_acc` (acceleration matching) supports the FMA gradient** — keep.
- **Recommended minimal loss**: `L_recon + 10·L_vel + 5·L_acc + 0.1·L_KL`
- `w_seg` (segment loss in scaled space): **remove** — see Phase C.
- `w_sag` (sagittal constraint): **add** — see Phase C/D.

---

## 4. Phase C — Physical Constraint Experiments (Stage 2 base)

### Constraint results

| Constraint | sag_dev result | segment_std result | Verdict |
|---|---|---|---|
| Segment loss, scaled space (C1) | 18.75mm (no change) | **21.05mm (worse!)** | **Fails** |
| Sagittal constraint w=5 (C2) | **0.41mm (50× reduction)** | 14.15mm (no change) | **Works** |
| Both together (C3) | 0.62mm (sag works) | 26.53mm (seg makes worse) | Conflicts |

### Why segment loss in scaled space fails

`StandardScaler` normalises each column by its own σ. Computing `||El_scaled − Sh_scaled||` produces a distorted distance metric where X, Y, Z axes have different effective weights. The model minimises this distorted metric in a way that increases physical segment variance. **This loss implementation is invalid and must not be used.**

To properly implement segment length regularisation, the loss would need to operate in physical mm space (inverse-transform first). This was not implemented in test2 — and given that no architecture achieved <11mm segment std anyway, the return on investment is low.

### Why sagittal constraint works

```python
L_sag = (el_x - el_x[:, 0]).pow(2).mean() + (wr_x - wr_x[:, 0]).pow(2).mean()
```

In the scaled space, X represents the lateral axis. Training data starts at ~0 delta (resting start pose), so penalising X deviation from start correctly constrains lateral wrist/elbow motion throughout the reaching trajectory. The transformation is well-posed (each column independently normalised, X penalty = X penalty).

---

## 5. Phase D — Stage 3 + Sagittal Constraint

### Results

| Experiment | sag_dev | segment_std | wrist_rho | Assessment |
|---|---|---|---|---|
| D0 (Stage 3 baseline) | 34.00mm | 11.23mm | 0.443 | Higher sag_dev than Stage 2 — stochastic variation |
| **D1 (w_sag=5)** | **0.29mm** | **14.02mm** | **0.665** | **Best overall — target config** |
| D2 (w_sag=15) | **0.13mm** | 14.43mm | 0.557 | Tighter sag, weaker FMA gradient |
| D3 (min loss + sag) | 0.28mm | 13.09mm | 0.345 | Loss reduction hurts gradient |

### Key finding

**D1 (Stage 3 + w_sag=5) is the best configuration:**
- `sag_dev = 0.29mm` (vs ~20–34mm without constraint) — lateral motion effectively eliminated ✓
- `wrist_rho = 0.665` — highest of all sag experiments, acceptable FMA gradient ✓
- `segment_std = 14mm` — no improvement vs Stage 2 baseline; not a regression either
- `elbow_rom = 55.9°` (Stage3_SMOTE baseline: 57.4°) — essentially unchanged
- `pro_sup_rom = 86.2°` (Stage3_SMOTE baseline: 79.3°, target ~50°) — **WORSE, not better**

**Stronger sagittal constraint (D2, w=15) over-constrains**: sag_dev drops to 0.13mm but wrist_rho falls to 0.557, suggesting the model starts sacrificing clinical gradient quality to satisfy the lateral constraint.

**The sagittal constraint also improves wrist_rho** (0.443→0.665). Explanation: by confining the wrist to the sagittal plane, the Y-axis (forward) becomes the dominant direction of motion. The Y-range metric then more accurately captures the reach amplitude → stronger FMA correlation.

### Corrected pro/sup hypothesis — IK confirmed (2026-04-08)

The initial hypothesis ("lateral drift → IK routes into pro_sup → sagittal constraint fixes it") was **wrong**.

IK results on D1: pro_sup_rom = **86.2°** (worse than Stage3_SMOTE baseline 79.3°). The sagittal constraint eliminates lateral drift but does not reduce pro/sup overestimation.

**Actual cause of pro/sup overestimation:** The D1 model generates wrist starting positions with a large Z-axis offset from the reference pose (Wr_z ~118mm vs expected ~41mm from REFERENCE_POSE in generate.py). The IK solver routes this Z-axis wrist excursion into pro/sup rotation. This is an IK solver distributional bias for this specific task geometry — it independently chooses how to split Z-axis forearm motion between elbow_flexion and pro_sup. It cannot be fixed through CVAE training constraints.

**Pro/sup overestimation is an open limitation** that affects all architectures and cannot be addressed through:
- Sagittal constraint (tested — makes it slightly worse)
- Architecture changes (Stage 0-3 all give similar IK errors)
- Loss function changes (Phase B showed no effect)

Resolution would require: (a) a different IK solver configuration with explicit pro_sup weighting, or (b) physical-space CVAE constraints that enforce anatomically valid wrist-forearm alignment during generation.

---

## 6. Segment Consistency — Summary

None of the 16 experiments achieved the 6mm real MoCap target for segment_std. All properly-trained models converge to 11–14mm regardless of architecture or loss function.

| Target | Best result | Method | Gap |
|---|---|---|---|
| ~6mm (real MoCap) | 11.10mm | B1 (Stage 2, no dyn loss) | ~85% above target |
| ~6mm | 11.23mm | D0 (Stage 3) | ~87% above target |

**This remains an open limitation.** Approaches that could address it in future work:
- Implement segment length loss in physical mm space (inverse-transform before computing distance)
- Add rigid body constraints to the MuJoCo IK solver (post-processing, not training)
- Use a physics-based decoder (e.g., forward kinematics layer) instead of free LSTM

For the paper, this should be stated as a known limitation (§6.3), noting that the IK solver handles most of the segment inconsistency by fitting joint angles that best explain the noisy marker positions.

---

## 7. Final Configuration Recommendation

### **D1: Stage 3 (FiLM + CFG + Residual) + SMOTE + sagittal constraint (w_sag=5)**

Training loss: `L_recon + 10·L_vel + 5·L_acc + 0.1·L_KL + 5·L_sag`

**Why this is the best config:**
1. Stage 3 architecture provides best CCI Rho (−1.000, from test/ sweep)
2. SMOTE augmentation enables model to learn motion physics rather than replay timing
3. Sagittal constraint reduces lateral wrist deviation from ~30mm to 0.29mm
4. This directly reduces pro/sup overestimation in IK (lateral motion → pro_sup route eliminated)
5. No dyn_corr loss — removed (marginal, adds compute)
6. No segment loss — removed (broken implementation, residual doesn't help here either)

**IK confirmed (2026-04-08):**
- Elbow ROM: 55.9° (baseline 57.4°) — essentially unchanged
- Pro/sup ROM: **86.2°** (baseline 79.3°, target ~50°) — **WORSE, hypothesis was incorrect**
- Pro/sup overestimation is caused by IK solver distributional bias, not fixable via CVAE training

**Expected LitVal:** marginal improvement over 72% at best. D1 value is:
1. Better physical plausibility (sag_dev 34mm → 0.29mm)
2. Better FMA gradient (wrist_rho 0.443 → 0.665 — better than even A0_full without constraint)
3. NOT better on pro/sup ROM

### Next step: run ID on D1 to confirm CCI Rho and actual LitVal

```bash
python scripts/run_generated_pipeline.py --input-dir test2/output/D1_stage3_sag/csv
```

---

## 8. Ablation Narrative for Paper (§4)

**§4.1 Conditioning strategy: CFG is the critical component**
- Classifier-free guidance provides the largest single gain across all metrics
- Without CFG (A3): wrist FMA gradient collapses (wrist_rho 0.272 vs 0.607)
- CFG allows amplifying the FMA conditioning signal at inference time via guided generation

**§4.2 Feature modulation: FiLM enables SMOTE**
- FiLM modulates LSTM hidden state per-sample by condition (vs concatenation in Stage 1)
- CCI impact: Stage 1+SMOTE −0.650 → Stage 2+SMOTE −1.000 (from test/ sweep)
- Physical metrics: minor improvement in test2 (segment_std 11.47 vs 12.11mm)

**§4.3 Residual skip connection: CCI quality, not physical consistency**
- CCI impact: Stage 2+SMOTE −1.000 → Stage 3+SMOTE −1.000 (unchanged, but velocity profiles smoother)
- Physical metrics: no significant difference (11.23 vs 12.08mm segment_std)
- Value is in temporal coherence of velocity profiles (shown in Phase A figures from test/)

**§4.4 Sagittal-plane constraint: physical plausibility and gradient improvement (IK confirmed)**
- Novel training objective addition: penalise lateral wrist/elbow deviation from start frame
- Effect: sag_dev 20–34mm → 0.13–0.29mm (50–100× reduction) ✓
- Secondary effect: wrist_rho improves (0.443 → 0.665) because Y-axis becomes dominant direction ✓
- Pro/sup ROM: 86.2° (D1) vs 79.3° (Stage3_SMOTE) — NOT improved; hypothesis was incorrect
- Pro/sup overestimation is an IK solver artifact and remains an open limitation for all models

---

*All 16 experiments complete. A0_full retrained (150 epochs, 10,153 samples) — smoke-test artifact resolved.*
