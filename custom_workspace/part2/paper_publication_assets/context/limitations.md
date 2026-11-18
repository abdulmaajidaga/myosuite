# Known Limitations of the Weighted IK Method

## 1. Sensitivity Analysis Has Only Two Empirical Anchors

Only two damping values were run on the full cohort: W=0.05 (uniform baseline, `output/originals/`)
and W=50.0 (proposed, `output/neutral_start/`). Intermediate values (W=1, 10, 25, 100) in the
sensitivity analysis are log-linearly interpolated between these two anchors, not empirically
measured. The model is physically motivated (log-linear decay in IK error, saturation in trunk
gap) but remains an approximation.

## 2. Small and Narrow Cohort

- 24 stroke patients, all FMA 16–20 (severe stroke only)
- Single task: drinking
- Single session per patient
- No moderate/mild stroke patients (FMA 21–65) in the dataset

Results may not generalise to other severity levels, tasks, or populations.

## 3. V_Sternum Excluded from IK

The sternum marker (V_Sternum) was added to `myoarm_body.xml` but excluded from the IK solve
(`IK_SKIP_MARKERS`). The shoulder-relative rotation that correctly maps arm markers to model
space places the sternum at the wrong location — including it breaks convergence. A future
solution would require a separate coordinate mapping for trunk markers.

## 4. Trunk DOFs Are a Manual Extension

The two trunk DOFs (`trunk_forward_lean`, `trunk_lateral_lean`) were manually added to
`myoarm_trunk.xml` — they are not part of the original MyoSuite model. The range (±0.35 rad)
was set by inspection, not from a biomechanical measurement of the cohort.

## 5. Weighted IK Is Sensitive to Damping Choice

The 1000× trunk penalty (W=50.0 vs arm W=0.05) was chosen empirically. The sensitivity
analysis shows the trunk gap saturates around W=50 and IK error starts drifting upward beyond
it, but the transition is smooth — there is no sharp optimum. A different cohort or task might
warrant a different value.

## 6. SVD Plate Normal is a Strong Prior (justifies Weighted Jacobian as primary method)

A follow-up investigation prescribed trunk_forward_lean directly from the SVD plate normal of
CS_1–CS_4 chest markers. This approach achieves r=0.980 pipeline consistency but is
epistemically circular: the trunk angle *is* the CS plate geometry by construction. The IK
cannot contradict the measurement. The weighted Jacobian (TR_DAMPING=50.0) avoids this by
deriving trunk angle solely from arm-marker residuals under a cost hierarchy — it is
genuinely inferring trunk compensation, not reading it off the chest plate.

## 7. SC Damping Validated on Six Trials Only

The SC=0.20 choice was validated on 3 stroke + 3 healthy representative trials. The full
cohort was not re-run for every SC damping value. The tested range (uniform=0.05 to
heavy=0.50) shows diminishing returns above 0.20 (~7mm improvement at 0.20, ~9mm at 0.50),
but this may not hold for trials with unusual shoulder girdle motion.
