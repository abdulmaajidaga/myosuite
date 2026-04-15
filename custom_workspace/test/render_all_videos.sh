#!/bin/bash
# render_all_videos.sh — Generate IK + video for FMA 16, 30, 45, 66 across all experiments.
# Run from custom_workspace root.

set -e
export LD_LIBRARY_PATH=/home/abdul/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=/home/abdul/Desktop/myosuite

PYTHON=/home/abdul/miniconda3/envs/MyoSuite/bin/python
WORKSPACE=/home/abdul/Desktop/myosuite/custom_workspace
OUTPUT=/home/abdul/Desktop/myosuite/custom_workspace/test2/output
FMA_TARGETS="FMA_16.csv FMA_30.csv FMA_45.csv FMA_66.csv"

cd "$WORKSPACE"

EXPERIMENTS=(
  A0_full A1_no_residual A2_no_film A3_no_cfg
  B0_full_loss B1_no_dyn B2_no_acc B3_recon_vel_only
  C0_no_constraints C1_seg_only C2_sag_only C3_both
  D0_stage3_baseline D1_stage3_sag D2_stage3_sag_strong D3_stage3_minimal D4_no_dyn
  E_gs_150 E_gs_200 E_gs_250 E_gs_300 E_gs_400
  F0_cdp_005 F2_cdp_020 F3_cdp_030
  G0_fma_split G1_standard_split_300
)

TOTAL=${#EXPERIMENTS[@]}
DONE=0

for EXP in "${EXPERIMENTS[@]}"; do
  DONE=$((DONE + 1))
  CSV_DIR="$OUTPUT/$EXP/csv"

  if [ ! -d "$CSV_DIR" ]; then
    echo "[$DONE/$TOTAL] SKIP $EXP — no csv/ dir"
    continue
  fi

  # Check if all 4 target videos already exist
  ALL_EXIST=true
  for f in FMA_16 FMA_30 FMA_45 FMA_66; do
    [ ! -f "$OUTPUT/$EXP/videos/${f}.mp4" ] && ALL_EXIST=false && break
  done

  if [ "$ALL_EXIST" = true ]; then
    echo "[$DONE/$TOTAL] SKIP $EXP — videos already exist"
    continue
  fi

  echo ""
  echo "[$DONE/$TOTAL] === $EXP ==="

  $PYTHON scripts/run_generated_pipeline.py \
    --input-dir "$CSV_DIR" \
    --output-dir "$OUTPUT/$EXP" \
    --skip-id \
    $FMA_TARGETS 2>&1 | grep -E "\[2/4\]|\[3/4\]|PIPELINE STOPPED|FAILED" || true

  echo "  Done: $EXP"
done

echo ""
echo "========================================"
echo "All video rendering complete."
echo "========================================"
echo ""
echo "Videos per experiment:"
for EXP in "${EXPERIMENTS[@]}"; do
  COUNT=$(ls "$OUTPUT/$EXP/videos/"*.mp4 2>/dev/null | wc -l)
  echo "  $EXP: $COUNT videos"
done
