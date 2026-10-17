#!/usr/bin/env bash
# Follow-ups requested in review, run after the main sweep (~5 h on one A40):
#   1. background-only shortcut test for every architecture x 3 seeds   (EfficientNet already done)
#   2. ViT-S pretrained on ImageNet-1k only (same arch + recipe), E0/E1/E2 x 3 seeds: separates
#      "attention vs convolution" from "ImageNet-21k vs 1k pretraining"
#   3. results table with cross-model paired tests (src.report)
# Resumable: finished runs are skipped; a failed run is recorded and the rest continue.
set -uo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2"}
export SKIP_FINISHED=1
FAILED=()
T0=$(date +%s)
hdr() { local el=$(( $(date +%s) - T0 )); echo; echo "=================== [followups] $(date '+%H:%M:%S') (elapsed $((el / 3600))h$(printf %02d $((el % 3600 / 60)))m)  $* ==================="; }

echo "##### 1/3 background-only shortcut test per architecture #####"
for m in resnet50 vit_s16 vit_s16_in1k efficientnet_b0; do for s in $SEEDS; do
  if [ -f "runs/shortcut_hull/$m/seed$s/metrics.json" ]; then echo "[shortcut] $m seed $s already done, skipping"; continue; fi
  hdr "shortcut model=$m seed=$s"
  python -m src.shortcut --config configs/audit/shortcut.yaml model=$m seed=$s || FAILED+=("shortcut $m seed $s")
done; done

echo "##### 2/3 ViT-S ImageNet-1k-only: E0, E1, E2 #####"
for s in $SEEDS; do for c in e0_baseline e1_strong_aug e2_composite; do
  hdr "$c model=vit_s16_in1k seed=$s"
  python -m src.train --config configs/experiments/$c.yaml model=vit_s16_in1k seed=$s || FAILED+=("$c vit_s16_in1k seed $s")
done; done

echo "##### 3/3 results table #####"
python -m src.report || FAILED+=("report")
el=$(( $(date +%s) - T0 ))
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "######## FOLLOW-UPS FINISHED WITH FAILURES in $((el / 3600))h$(printf %02d $((el % 3600 / 60)))m:"; printf '   - %s\n' "${FAILED[@]}"
  echo "Fix, then re-run: finished runs are skipped."; exit 1
fi
echo "######## FOLLOW-UPS DONE in $((el / 3600))h$(printf %02d $((el % 3600 / 60)))m ########"
