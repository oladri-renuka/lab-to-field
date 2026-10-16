#!/usr/bin/env bash
# Overnight: CycleGAN + CVAE -> generated images -> FID -> E3 + E4 (one model/seed) -> results table.
#   bash scripts/overnight_generative.sh            # defaults: efficientnet_b0, seed 0
#   MODEL=resnet50 SEED=1 bash scripts/overnight_generative.sh
# Everything is also written to logs/overnight_<date>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-efficientnet_b0}
SEED=${SEED:-0}
mkdir -p logs
LOG=logs/overnight_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1

T0=$(date +%s)
N=6
stage() {
  local el=$(( $(date +%s) - T0 ))
  printf '\n######## [stage %s/%s] %s   (started %s, total elapsed %dh%02dm) ########\n' \
    "$1" "$N" "$2" "$(date '+%H:%M:%S')" $((el / 3600)) $((el % 3600 / 60))
}
fail() { echo; echo "!!!!!!!! FAILED at stage $1: $2 -- see $LOG"; exit 1; }

echo "log: $LOG | model=$MODEL seed=$SEED"
if pgrep -f "src.train |src.train_cyclegan|src.train_cvae|src.generate" >/dev/null; then
  echo "another training/generation process is running:"; pgrep -af "src.train |src.train_cyclegan|src.train_cvae|src.generate"
  exit 1
fi

if [ -f data/generated/cyclegan/manifest.csv ]; then
  stage 1 "CycleGAN: already generated, skipping training + translation"
else
  stage 1 "CycleGAN: train lab->field (100 epochs)"
  python -m src.train_cyclegan || fail 1 "CycleGAN training"
  stage 2 "CycleGAN: translate every PlantVillage training image"
  python -m src.generate cyclegan || fail 2 "CycleGAN generation"
fi

if [ -f data/generated/cvae/manifest.csv ]; then
  stage 3 "CVAE: already generated, skipping training + sampling"
else
  stage 3 "CVAE: train one model per crop group, then sample 500 field-style images per class"
  python -m src.train_cvae || fail 3 "CVAE training"
  python -m src.generate cvae || fail 3 "CVAE sampling"
fi

stage 4 "FID of generated images vs PlantDoc train + sample grids"
python -m src.fid || fail 4 "FID"

stage 5 "E3 (+CycleGAN) and E4 (+CVAE): $MODEL seed $SEED"
for c in e3_cyclegan e4_cvae; do
  python -m src.train --config configs/experiments/$c.yaml model=$MODEL seed=$SEED || fail 5 "$c"
done

stage 6 "results table"
python -m src.report || fail 6 "report"
el=$(( $(date +%s) - T0 ))
printf '\n######## ALL DONE in %dh%02dm. Log: %s ########\n' $((el / 3600)) $((el % 3600 / 60)) "$LOG"
