#!/usr/bin/env bash
# Full E0-E7 sweep: 3 models x 3 seeds. Run on a single-GPU RunPod A40/A100.
#   SEEDS="0" MODELS="efficientnet_b0" bash scripts/run_experiments.sh   # exploratory
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
MODELS=${MODELS:-"resnet50 efficientnet_b0 vit_s16"}
E=configs/experiments

RUN=0
train() {
  RUN=$((RUN + 1))
  echo; echo "=================== [run $RUN] $(date '+%Y-%m-%d %H:%M:%S')  $(basename "$1" .yaml)  ${*:2} ==================="
  python -m src.train --config "$@"
}
phase() { echo; echo "##### $(date '+%H:%M:%S')  $* #####"; }

phase "Phase A: E0, E1, E2, E7 (models: $MODELS | seeds: $SEEDS)"
for m in $MODELS; do for s in $SEEDS; do
  train $E/e0_baseline.yaml      model=$m seed=$s
  train $E/e1_strong_aug.yaml    model=$m seed=$s
  train $E/e2_composite.yaml     model=$m seed=$s
  train $E/e7_plantdoc_only.yaml model=$m seed=$s
done; done

# Generative augmentation (once, shared by all classifiers)
phase "Phase B: CycleGAN + CVAE (skipped if already generated), FID"
[ -f data/generated/cyclegan/manifest.csv ] || { python -m src.train_cyclegan && python -m src.generate cyclegan; }
[ -f data/generated/cvae/manifest.csv ]     || { python -m src.train_cvae     && python -m src.generate cvae; }
python -m src.fid

phase "Phase C: E3, E4, E5 (k = 5, 10, 25)"
for m in $MODELS; do for s in $SEEDS; do
  train $E/e3_cyclegan.yaml model=$m seed=$s
  train $E/e4_cvae.yaml     model=$m seed=$s
  for k in 5 10 25; do train $E/e5_fewshot.yaml model=$m seed=$s data.fewshot_k=$k; done
done; done

# E6: best of E1-E4 per model, chosen on PlantDoc-VAL (never test)
phase "Phase D: select best of E1-E4 on PlantDoc-val, then E6"
python -m src.report --select
for m in $MODELS; do
  best=$(python - "$m" <<'PY'
import sys
from src.report import collect
from pathlib import Path
df = collect(Path("runs")); df = df[df.model == sys.argv[1]]
print(df[df.exp.isin(["e1", "e2", "e3", "e4"])].groupby("exp").pd_val_acc.mean().idxmax())
PY
)
  for s in $SEEDS; do train $E/e6_best_combo.yaml model=$m seed=$s init_experiment=$best; done
done

phase "Phase E: results table"
python -m src.report
