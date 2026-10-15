#!/usr/bin/env bash
# Full E0-E7 sweep: 3 models x 3 seeds. Run on a single-GPU RunPod A40/A100.
#   SEEDS="0" MODELS="efficientnet_b0" bash scripts/run_experiments.sh   # exploratory
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
MODELS=${MODELS:-"resnet50 efficientnet_b0 vit_s16"}
E=configs/experiments

train() { python -m src.train --config "$@"; }

for m in $MODELS; do for s in $SEEDS; do
  train $E/e0_baseline.yaml      model=$m seed=$s
  train $E/e1_strong_aug.yaml    model=$m seed=$s
  train $E/e2_composite.yaml     model=$m seed=$s
  train $E/e7_plantdoc_only.yaml model=$m seed=$s
done; done

# Generative augmentation (once, shared by all classifiers)
[ -f data/generated/cyclegan/manifest.csv ] || { python -m src.train_cyclegan && python -m src.generate cyclegan; }
[ -f data/generated/cvae/manifest.csv ]     || { python -m src.train_cvae     && python -m src.generate cvae; }
python -m src.fid

for m in $MODELS; do for s in $SEEDS; do
  train $E/e3_cyclegan.yaml model=$m seed=$s
  train $E/e4_cvae.yaml     model=$m seed=$s
  for k in 5 10 25; do train $E/e5_fewshot.yaml model=$m seed=$s data.fewshot_k=$k; done
done; done

# E6: best of E1-E4 per model, chosen on PlantDoc-VAL (never test)
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

python -m src.report
