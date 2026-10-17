#!/usr/bin/env bash
# Full E0-E7 sweep: 3 models x 3 seeds. Run on a single-GPU RunPod A40/A100.
#   SEEDS="0" MODELS="efficientnet_b0" bash scripts/run_experiments.sh   # exploratory
# Resumable: finished runs (metrics.json present) are skipped. A failed run is recorded and the sweep continues;
# failures are listed at the end. Re-run the same command after fixing to do only what is missing.
set -uo pipefail
SEEDS=${SEEDS:-"0 1 2"}
MODELS=${MODELS:-"resnet50 efficientnet_b0 vit_s16"}
E=configs/experiments
export SKIP_FINISHED=1
T0=$(date +%s)

RUN=0
FAILED=()
train() {
  RUN=$((RUN + 1))
  local el=$(( $(date +%s) - T0 ))
  echo; echo "=================== [run $RUN] $(date '+%Y-%m-%d %H:%M:%S') (sweep elapsed $((el / 3600))h$(printf %02d $((el % 3600 / 60)))m)  $(basename "$1" .yaml)  ${*:2} ==================="
  if ! python -m src.train --config "$@"; then
    echo "!!!!!!!! [run $RUN] FAILED: $(basename "$1" .yaml) ${*:2} -- continuing with the next run"
    FAILED+=("$(basename "$1" .yaml) ${*:2}")
  fi
}
phase() { echo; echo "##### $(date '+%H:%M:%S')  $* #####"; }
fatal() { echo "!!!!!!!! FATAL: $* -- later phases depend on this, stopping"; exit 1; }

phase "Phase A: E0, E1, E2, E7 (models: $MODELS | seeds: $SEEDS)"
for m in $MODELS; do for s in $SEEDS; do
  train $E/e0_baseline.yaml      model=$m seed=$s
  train $E/e1_strong_aug.yaml    model=$m seed=$s
  train $E/e2_composite.yaml     model=$m seed=$s
  train $E/e7_plantdoc_only.yaml model=$m seed=$s
done; done

# Generative augmentation (once, shared by all classifiers)
phase "Phase B: CycleGAN + CVAE (skipped if already generated), FID"
[ -f data/generated/cyclegan/manifest.csv ] || { python -m src.train_cyclegan && python -m src.generate cyclegan; } || fatal "CycleGAN"
[ -f data/generated/cvae/manifest.csv ]     || { python -m src.train_cvae     && python -m src.generate cvae; }     || fatal "CVAE"
python -m src.fid || echo "!!!!!!!! FID failed (not needed by later phases) -- continuing"

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
python -m src.report || echo "!!!!!!!! report failed"
el=$(( $(date +%s) - T0 ))
if [ ${#FAILED[@]} -gt 0 ]; then
  echo; echo "######## SWEEP FINISHED WITH ${#FAILED[@]} FAILED RUN(S) in $((el / 3600))h$(printf %02d $((el % 3600 / 60)))m:"
  printf '   - %s\n' "${FAILED[@]}"
  echo "Fix the cause, then re-run the same command: finished runs are skipped."
  exit 1
fi
echo; echo "######## SWEEP DONE in $((el / 3600))h$(printf %02d $((el % 3600 / 60)))m, no failures ########"
