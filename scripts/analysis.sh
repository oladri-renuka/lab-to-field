#!/usr/bin/env bash
# Everything after the sweep + follow-ups, unattended (~4-5 h on one A40):
#   learning curves, failure analysis, saliency figures, final report, then compute experiments (no DDP).
#   bash scripts/analysis.sh 2>&1 | tee logs/analysis_$(date +%m%d_%H%M).log
set -uo pipefail
cd "$(dirname "$0")/.."
FAILED=()
step() { echo; echo "##### $(date '+%H:%M:%S')  $* #####"; }

step "1/5 lab vs field validation curves"
python -m src.curves || FAILED+=("curves")

step "2/5 failure analysis (from saved predictions)"
python -m src.failure || FAILED+=("failure")

step "3/5 saliency: Grad-CAM (CNNs) / attention rollout (ViT) on 20 fixed PlantDoc-test images"
for m in resnet50 efficientnet_b0 vit_s16 vit_s16_in1k; do
  r=()
  for e in e0 e2 e3 e5_k10; do [ -f "runs/$e/$m/seed0/best.pt" ] && r+=("runs/$e/$m/seed0"); done
  [ ${#r[@]} -gt 0 ] || continue
  python -m src.explain --runs "${r[@]}" --out "reports/figures/saliency_$m.png" || FAILED+=("saliency $m")
done

step "4/5 results table"
python -m src.report > /dev/null || FAILED+=("report")

step "5/5 compute experiments (AMP, gradient checkpointing, profiler)"
bash scripts/compute_experiments.sh || FAILED+=("compute experiments")

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "######## ANALYSIS FINISHED WITH FAILURES:"; printf '   - %s\n' "${FAILED[@]}"; exit 1
fi
echo "######## ANALYSIS DONE: reports/results.md, reports/failure_analysis.md, reports/val_curves.md, reports/figures/ ########"
