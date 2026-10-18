#!/usr/bin/env bash
# Compute-cost experiments (single seed each; they measure cost, not accuracy differences).
#   bash scripts/compute_experiments.sh              # AMP, grad checkpointing, profiler (1 GPU)
#   ONLY=ddp bash scripts/compute_experiments.sh     # DDP 1/2/4 GPUs, on a 4-GPU pod
# Resumable: finished runs are skipped. A failed run is recorded and the rest continue.
set -uo pipefail
cd "$(dirname "$0")/.."
C=configs/compute
ONLY=${ONLY:-single}
export SKIP_FINISHED=1
RUN=0
FAILED=()
run() {
  RUN=$((RUN + 1))
  echo; echo "=================== [compute run $RUN] $(date '+%H:%M:%S')  ${*:4} ==================="
  "$@" || FAILED+=("${*:4}")
}

if [ "$ONLY" = "single" ]; then
  for m in resnet50 vit_s16; do
    run python -m src.train --config $C/amp_fp32.yaml model=$m
    run python -m src.train --config $C/amp_bf16.yaml model=$m
  done
  run python -m src.train --config $C/grad_ckpt.yaml train.grad_checkpointing=false
  run python -m src.train --config $C/grad_ckpt.yaml train.grad_checkpointing=true
  run python -m src.train --config $C/profile.yaml model=resnet50
fi

if [ "$ONLY" = "ddp" ]; then
  # On RunPod's multi-A40 pods NCCL peer-to-peer hangs at the first collective (100% GPU util, no progress);
  # routing through host memory fixes it. Override with NCCL_P2P_DISABLE=0 on hardware with working P2P/NVLink.
  export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
  NGPU=$(nvidia-smi -L | wc -l)
  echo "[ddp] NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
  echo "[ddp] $NGPU GPU(s) visible"
  for n in 1 2 4; do
    [ "$n" -le "$NGPU" ] || { echo "[ddp] skipping world size $n (only $NGPU GPUs)"; continue; }
    run torchrun --standalone --nproc_per_node=$n -m src.train --config $C/ddp.yaml exp_id=ddp_ws$n
  done
fi

python -m src.report > /dev/null || FAILED+=("report")
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "######## COMPUTE EXPERIMENTS FINISHED WITH FAILURES:"; printf '   - %s\n' "${FAILED[@]}"; exit 1
fi
echo "######## COMPUTE EXPERIMENTS DONE ($ONLY) -- see the 'Compute experiments' section of reports/results.md ########"
