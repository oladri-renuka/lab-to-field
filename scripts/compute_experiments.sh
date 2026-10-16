#!/usr/bin/env bash
# Compute-cost experiments. AMP / checkpointing / profiling: 1 GPU. DDP: a 4-GPU RunPod pod.
set -euo pipefail
C=configs/compute
RUN=0
run() { RUN=$((RUN + 1)); echo; echo "=================== [compute run $RUN] $(date '+%H:%M:%S')  $* ==================="; "$@"; }
for m in resnet50 vit_s16; do
  run python -m src.train --config $C/amp_fp32.yaml model=$m
  run python -m src.train --config $C/amp_bf16.yaml model=$m
done
run python -m src.train --config $C/grad_ckpt.yaml train.grad_checkpointing=false
run python -m src.train --config $C/grad_ckpt.yaml train.grad_checkpointing=true
run python -m src.train --config $C/profile.yaml model=resnet50

NGPU=$(nvidia-smi -L | wc -l)
for n in 1 2 4; do
  [ "$n" -le "$NGPU" ] || continue
  run torchrun --standalone --nproc_per_node=$n -m src.train --config $C/ddp.yaml exp_id=ddp_ws$n
done
python -m src.report
