#!/usr/bin/env bash
# One-shot setup on a fresh RunPod pod (container disk only, so this re-runs after every pod restart).
#   mkdir -p /workspace && cd /workspace && git clone https://github.com/oladri-renuka/lab-to-field.git \
#     && cd lab-to-field && bash scripts/pod_setup.sh
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[setup 1/4] system tools"
command -v make >/dev/null || { apt-get update -qq && apt-get install -y -qq make; }

grep -q HF_HOME ~/.bashrc || echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
export HF_HOME=/workspace/.cache/huggingface

echo "[setup 2/4] python packages (a few minutes)"
# Ubuntu-installed Python packages (blinker, cryptography, ...) cannot be uninstalled by pip:
# force-install just those, never the whole requirements file (that would replace the pod's CUDA torch).
while ! out=$(pip install -r requirements.txt 2>&1); do
  p=$(echo "$out" | grep -oP 'Cannot uninstall \K[^ ]+')
  [ -z "$p" ] && { echo "$out" | tail -20; exit 1; }
  echo "fixing $p"
  pip install -q --ignore-installed "$p"
done
echo "[setup] python packages installed"

echo "[setup 3/4] checking GPU"
python -c "import torch, timm; assert torch.cuda.is_available(), 'no GPU'; print('[setup] torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0), '| timm', timm.__version__)" || exit 1

echo "[setup 4/4] downloading + preparing data (~10 min)"
make data || exit 1
echo "[setup] done. Data summary: reports/data_summary.md"
