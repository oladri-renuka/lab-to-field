#!/usr/bin/env bash
# Assemble a Hugging Face Space from the exported model (run `python -m serve.export` first), then upload it.
#   bash scripts/make_space.sh                      # builds build/space/
#   SPACE=<user>/<space-name> bash scripts/make_space.sh   # builds and uploads (needs `huggingface-cli login`)
set -euo pipefail
cd "$(dirname "$0")/.."
A=serve/artifacts
for f in meta.json model_int8.onnx model_fp32.pt; do
  [ -f "$A/$f" ] || { echo "missing $A/$f: run python -m serve.export first"; exit 1; }
done
OUT=build/space
rm -rf "$OUT" && mkdir -p "$OUT/serve/artifacts"
cp serve/space/README.md serve/space/app.py "$OUT/"
cp serve/requirements.txt "$OUT/requirements.txt"
cp serve/__init__.py serve/predictor.py serve/app.py "$OUT/serve/"
cp "$A/meta.json" "$A/model_int8.onnx" "$A/model_fp32.pt" "$OUT/serve/artifacts/"
echo "[space] built $OUT ($(du -sh "$OUT" | cut -f1))"
if [ -n "${SPACE:-}" ]; then
  huggingface-cli upload "$SPACE" "$OUT" . --repo-type space --commit-message "Deploy crop disease demo"
  echo "[space] uploaded: https://huggingface.co/spaces/$SPACE"
fi
