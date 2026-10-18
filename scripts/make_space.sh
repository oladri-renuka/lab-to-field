#!/usr/bin/env bash
# Assemble a Hugging Face Space from the exported model (run `python -m serve.export` first), then upload it.
#   bash scripts/make_space.sh                      # builds build/space/
#   SPACE=<user>/<space-name> bash scripts/make_space.sh   # builds and uploads (needs `hf auth login`)
set -euo pipefail
cd "$(dirname "$0")/.."
A=serve/artifacts
APP_MODEL=$(python -c "import json;print(json.load(open('$A/meta.json')).get('app_model','model_int8.onnx'))")
for f in meta.json "$APP_MODEL" model_fp32.pt; do
  [ -f "$A/$f" ] || { echo "missing $A/$f: run python -m serve.export first"; exit 1; }
done
echo "[space] serving $APP_MODEL"
OUT=build/space
rm -rf "$OUT" && mkdir -p "$OUT/serve/artifacts"
cp serve/space/README.md serve/space/app.py "$OUT/"
cp serve/requirements.txt "$OUT/requirements.txt"
echo "spaces" >> "$OUT/requirements.txt"   # ZeroGPU runtime (free-tier Gradio Spaces run on ZeroGPU)
cp serve/__init__.py serve/predictor.py serve/app.py "$OUT/serve/"
cp "$A/meta.json" "$A/$APP_MODEL" "$A/model_fp32.pt" "$OUT/serve/artifacts/"
echo "[space] built $OUT ($(du -sh "$OUT" | cut -f1))"
if [ -n "${SPACE:-}" ]; then
  # Python API rather than the CLI: the CLI was renamed (huggingface-cli -> hf) across huggingface_hub versions.
  python - "$SPACE" "$OUT" <<'PY'
import sys
from huggingface_hub import create_repo, upload_folder
space, folder = sys.argv[1], sys.argv[2]
create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
upload_folder(repo_id=space, repo_type="space", folder_path=folder, commit_message="Deploy crop disease demo")
PY
  echo "[space] uploaded: https://huggingface.co/spaces/$SPACE"
fi
