#!/usr/bin/env bash
# Copy small result files (no checkpoints) off the pod to the `results` branch on GitHub.
# Uses a separate clone, so the working repo and `main` are never touched.
#   bash scripts/push_results.sh                                   # once
#   while true; do bash scripts/push_results.sh; sleep 1800; done   # every 30 min, in a spare terminal
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(pwd)
DST=${RESULTS_CLONE:-/workspace/lab-to-field-results}
REMOTE=$(git remote get-url origin)

if [ ! -d "$DST/.git" ]; then
  git clone -q "$REMOTE" "$DST"
  cd "$DST"
  if git ls-remote --exit-code --heads origin results >/dev/null 2>&1; then
    git checkout -q -B results origin/results
  else
    git checkout -q --orphan results
    git rm -rq --cached . >/dev/null 2>&1 || true
    find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
  fi
fi
cd "$DST"
git pull -q --rebase origin results 2>/dev/null || true

mkdir -p snapshot
cd "$SRC"
for d in runs runs_exploratory; do
  [ -d "$d" ] || continue
  find "$d" -type f \( -name 'metrics*.json' -o -name 'train_log.json' -o -name 'config.yaml' -o -name 'config_hash.txt' \
       -o -name 'preds_*.npz' -o -name 'profile_top5.txt' -o -path '*/samples/*.jpg' \) -print0 \
    | xargs -0 -r cp --parents -t "$DST/snapshot/"
done
[ -d reports ] && cp -r reports "$DST/snapshot/"
[ -d logs ] && cp -r logs "$DST/snapshot/"
# mlflow.db is not pushed: binary SQLite trips GitHub secret scanning (false positives) and every number is in the JSON files.
[ -f data/processed/classes.json ] && mkdir -p "$DST/snapshot/data" && cp data/processed/classes.json data/processed/*.json "$DST/snapshot/data/" 2>/dev/null || true

cd "$DST"
git add -A snapshot
if ! git diff --cached --quiet; then
  git commit -q -m "Results snapshot $(date '+%Y-%m-%d %H:%M')"
fi
# Push whenever the local branch is ahead (also retries an earlier commit whose push failed).
if git ls-remote --exit-code --heads origin results >/dev/null 2>&1 \
   && [ "$(git rev-parse HEAD)" = "$(git ls-remote origin refs/heads/results | cut -f1)" ]; then
  echo "[push-results] $(date '+%H:%M:%S') nothing new"
  exit 0
fi
git push -q -u origin results
echo "[push-results] $(date '+%H:%M:%S') pushed $(git rev-parse --short HEAD) to the results branch"
