#!/usr/bin/env bash
# PlantDoc (CC BY 4.0) from GitHub. PlantVillage is pulled from the HF hub by src.data.prepare.
set -euo pipefail
mkdir -p data/raw
if [ ! -d data/raw/PlantDoc-Dataset ]; then
  git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/raw/PlantDoc-Dataset
fi
echo "PlantDoc train classes: $(ls data/raw/PlantDoc-Dataset/train | wc -l)  test classes: $(ls data/raw/PlantDoc-Dataset/test | wc -l)"
