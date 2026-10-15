# Notebooks

Analysis only: reading `runs/*/metrics.json`, `reports/*.csv`, plotting, and browsing failure cases.
**No training happens in notebooks.** Every number in the README comes from a config-launched run
(`src/train.py`) and is aggregated by `src/report.py`.

Suggested notebooks:
- `01_data_audit.ipynb`: class counts per split, near-duplicate pairs from `reports/dedup_plantdoc.csv`, sample grids.
- `02_failure_analysis.ipynb`: per-class recall from `metrics.json`, confusion matrices, misclassified PlantDoc-test images.
- `03_calibration.ipynb`: reliability diagrams per experiment.
