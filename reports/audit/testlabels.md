## 4c. Full PlantDoc test-label review

All 236 test images: `reports/audit/labels/plantdoc_test_all/<class>.jpg`. In `test_label_review.csv` set `correct` to 1 (label right) or 0 (wrong). For a wrong one, put the right class name (exactly as in `data/processed/classes.json`) in `corrected_class`, or leave it empty if the image fits no class / is unusable. `python -m src.report` then adds accuracy on the reviewed labels, using saved predictions (no re-training).
