---
title: Crop Disease Field Demo
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: cc-by-4.0
---

# Crop disease detection (field photos)

Upload a photo of a single leaf: the model returns the top-3 diseases with confidence and a Grad-CAM heat map.
Below 50% confidence it answers **"unsure"** instead of guessing.

- Model: EfficientNet-B0, INT8 ONNX, chosen by validation accuracy on PlantDoc field photos.
- Covers 27 crop/disease classes shared by PlantVillage and PlantDoc only; anything else is forced into one of them.
- Research demo, not a diagnostic tool. Code and results: https://github.com/oladri-renuka/lab-to-field
