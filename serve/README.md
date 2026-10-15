# Serving

```bash
python -m serve.export --config configs/deploy.yaml   # ONNX FP32 -> INT8 (static, 500 PlantDoc-train calib images) + report
python -m serve.app                                   # Gradio demo at http://127.0.0.1:7860
uvicorn serve.api:app --port 8000                     # FastAPI: POST /predict, GET /health
docker build -f serve/Dockerfile -t crop-disease . && docker run -p 8000:8000 crop-disease
```

`serve/artifacts/` (git-ignored) holds `model_int8.onnx`, `model_fp32.onnx`, `model_fp32.pt` (for Grad-CAM) and `meta.json`.

## Hugging Face Spaces

Create a Gradio Space and push: `serve/__init__.py`, `serve/predictor.py`, `serve/app.py`,
`serve/artifacts/{meta.json,model_int8.onnx,model_fp32.pt}`, plus a top-level `app.py` containing
`from serve.app import demo; demo.launch()` and `requirements.txt` copied from `serve/requirements.txt`.
Predictions below 0.5 confidence are shown as **Unsure** rather than a likely-wrong label (`UNSURE_THRESHOLD`).
