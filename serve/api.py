"""FastAPI service: POST /predict (multipart image) -> top-3 + unsure flag.

    uvicorn serve.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import os
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from serve.predictor import Predictor

app = FastAPI(title="crop-disease")
predictor = Predictor(threads=int(os.environ.get("ORT_THREADS", 0)) or None)
UNSURE = float(os.environ.get("UNSURE_THRESHOLD", 0.5))


@app.get("/health")
def health():
    return {"status": "ok", "n_classes": len(predictor.meta["classes"])}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="not an image")
    t = time.perf_counter()
    out = predictor.predict(img, top_k=3, unsure_threshold=UNSURE)
    out["latency_ms"] = (time.perf_counter() - t) * 1e3
    return out
