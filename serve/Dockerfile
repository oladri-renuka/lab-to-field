# CPU inference image for the FastAPI service. Build from the repo root:
#   docker build -f serve/Dockerfile -t crop-disease .
#   docker run -p 8000:8000 crop-disease
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/serve/artifacts ORT_THREADS=4
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir onnxruntime==1.20.1 "numpy>=1.26,<3" "pillow>=10.0" fastapi==0.115.6 \
    "uvicorn[standard]==0.32.1" python-multipart==0.0.20
COPY serve/__init__.py serve/predictor.py serve/api.py serve/
COPY serve/artifacts/meta.json serve/artifacts/model_int8.onnx serve/artifacts/
EXPOSE 8000
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
