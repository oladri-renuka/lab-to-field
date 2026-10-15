"""Inference used by both the Gradio app and the FastAPI service. Depends only on onnxruntime/numpy/PIL;
Grad-CAM additionally needs torch + timm and the FP32 weights (optional)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ARTIFACTS = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "artifacts"))


def preprocess(img: Image.Image, size: int, mean, std) -> np.ndarray:
    x = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return x.transpose(2, 0, 1)[None]


class Predictor:
    def __init__(self, model_dir: Path = ARTIFACTS, model_file: str = "model_int8.onnx", threads: int | None = None):
        self.meta = json.loads((model_dir / "meta.json").read_text())
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / model_file), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.model_dir = model_dir
        self._cam_model = None

    def probs(self, img: Image.Image) -> np.ndarray:
        x = preprocess(img, self.meta["img_size"], self.meta["mean"], self.meta["std"])
        logits = self.session.run(None, {self.input: x})[0][0]
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def predict(self, img: Image.Image, top_k: int = 3, unsure_threshold: float = 0.5) -> dict:
        p = self.probs(img)
        order = np.argsort(-p)[:top_k]
        top = [{"label": self.meta["classes"][i], "confidence": float(p[i])} for i in order]
        return {"top_k": top, "unsure": top[0]["confidence"] < unsure_threshold}

    def grad_cam(self, img: Image.Image) -> np.ndarray | None:
        """Heatmap in [0, 1] at input resolution, from the FP32 PyTorch weights (ONNX has no gradients)."""
        weights = self.model_dir / "model_fp32.pt"
        if not weights.exists():
            return None
        import timm
        import torch
        import torch.nn.functional as F

        if self._cam_model is None:
            m = timm.create_model(self.meta["timm_name"], pretrained=False, num_classes=len(self.meta["classes"]))
            m.load_state_dict(torch.load(weights, map_location="cpu"))
            self._cam_model = m.eval()
        x = torch.from_numpy(preprocess(img, self.meta["img_size"], self.meta["mean"], self.meta["std"]))
        feats = self._cam_model.forward_features(x)
        feats.retain_grad()
        logits = self._cam_model.forward_head(feats)
        logits[0, logits.argmax(1)].sum().backward()
        cam = F.relu((feats.grad.mean((2, 3), keepdim=True) * feats).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=img.size[::-1], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().numpy()
