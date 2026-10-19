"""Gradio demo (Hugging Face Spaces): photo of a leaf -> top-3 diseases + Grad-CAM overlay."""
from __future__ import annotations

import os

import gradio as gr
import numpy as np
from matplotlib import colormaps
from PIL import Image

from serve.predictor import Predictor

UNSURE = float(os.environ.get("UNSURE_THRESHOLD", 0.5))
predictor = Predictor()

try:  # Hugging Face ZeroGPU requires at least one @spaces.GPU function; elsewhere this is a no-op
    import spaces

    on_gpu = spaces.GPU(duration=20)
except ImportError:
    def on_gpu(fn):
        return fn


@on_gpu
def cam_overlay(img: Image.Image) -> Image.Image | None:
    cam = predictor.grad_cam(img)
    if cam is None:
        return None
    base = np.asarray(img.convert("RGB"), dtype=float) / 255
    return Image.fromarray((255 * (0.55 * base + 0.45 * colormaps["jet"](cam)[..., :3])).astype(np.uint8))


def pretty(label: str) -> str:
    crop, _, disease = label.partition("___")
    return f"{crop.split('_(')[0].replace('_', ' ')} — {disease.replace('_', ' ').strip()}"


def run(img: Image.Image):
    if img is None:
        return "Upload a photo of a single leaf.", {}, None
    out = predictor.predict(img, top_k=3, unsure_threshold=UNSURE)
    scores = {pretty(t["label"]): t["confidence"] for t in out["top_k"]}
    top = out["top_k"][0]
    if out["unsure"]:
        verdict = (f"**Unsure** (top confidence {top['confidence']:.0%} < {UNSURE:.0%}). "
                   "Try a closer, well-lit photo of one leaf.")
    else:
        verdict = f"**{pretty(top['label'])}** ({top['confidence']:.0%})"
    return verdict, scores, cam_overlay(img)


demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Leaf photo"),
    outputs=[gr.Markdown(label="Prediction"), gr.Label(num_top_classes=3, label="Top-3"), gr.Image(label="Grad-CAM")],
    title="Crop disease detection (field-robust)",
    description=(f"EfficientNet-B0 ({predictor.meta.get('app_model', 'model_int8.onnx').replace('model_', '').replace('.onnx', '').upper()} "
                 f"ONNX, CPU), checkpoint `{predictor.meta.get('source_checkpoint', '?')}`, chosen by accuracy on PlantDoc field "
                 f"validation photos. Covers {len(predictor.meta['classes'])} crop/disease classes shared by PlantVillage and "
                 "PlantDoc; anything else is forced into one of them. Expect about 54% top-1 accuracy on field photos "
                 "(PlantDoc test), so wrong or 'unsure' answers are normal. Below 50% confidence it answers 'unsure'. "
                 "Research demo, not a substitute for an agronomist."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
