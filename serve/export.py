"""Export the best EfficientNet-B0 to ONNX, INT8 static quantization, then FP32 vs INT8 report.

    python -m serve.export --config configs/deploy.yaml

Calibration: 500 PlantDoc *train-split* images. Reports PlantDoc-test accuracy, size (MB) and CPU
latency p50/p95 at batch 1 -> reports/deploy.md + reports/deploy.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from serve.predictor import preprocess
from src.data.io import load_image
from src.data.datasets import read_manifest
from src.eval import load_checkpoint
from src.models.classifier import norm_stats
from src.utils import Steps, load_config, progress, save_json


class ImageReader(CalibrationDataReader):
    def __init__(self, paths, input_name, size, mean, std):
        self.it = iter(progress(paths, "calibrate INT8", unit="img"))
        self.args = (input_name, size, mean, std)

    def get_next(self):
        p = next(self.it, None)
        if p is None:
            return None
        name, size, mean, std = self.args
        return {name: preprocess(load_image(p), size, mean, std)}


def latency_ms(session: ort.InferenceSession, x: np.ndarray, warmup: int, iters: int) -> tuple[float, float]:
    name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {name: x})
    t = []
    for _ in range(iters):
        s = time.perf_counter()
        session.run(None, {name: x})
        t.append((time.perf_counter() - s) * 1e3)
    return float(np.percentile(t, 50)), float(np.percentile(t, 95))


def accuracy(session, paths, labels, size, mean, std, desc: str = "accuracy") -> float:
    name = session.get_inputs()[0].name
    preds = [int(session.run(None, {name: preprocess(load_image(p), size, mean, std)})[0].argmax()) for p in progress(list(paths), desc, unit="img")]
    return float((np.asarray(preds) == np.asarray(labels)).mean())


def pick_checkpoint(model: str, runs: Path = Path("runs")) -> str:
    """Best run of `model` by PlantDoc-VAL accuracy of its selected checkpoint (test is never looked at)."""
    best = None
    for f in sorted(runs.glob(f"*/{model}/seed*/metrics.json")):
        m = json.loads(f.read_text())
        if str(m.get("exp_id", "")).startswith(("compute_", "ddp")) or not (f.parent / "best.pt").exists():
            continue
        val = m.get("val_acc", {})
        acc = val.get("pd_val") if isinstance(val, dict) else m.get("pd_val_acc")
        if acc is not None and (best is None or acc > best[0]):
            best = (acc, f.parent / "best.pt", m["exp_id"])
    if best is None:
        raise SystemExit(f"no {model} run with a checkpoint under {runs}")
    print(f"[deploy] auto-selected {best[1]} ({best[2]}, PlantDoc-val acc {best[0]:.3f})", flush=True)
    return str(best[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deploy.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    dcfg = load_config(args.config, args.overrides)
    out = Path(dcfg.export_dir)
    out.mkdir(parents=True, exist_ok=True)

    step = Steps("deploy", 4)
    if str(dcfg.checkpoint) == "auto":
        dcfg.checkpoint = pick_checkpoint(str(dcfg.deploy_model))
    step(f"loading {dcfg.checkpoint}")
    model, cfg, ckpt = load_checkpoint(dcfg.checkpoint, "cpu")
    model.eval()
    mean, std = norm_stats(model)
    size = int(cfg.data.img_size)
    meta = {"classes": ckpt["classes"], "mean": list(mean), "std": list(std), "img_size": size,
            "timm_name": cfg.models[cfg.model].timm_name, "source_checkpoint": str(dcfg.checkpoint)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    torch.save(model.state_dict(), out / "model_fp32.pt")  # for Grad-CAM in the app

    fp32, pre, int8 = out / "model_fp32.onnx", out / "model_fp32_pre.onnx", out / "model_int8.onnx"
    step("exporting ONNX FP32")
    torch.onnx.export(model, torch.randn(1, 3, size, size), fp32, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=dcfg.opset, dynamo=False)
    step(f"INT8 static quantization, calibrating on {dcfg.calibration_images} PlantDoc-train images")
    quant_pre_process(str(fp32), str(pre))
    calib = read_manifest(cfg, "pd_train").sample(n=dcfg.calibration_images, random_state=dcfg.calibration_seed)
    quantize_static(str(pre), str(int8), ImageReader(calib.path.tolist(), "input", size, mean, std),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink()

    test = read_manifest(cfg, "pd_test")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = dcfg.latency.threads
    x = np.random.default_rng(0).standard_normal((1, 3, size, size)).astype(np.float32)
    report = {"cpu": platform.processor() or platform.machine(), "threads": dcfg.latency.threads,
              "onnxruntime": ort.__version__, "n_test": len(test)}
    step(f"benchmark FP32 vs INT8: PlantDoc-test accuracy, size, latency ({dcfg.latency.iters} runs, batch 1, "
         f"{dcfg.latency.threads} threads)")
    for tag, path in (("fp32", fp32), ("int8", int8)):
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        p50, p95 = latency_ms(sess, x, dcfg.latency.warmup, dcfg.latency.iters)
        report[tag] = {"pd_test_acc": accuracy(sess, test.path, test.label, size, mean, std, f"{tag} accuracy"),
                       "size_mb": path.stat().st_size / 2**20, "latency_p50_ms": p50, "latency_p95_ms": p95}
    save_json(report, "reports/deploy.json")
    md = ("| Model | PlantDoc acc | Size (MB) | p50 (ms) | p95 (ms) |\n|---|---|---|---|---|\n"
          + "".join(f"| EfficientNet-B0 {t.upper()} | {report[t]['pd_test_acc']:.3f} | {report[t]['size_mb']:.1f} | "
                    f"{report[t]['latency_p50_ms']:.1f} | {report[t]['latency_p95_ms']:.1f} |\n" for t in ("fp32", "int8"))
          + f"\nCPU: {report['cpu']}, {report['threads']} threads, batch 1, onnxruntime {report['onnxruntime']}.\n")
    Path("reports/deploy.md").write_text(md)
    print(md)
    step.done("wrote reports/deploy.md, reports/deploy.json")


if __name__ == "__main__":
    main()
