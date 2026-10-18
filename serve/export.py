"""Export the best EfficientNet-B0 to ONNX, INT8 static quantization, then FP32 vs INT8 report.

    python -m serve.export --config configs/deploy.yaml

EfficientNet (SiLU + squeeze-excite) is fragile under naive INT8, so several standard quantization variants are
tried (calibration method x which ops are quantized). The variant is chosen on PlantDoc-VAL accuracy; PlantDoc-test
accuracy is reported only for FP32 and the chosen variant. Calibration: 500 PlantDoc *train-split* images.
-> reports/deploy.md + reports/deploy.json (all variants listed).
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
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
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
    step(f"INT8 static quantization: {len(dcfg.quant_variants)} variants, calibration on {dcfg.calibration_images} "
         f"PlantDoc-train images, selection on PlantDoc-val")
    quant_pre_process(str(fp32), str(pre))
    calib = read_manifest(cfg, "pd_train").sample(n=dcfg.calibration_images, random_state=dcfg.calibration_seed)
    val, test = read_manifest(cfg, "pd_val"), read_manifest(cfg, "pd_test")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = dcfg.latency.threads
    x = np.random.default_rng(0).standard_normal((1, 3, size, size)).astype(np.float32)

    def evaluate(path: Path, tag: str, on_test: bool) -> dict:
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        p50, p95 = latency_ms(sess, x, dcfg.latency.warmup, dcfg.latency.iters)
        r = {"pd_val_acc": accuracy(sess, val.path, val.label, size, mean, std, f"{tag} val"),
             "size_mb": path.stat().st_size / 2**20, "latency_p50_ms": p50, "latency_p95_ms": p95}
        if on_test:
            r["pd_test_acc"] = accuracy(sess, test.path, test.label, size, mean, std, f"{tag} test")
        return r

    variants = {}
    for v in dcfg.quant_variants:
        path = out / f"model_int8_{v.name}.onnx"
        extra = {"ActivationSymmetric": False, "WeightSymmetric": True}
        if v.method == "Percentile":
            extra["CalibPercentile"] = 99.999
        ops = ["Conv", "Gemm", "MatMul"] if v.ops == "conv" else None
        # Histogram calibrators (Percentile, Entropy) keep every activation of every calibration image in RAM:
        # 500 EfficientNet images exceed a 50 GB pod, so they use a smaller subset of the same calibration draw.
        n_cal = len(calib) if v.method == "MinMax" else min(len(calib), int(dcfg.calibration_images_histogram))
        print(f"[deploy] variant {v.name}: calibration {v.method} on {n_cal} images, quantized ops "
              f"{'Conv/Gemm/MatMul only' if ops else 'all'}", flush=True)
        try:
            quantize_static(str(pre), str(path), ImageReader(calib.path.tolist()[:n_cal], "input", size, mean, std),
                            quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8,
                            weight_type=QuantType.QInt8, calibrate_method=getattr(CalibrationMethod, v.method),
                            op_types_to_quantize=ops, extra_options=extra)
            variants[v.name] = {"method": v.method, "ops": v.ops, "calibration_images": n_cal,
                                **evaluate(path, v.name, on_test=False)}
            print(f"[deploy]   {v.name}: PlantDoc-val acc {variants[v.name]['pd_val_acc']:.3f}, "
                  f"p50 {variants[v.name]['latency_p50_ms']:.1f} ms", flush=True)
        except Exception as e:  # one failing variant must not sink the export
            print(f"[deploy]   {v.name} failed: {e}", flush=True)
    pre.unlink()
    if not variants:
        raise SystemExit("every quantization variant failed")
    best = max(variants, key=lambda k: (round(variants[k]["pd_val_acc"], 3), -variants[k]["latency_p50_ms"]))
    (out / f"model_int8_{best}.onnx").replace(int8)
    for k in variants:
        (out / f"model_int8_{k}.onnx").unlink(missing_ok=True)

    step(f"benchmark FP32 vs INT8 ({best}): PlantDoc val + test accuracy, size, latency ({dcfg.latency.iters} runs, "
         f"batch 1, {dcfg.latency.threads} threads)")
    report = {"cpu": platform.processor() or platform.machine(), "threads": dcfg.latency.threads,
              "onnxruntime": ort.__version__, "n_val": len(val), "n_test": len(test),
              "checkpoint": str(dcfg.checkpoint), "int8_variant": best, "variants": variants,
              "fp32": evaluate(fp32, "fp32", on_test=True), "int8": evaluate(int8, f"int8 {best}", on_test=True)}
    drop = report["fp32"]["pd_val_acc"] - report["int8"]["pd_val_acc"]
    app_model = "model_int8.onnx" if drop <= float(dcfg.max_int8_val_drop) else "model_fp32.onnx"
    report["app_model"] = app_model
    meta["app_model"] = app_model
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    save_json(report, "reports/deploy.json")
    md = (f"Deployed checkpoint: `{dcfg.checkpoint}` (best EfficientNet-B0 by PlantDoc-val accuracy).\n"
          f"The demo serves **{app_model}** (INT8 is used only if it loses at most "
          f"{float(dcfg.max_int8_val_drop):.0%} PlantDoc-val accuracy; it lost {drop:.1%}).\n\n"
          "| Model | PlantDoc-val acc | PlantDoc-test acc | Size (MB) | CPU p50 (ms) | CPU p95 (ms) |\n|---|---|---|---|---|---|\n"
          + "".join(f"| EfficientNet-B0 {name} | {report[t]['pd_val_acc']:.3f} | {report[t]['pd_test_acc']:.3f} | "
                    f"{report[t]['size_mb']:.1f} | {report[t]['latency_p50_ms']:.1f} | {report[t]['latency_p95_ms']:.1f} |\n"
                    for t, name in (("fp32", "FP32"), ("int8", f"INT8 ({best})")))
          + f"\nCPU: {report['cpu']}, {report['threads']} threads, batch 1, onnxruntime {report['onnxruntime']}.\n\n"
          "INT8 variants tried (chosen on PlantDoc-val; test not used for the choice):\n\n"
          "| Variant | Calibration | Calib. images | Quantized ops | PlantDoc-val acc | Size (MB) | p50 (ms) |\n|---|---|---|---|---|---|---|\n"
          + "".join(f"| {k} | {v['method']} | {v['calibration_images']} | {'Conv/Gemm/MatMul' if v['ops'] == 'conv' else 'all'} | {v['pd_val_acc']:.3f} | "
                    f"{v['size_mb']:.1f} | {v['latency_p50_ms']:.1f} |\n" for k, v in variants.items()))
    Path("reports/deploy.md").write_text(md)
    print(md)
    step.done("wrote reports/deploy.md, reports/deploy.json")


if __name__ == "__main__":
    main()
