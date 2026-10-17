from pathlib import Path

"""End-to-end smoke test on synthetic images (CPU, no downloads):
prepare-like manifests -> train E2 (composite) -> eval -> E5 fine-tune from it -> report."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

CLASSES = ["Tomato___Early_blight", "Tomato___healthy", "Potato___Late_blight"]


def make_split(root, name, n_per_class, seg=False):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    rows = []
    for label, cls in enumerate(CLASSES):
        for i in range(n_per_class):
            arr = np.zeros((48, 48, 3), np.uint8) if seg else (rng.random((48, 48, 3)) * 80).astype(np.uint8)
            arr[12:36, 12:36] = (40 + 80 * label, 150, 40)
            p = root / name / f"{label}_{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(p)
            rows.append({"path": str(p), "label": label, "class_name": cls, "source": name})
    pd.DataFrame(rows).to_csv(root / "manifests" / f"{name}.csv", index=False)


def run(args, cwd):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-m", *args], cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.mark.slow
def test_pipeline(tmp_path):
    proc = tmp_path / "processed"
    (proc / "manifests").mkdir(parents=True)
    for name, n, seg in [("pv_train", 8, False), ("pv_seg_train", 8, True), ("pv_val", 3, False), ("pv_test", 4, False),
                         ("pd_train", 6, False), ("pd_val", 3, False), ("pd_test", 4, False)]:
        make_split(proc, name, n, seg)
    (proc / "classes.json").write_text(json.dumps({"classes": CLASSES}))

    common = [f"paths.processed_dir={proc}", f"output_dir={tmp_path / 'runs'}", "model=efficientnet_b0",
              "train.pretrained=false", "train.epochs=2", "train.warmup_epochs=1", "train.batch_size=8",
              "data.num_workers=0", "logging.backend=none", "eval.batch_size=8", "eval.bootstrap=50",
              "composite.canvas=64"]
    run(["src.train", "--config", str(ROOT / "configs/experiments/e2_composite.yaml"), *common], tmp_path)
    run_e2 = tmp_path / "runs/e2/efficientnet_b0/seed0"
    m = json.loads((run_e2 / "metrics.json").read_text())
    assert {"pd_test", "pv_test", "gap"} <= m.keys() and m["n_classes"] == 3
    assert m["selected_on"] == "pv_val" and (run_e2 / "metrics_pd_val_selected.json").exists()
    assert (run_e2 / "preds_primary_pd_test.npz").exists()

    # a second run into a finished run directory must fail loudly
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-m", "src.train", "--config", str(ROOT / "configs/experiments/e2_composite.yaml"),
                        *common], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "already holds a finished run" in r.stderr

    run(["src.train", "--config", str(ROOT / "configs/experiments/e5_fewshot.yaml"), *common,
         "data.fewshot_k=3", "train.init_from=" + str(tmp_path / "runs/e2/efficientnet_b0/seed0/best.pt")], tmp_path)
    run(["src.train", "--config", str(ROOT / "configs/experiments/e0_baseline.yaml"), *common], tmp_path)
    assert (tmp_path / "runs/e5_k3/efficientnet_b0/seed0/metrics.json").exists()

    out = run(["src.report", "--runs", str(tmp_path / "runs")], tmp_path)
    assert "e5_k3" in out and (tmp_path / "reports/figures/pd_acc.png").exists()
    assert "Paired comparisons" in out
