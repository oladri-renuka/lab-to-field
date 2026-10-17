import numpy as np
import pandas as pd
import pytest

from src.data.leaf import grouped_val_split, leaf_id
from src.report import holm, mcnemar_exact, paired_bootstrap


def test_holm_matches_hand_computation():
    # p sorted: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 (monotone)
    assert np.allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])


def test_paired_bootstrap_detects_clear_difference_and_not_noise():
    rng = np.random.default_rng(0)
    a = (rng.random(236) < 0.6).astype(float)
    b = a.copy()
    b[:40] = 0  # B strictly worse on 40 images A got right or wrong
    diff, lo, hi, p = paired_bootstrap(a, b)
    assert diff > 0 and lo > 0 and p < 0.01
    _, _, _, p_same = paired_bootstrap(a, a)
    assert p_same == 1.0


def test_mcnemar_symmetric_null():
    ca = np.array([1, 0, 1, 0], bool)
    assert mcnemar_exact(ca, ~ca) == 1.0


def test_grouped_val_split_keeps_leaves_together():
    df = pd.DataFrame({"class_name": ["a"] * 40 + ["b"] * 40, "path": [f"p{i}" for i in range(80)]})
    groups = [f"leaf{i // 4}" for i in range(80)]  # 4 photos per leaf
    tr, va = grouped_val_split(df, groups, 0.1, 0)
    g = dict(zip(df.path, groups))
    assert set(map(g.get, tr.path)).isdisjoint(set(map(g.get, va.path)))
    assert len(va) > 0 and len(tr) + len(va) == len(df)


def test_leaf_id_strips_copy_suffix():
    a = leaf_id("raw/color/Corn___x/uuid1___RS_NLB 4165.JPG", {})
    b = leaf_id("raw/color/Corn___x/uuid2___RS_NLB 4165 copy 2.jpg", {})
    assert a == b


def test_cyclegan_source_filtered_to_current_train(tmp_path):
    from omegaconf import OmegaConf

    from src.data.datasets import source_frame

    proc, gen = tmp_path / "processed", tmp_path / "generated"
    (proc / "manifests").mkdir(parents=True)
    (gen / "cyclegan").mkdir(parents=True)
    pd.DataFrame({"path": ["x/raw/color/A/img1.JPG"], "label": [0], "class_name": ["A"]}).to_csv(
        proc / "manifests" / "pv_train.csv", index=False)
    pd.DataFrame({"path": ["g/A/img1.jpg", "g/A/img2.jpg"], "label": [0, 0], "class_name": ["A", "A"]}).to_csv(
        gen / "cyclegan" / "manifest.csv", index=False)
    cfg = OmegaConf.create({"paths": {"processed_dir": str(proc), "generated_dir": str(gen)}, "data": {}})
    assert source_frame(cfg, "cyclegan", 0).path.tolist() == ["g/A/img1.jpg"]  # img2's source is now in lab val


@pytest.mark.parametrize("exp", ["e0_baseline", "e1_strong_aug", "e2_composite", "e3_cyclegan", "e4_cvae"])
def test_lab_only_rows_select_on_lab_val(exp):
    from pathlib import Path

    from src.utils import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / f"configs/experiments/{exp}.yaml")
    assert cfg.train.select_on == "pv_val"


def test_cross_model_pairs_are_compared_on_same_images(tmp_path, monkeypatch):
    from src import report

    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    rng = np.random.default_rng(0)
    paths = np.array([f"img{i}.jpg" for i in range(50)])
    labels = rng.integers(0, 3, 50)
    rows = []
    for model, acc in (("vit_s16", 0.9), ("resnet50", 0.3)):
        run = tmp_path / model
        run.mkdir()
        probs = np.eye(3)[np.where(rng.random(50) < acc, labels, (labels + 1) % 3)]
        np.savez(run / "preds_primary_pd_test.npz", probs=probs, labels=labels, paths=paths)
        rows.append({"exp": "e0", "model": model, "run": str(run)})
    out = report.paired_tests(pd.DataFrame(rows))
    assert "across models" in out and "vit_s16" in out
