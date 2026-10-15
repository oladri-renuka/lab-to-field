from pathlib import Path

from src.utils import load_config, run_dir

ROOT = Path(__file__).resolve().parents[1]

EXP = ROOT / "configs/experiments"


def test_inheritance_and_overrides():
    cfg = load_config(EXP / "e1_strong_aug.yaml", ["model=vit_s16", "seed=2"])
    assert cfg.aug.kind == "strong" and cfg.aug.mixup and cfg.train.epochs == 30
    assert str(run_dir(cfg)) == "runs/e1/vit_s16/seed2"


def test_fewshot_interpolation():
    cfg = load_config(EXP / "e5_fewshot.yaml", ["data.fewshot_k=25", "model=resnet50", "seed=1"])
    assert cfg.exp_id == "e5_k25"
    assert cfg.train.init_from == "runs/e0/resnet50/seed1/best.pt"


def test_e6_inherits_e5_and_points_at_selected_experiment():
    cfg = load_config(EXP / "e6_best_combo.yaml", ["init_experiment=e3"])
    assert cfg.exp_id == "e6_from_e3" and cfg.data.fewshot_k == 10
    assert cfg.train.init_from.startswith("runs/e3/")
