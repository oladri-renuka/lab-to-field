"""Seeding, config loading, run directories and experiment logging."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive python/numpy seeds from the torch worker seed."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def _load_with_bases(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    base = cfg.pop("_base_", None)
    if base is None:
        return cfg
    return OmegaConf.merge(_load_with_bases((path.parent / base).resolve()), cfg)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> DictConfig:
    """Load a YAML config, following `_base_` inheritance, then apply `key=value` CLI overrides."""
    cfg = _load_with_bases(Path(path).resolve())
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def run_dir(cfg: DictConfig) -> Path:
    return Path(cfg.output_dir) / str(cfg.exp_id) / str(cfg.model) / f"seed{cfg.seed}"


def load_classes(processed_dir: str | Path) -> list[str]:
    """Label space written by `src.data.prepare` (shared PlantVillage/PlantDoc classes)."""
    path = Path(processed_dir) / "classes.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m src.data.prepare` first")
    return json.loads(path.read_text())["classes"]


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


class RunLogger:
    """Thin wrapper over MLflow / W&B so training code is backend-agnostic. No-op off rank 0."""

    def __init__(self, cfg: DictConfig, name: str, enabled: bool = True):
        self.backend = cfg.logging.backend if enabled else "none"
        params = OmegaConf.to_container(cfg, resolve=True)
        if self.backend == "mlflow":
            import mlflow

            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
            mlflow.set_experiment(cfg.logging.project)
            mlflow.start_run(run_name=name)
            flat = {k: str(v)[:250] for k, v in _flatten(params).items()}
            mlflow.log_params(flat)
            self._mlflow = mlflow
        elif self.backend == "wandb":
            import wandb

            self._run = wandb.init(project=cfg.logging.project, name=name, config=params)

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.backend == "mlflow":
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
        elif self.backend == "wandb":
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self.backend == "mlflow":
            self._mlflow.end_run()
        elif self.backend == "wandb":
            self._run.finish()


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
