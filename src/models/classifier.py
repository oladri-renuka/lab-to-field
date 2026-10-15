"""timm classifiers: ResNet-50, EfficientNet-B0, ViT-S/16."""
from __future__ import annotations

import timm
import torch.nn as nn


def build_classifier(timm_name: str, num_classes: int, pretrained: bool = True, grad_checkpointing: bool = False) -> nn.Module:
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    if grad_checkpointing:
        model.set_grad_checkpointing(True)
    return model


def norm_stats(model: nn.Module) -> tuple[tuple, tuple]:
    """Normalization the backbone was pretrained with (ImageNet for the CNNs, 0.5/0.5 for AugReg ViT)."""
    cfg = timm.data.resolve_data_config({}, model=model)
    return tuple(cfg["mean"]), tuple(cfg["std"])


def is_vit(model: nn.Module) -> bool:
    return hasattr(model, "blocks") and hasattr(model, "patch_embed")
