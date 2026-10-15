"""Train/eval transforms. Mean/std come from the timm model's pretrained config."""
from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(kind: str, img_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> T.Compose:
    crop = [T.RandomResizedCrop(img_size, scale=(0.6, 1.0)), T.RandomHorizontalFlip()]
    if kind == "baseline":
        ops = crop + [T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.RandomRotation(20)]
    elif kind == "strong":
        ops = crop + [T.RandAugment(num_ops=2, magnitude=9)]  # MixUp/CutMix is applied on the batch
    else:
        raise ValueError(f"unknown aug kind: {kind}")
    return T.Compose(ops + [T.ToTensor(), T.Normalize(mean, std)])


def eval_transform(img_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> T.Compose:
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean, std)])
