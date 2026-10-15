"""PlantDoc <-> PlantVillage class mapping (configs/class_map.csv)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["plantdoc_class", "plantvillage_class", "include"]


def normalize(name: str) -> str:
    """Case/punctuation-insensitive key so HF label strings and folder names match robustly."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_class_map(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"class map missing columns: {missing}")
    if df["plantdoc_class"].duplicated().any():
        raise ValueError("duplicate plantdoc_class rows in class map")
    return df


def included(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["include"].astype(int) == 1]


def mapped_plantvillage_classes(df: pd.DataFrame) -> list[str]:
    """Sorted unique PlantVillage classes targeted by included rows (several PlantDoc folders may map to one)."""
    return sorted(included(df)["plantvillage_class"].unique())


def plantdoc_lookup(df: pd.DataFrame) -> dict[str, str]:
    """normalized PlantDoc folder name -> PlantVillage class name (included rows only)."""
    return {normalize(r.plantdoc_class): r.plantvillage_class for r in included(df).itertuples()}


def crop_of(pv_class: str) -> str:
    return pv_class.split("___")[0]
