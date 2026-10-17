"""PlantVillage leaf identity (several photos of one physical leaf must stay on one side of any split)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_leaf_map(hf_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(hf_id, "leaf_grouping/leaf-map.json", repo_type="dataset")).read_text())


def leaf_id(rel: str, leaf_map: dict) -> str:
    """Replicates the HF repo's loader (plant_village.py) leaf-id logic. Unmapped images get `fallback_<photo id>`,
    with ' copy' suffixes stripped so duplicated photos share an id."""
    _, _, cls, name = rel.split("/")[:4]
    ident = name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    ident = ident.strip()
    sugg = leaf_map.get(ident.lower().strip())
    if not sugg:
        return f"fallback_{ident}"
    if len(sugg) == 1:
        return sugg[0]
    return next((s for s in sugg if cls in s), f"fallback_{ident}")


def leaf_groups(rels: list[str], leaf_map: dict) -> list[str]:
    """Group key per image: class + leaf id (fallback ids are only unique within a class)."""
    return [f"{r.split('/')[2]}|{leaf_id(r, leaf_map)}" for r in rels]


def grouped_val_split(df: pd.DataFrame, groups: list[str], frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (by class_name) split where no leaf group is on both sides. Returns (train, val)."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(df, df["class_name"], groups))
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
