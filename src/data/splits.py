"""Stratified validation carve-out and few-shot sampling (both deterministic)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_val_split(df: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by `label`. Classes with < 2 images stay entirely in train (cannot be stratified)."""
    counts = df["label"].value_counts()
    rare = df["label"].isin(counts[counts < 2].index)
    train, val = train_test_split(df[~rare], test_size=val_frac, stratify=df.loc[~rare, "label"], random_state=seed)
    train = pd.concat([train, df[rare]])
    return train.sort_index().reset_index(drop=True), val.sort_index().reset_index(drop=True)


def few_shot(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Up to k images per class. Uses the seed so each training seed sees a different k-shot draw."""
    parts = [g.sample(n=min(k, len(g)), random_state=seed) for _, g in df.groupby("label", sort=True)]
    return pd.concat(parts).reset_index(drop=True)
