"""Build all manifests from the raw datasets. Run once: `python -m src.data.prepare`.

1. PlantVillage (HF, color + segmented): keep the HF splits (leaf-grouped), keep shared classes only.
2. PlantDoc: map folders through configs/class_map.csv, drop unreadable files.
3. Perceptual-hash dedup: drop PlantDoc-train images within Hamming <= d of any PlantDoc-test image.
4. Carve a stratified 20% PlantDoc-val from PlantDoc-train (fixed split seed).
5. Write classes.json (label space), manifests/*.csv and reports/data_summary.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.class_map import load_class_map, mapped_plantvillage_classes, normalize, plantdoc_lookup
from src.data.datasets import IMG_EXTS
from src.data.dedup import cross_split_duplicates, phash_bits
from src.data.splits import stratified_val_split
from src.utils import load_config, save_json


def export_plantvillage(cfg, config_name: str, wanted: dict[str, str], out_dir: Path) -> dict[str, pd.DataFrame]:
    """Write raw image bytes for wanted classes to disk; return {split: frame with class_name}."""
    from datasets import Image as HFImage, load_dataset

    ds = load_dataset(cfg.data.hf_id, config_name)
    print(f"[plantvillage/{config_name}] splits: { {k: len(v) for k, v in ds.items()} }")
    if "test" not in ds:
        raise RuntimeError("PlantVillage HF dataset has no test split; refusing to make an ungrouped one")
    frames = {}
    for split, d in ds.items():
        split_name = {"validation": "val"}.get(split, split)
        feat = d.features[cfg.data.hf_label_col]
        names = feat.names if hasattr(feat, "names") else None
        d = d.cast_column(cfg.data.hf_image_col, HFImage(decode=False))
        rows, seen = [], set()
        for i, ex in enumerate(d):
            raw = ex[cfg.data.hf_label_col]
            name = names[raw] if names is not None else str(raw)
            seen.add(name)
            pv_class = wanted.get(normalize(name))
            if pv_class is None:
                continue
            img = ex[cfg.data.hf_image_col]
            suffix = Path(img.get("path") or "x.jpg").suffix or ".jpg"
            path = out_dir / config_name / split_name / normalize(pv_class) / f"{i:06d}{suffix}"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(img["bytes"])
            rows.append({"path": str(path), "class_name": pv_class, "source": f"pv_{config_name}"})
        missing = set(wanted) - {normalize(n) for n in seen}
        if missing:
            raise RuntimeError(f"class map targets not found in HF labels: {sorted(missing)}")
        frames[split_name] = pd.DataFrame(rows)
    return frames


def read_plantdoc(root: Path, lookup: dict[str, str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames, unmapped, bad = {}, set(), []
    for split in ("train", "test"):
        rows = []
        for folder in sorted(p for p in (root / split).iterdir() if p.is_dir()):
            pv_class = lookup.get(normalize(folder.name))
            if pv_class is None:
                unmapped.add(folder.name)
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                try:
                    with Image.open(f) as im:
                        im.convert("RGB").load()
                except Exception:
                    bad.append(str(f))
                    continue
                rows.append({"path": str(f), "class_name": pv_class, "source": "plantdoc", "plantdoc_class": folder.name})
        frames[split] = pd.DataFrame(rows)
    if unmapped:
        print(f"[plantdoc] folders not in class map / excluded: {sorted(unmapped)}")
    if bad:
        print(f"[plantdoc] skipped {len(bad)} unreadable files")
    return frames, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    processed = Path(cfg.paths.processed_dir)
    manifests = processed / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    cmap = load_class_map(cfg.paths.class_map)
    targets = mapped_plantvillage_classes(cmap)
    wanted = {normalize(c): c for c in targets}

    pd_frames, _ = read_plantdoc(Path(cfg.paths.plantdoc_dir), plantdoc_lookup(cmap))
    pv_color = export_plantvillage(cfg, "color", wanted, processed / "plantvillage")
    pv_seg = export_plantvillage(cfg, "segmented", wanted, processed / "plantvillage")

    # Label space: mapped classes present in PlantVillage train AND PlantDoc test.
    in_pv = set(pv_color["train"]["class_name"])
    in_pd_test = set(pd_frames["test"]["class_name"])
    classes = sorted(c for c in targets if c in in_pv and c in in_pd_test)
    dropped = sorted(set(targets) - set(classes))
    if dropped:
        print(f"[classes] dropped (absent from PV train or PlantDoc test): {dropped}")
    idx = {c: i for i, c in enumerate(classes)}

    def finalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["class_name"].isin(idx)].copy()
        df["label"] = df["class_name"].map(idx).astype(int)
        return df.reset_index(drop=True)

    pd_train, pd_test = finalize(pd_frames["train"]), finalize(pd_frames["test"])

    # Near-duplicate removal: test is held fixed, offending train images are dropped.
    dups = cross_split_duplicates(phash_bits(pd_train["path"].tolist()), phash_bits(pd_test["path"].tolist()),
                                  int(cfg.data.dedup_hamming))
    dup_report = pd.DataFrame(
        [{"train_path": pd_train.path[i], "test_path": pd_test.path[j], "hamming": d} for i, j, d in dups],
        columns=["train_path", "test_path", "hamming"],
    )
    Path("reports").mkdir(exist_ok=True)
    dup_report.to_csv("reports/dedup_plantdoc.csv", index=False)
    pd_train = pd_train.drop(index=[i for i, _, _ in dups]).reset_index(drop=True)
    print(f"[dedup] removed {len(dups)} PlantDoc-train images near-duplicate to test (hamming <= {cfg.data.dedup_hamming})")

    pd_train, pd_val = stratified_val_split(pd_train, float(cfg.data.val_frac), int(cfg.data.split_seed))

    out = {
        "pv_train": finalize(pv_color["train"]),
        "pv_test": finalize(pv_color["test"]),
        "pv_seg_train": finalize(pv_seg["train"]),
        "pd_train": pd_train,
        "pd_val": pd_val,
        "pd_test": pd_test,
    }
    if "val" in pv_color:
        out["pv_val"] = finalize(pv_color["val"])
    for name, df in out.items():
        df.to_csv(manifests / f"{name}.csv", index=False)

    save_json({"classes": classes, "dropped": dropped}, processed / "classes.json")
    counts = pd.DataFrame({name: df["class_name"].value_counts() for name, df in out.items()}).reindex(classes).fillna(0).astype(int)
    summary = (
        f"# Data summary\n\nShared classes (label space): **{len(classes)}**\n\n"
        f"Near-duplicates removed from PlantDoc train: **{len(dups)}**\n\n"
        f"Dropped mapped classes: {dropped or 'none'}\n\n" + counts.to_markdown() + "\n"
    )
    Path("reports/data_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
