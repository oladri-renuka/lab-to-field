"""Build all manifests from the raw datasets. Run once: `python -m src.data.prepare`.

1. PlantVillage (HF repo data.zip, color + segmented): official leaf-grouped split lists, shared classes only.
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


def fetch_plantvillage(cfg, configs=("color", "segmented")) -> Path:
    """Download data.zip + official split lists from the HF repo and extract only the needed configs.

    The repo ships a loading script that `datasets` >= 4 no longer runs, so we read its files directly.
    Returns the directory that contains `raw/<config>/<class>/<file>`.
    """
    import zipfile

    from huggingface_hub import hf_hub_download

    root = Path(cfg.paths.raw_dir) / "plantvillage"
    done = root / ".extracted"
    if not done.exists():
        zpath = hf_hub_download(cfg.data.hf_id, "data.zip", repo_type="dataset")
        prefixes = tuple(f"raw/{c}/" for c in configs)
        with zipfile.ZipFile(zpath) as z:
            members = [m for m in z.namelist() if not m.endswith("/") and any(p in m for p in prefixes)]
            print(f"[plantvillage] extracting {len(members)} files from data.zip")
            z.extractall(root, members)
        done.touch()
    # zip may or may not have a top-level folder above raw/
    for base in (root, *[d for d in root.iterdir() if d.is_dir()]):
        if (base / "raw" / "color").is_dir():
            return base
    raise RuntimeError(f"could not find raw/color under {root}")


def export_plantvillage(cfg, config_name: str, wanted: dict[str, str], base: Path) -> dict[str, pd.DataFrame]:
    """Manifests from the official leaf-grouped split lists (`splits/<config>_<split>.txt`)."""
    from huggingface_hub import hf_hub_download

    frames = {}
    for split in ("train", "test"):
        lst = Path(hf_hub_download(cfg.data.hf_id, f"splits/{config_name}_{split}.txt", repo_type="dataset"))
        lines = [ln.strip() for ln in lst.read_text().splitlines() if ln.strip()]
        rows, seen, missing = [], set(), 0
        for rel in lines:
            cls = rel.split("/")[2]  # raw/<config>/<class>/<file>
            seen.add(cls)
            pv_class = wanted.get(normalize(cls))
            if pv_class is None:
                continue
            path = base / rel
            if not path.exists():
                missing += 1
                continue
            rows.append({"path": str(path), "class_name": pv_class, "source": f"pv_{config_name}"})
        print(f"[plantvillage/{config_name}/{split}] listed={len(lines)} kept={len(rows)} missing_files={missing}")
        if missing > 0.01 * max(1, len(rows)):
            raise RuntimeError(f"{missing} listed files not found under {base}; extraction incomplete?")
        absent = set(wanted) - {normalize(c) for c in seen}
        if absent:
            raise RuntimeError(f"class map targets not found in PlantVillage labels: {sorted(absent)}")
        frames[split] = pd.DataFrame(rows)
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
    pv_base = fetch_plantvillage(cfg)
    pv_color = export_plantvillage(cfg, "color", wanted, pv_base)
    pv_seg = export_plantvillage(cfg, "segmented", wanted, pv_base)

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
