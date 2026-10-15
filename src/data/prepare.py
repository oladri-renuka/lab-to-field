"""Build all manifests from the raw datasets. Run once: `python -m src.data.prepare`.

1. PlantVillage (HF repo data.zip): official leaf-grouped color split lists, shared classes only.
   Compositing leaves (segmented) are taken only for color-*train* images, so no test leaf leaks into E2.
2. PlantDoc: map folders through configs/class_map.csv, drop files that fail to decode.
3. Contamination: drop every PlantDoc image (any split) within pHash distance <= d of ANY PlantVillage image.
4. Near-duplicates: drop PlantDoc-train images within distance <= d of a PlantDoc-test image.
5. Carve a stratified 20% PlantDoc-val from PlantDoc-train (fixed split seed).
6. Write classes.json, manifests/*.csv, hashes.npz (for src.data.audit) and reports/data_summary.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.class_map import load_class_map, mapped_plantvillage_classes, normalize, plantdoc_lookup
from src.data.datasets import IMG_EXTS
from src.data.dedup import matches, phash
from src.data.io import load_image
from src.data.splits import stratified_val_split
from src.utils import load_config, save_json

SEG_SUFFIX = "_final_masked"


def fetch_plantvillage(cfg, configs=("color", "segmented")) -> Path:
    """Download data.zip from the HF repo and extract only the needed configs.

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


def split_list(cfg, config_name: str, split: str) -> list[str]:
    from huggingface_hub import hf_hub_download

    lst = Path(hf_hub_download(cfg.data.hf_id, f"splits/{config_name}_{split}.txt", repo_type="dataset"))
    return [ln.strip() for ln in lst.read_text().splitlines() if ln.strip()]


def leaf_key(rel: str) -> tuple[str, str]:
    """(class, stem) shared by a color image and its segmented version: raw/<config>/<class>/<file>."""
    parts = rel.split("/")
    return normalize(parts[2]), Path(parts[3]).stem.lower().replace(SEG_SUFFIX, "")


def plantvillage_frames(cfg, base: Path, wanted: dict[str, str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Shared-class color train/test, segmented counterparts of color-train, and all color images (for hashing)."""
    color = {s: split_list(cfg, "color", s) for s in ("train", "test")}
    seg = {s: split_list(cfg, "segmented", s) for s in ("train", "test")}

    all_rows, missing = [], 0
    for split, lines in color.items():
        for rel in lines:
            if not (base / rel).exists():
                missing += 1
                continue
            cls = rel.split("/")[2]
            all_rows.append({"path": str(base / rel), "rel": rel, "split": split, "pv_label": cls,
                             "class_name": wanted.get(normalize(cls))})
    all_color = pd.DataFrame(all_rows)
    if missing > 0.01 * len(all_color):
        raise RuntimeError(f"{missing} listed color files missing under {base}; extraction incomplete?")
    absent = set(wanted) - {normalize(c) for c in all_color.pv_label}
    if absent:
        raise RuntimeError(f"class map targets not found in PlantVillage labels: {sorted(absent)}")

    seg_index = {leaf_key(rel): rel for lines in seg.values() for rel in lines}
    color_split = {leaf_key(r): s for s, lines in color.items() for r in lines}
    shared = all_color[all_color.class_name.notna()]
    out = {s: shared[shared.split == s][["path", "class_name"]].assign(source="pv_color") for s in ("train", "test")}

    seg_rows = []
    for r in shared[shared.split == "train"].itertuples():
        rel = seg_index.get(leaf_key(r.rel))
        if rel and (base / rel).exists():
            seg_rows.append({"path": str(base / rel), "class_name": r.class_name, "source": "pv_segmented",
                             "color_path": r.path})
    out["seg_train"] = pd.DataFrame(seg_rows)

    seg_stats = {
        "color_train_shared": len(out["train"]),
        "color_train_with_segmented": len(seg_rows),
        "segmented_total": sum(len(v) for v in seg.values()),
        "segmented_without_color": sum(k not in color_split for k in seg_index),
        # leak the naive pipeline would have: segmented-train items whose color version is in color-test
        "segmented_train_listed_but_color_test": sum(color_split.get(leaf_key(r)) == "test" for r in seg["train"]),
    }
    print(f"[plantvillage] shared-class color train={len(out['train'])} test={len(out['test'])}; "
          f"compositing leaves={len(seg_rows)}; segmented-train items with color in test (excluded)="
          f"{seg_stats['segmented_train_listed_but_color_test']}")
    return out, all_color, seg_stats


def read_plantdoc(root: Path, lookup: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    rows, unmapped, bad = [], set(), []
    for split in ("train", "test"):
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
                        im.verify()
                    load_image(f).load()
                except Exception:
                    bad.append(str(f))
                    continue
                rows.append({"path": str(f), "class_name": pv_class, "source": "plantdoc",
                             "plantdoc_class": folder.name, "split": split})
    if unmapped:
        print(f"[plantdoc] folders not in class map / excluded: {sorted(unmapped)}")
    print(f"[plantdoc] {len(rows)} readable images, {len(bad)} unreadable skipped")
    return pd.DataFrame(rows), bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    d = int(cfg.data.dedup_hamming)

    processed = Path(cfg.paths.processed_dir)
    manifests = processed / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    cmap = load_class_map(cfg.paths.class_map)
    targets = mapped_plantvillage_classes(cmap)
    wanted = {normalize(c): c for c in targets}

    plantdoc, bad = read_plantdoc(Path(cfg.paths.plantdoc_dir), plantdoc_lookup(cmap))
    pd.Series(bad, name="path", dtype=str).to_csv(reports / "plantdoc_unreadable.csv", index=False)
    pv, all_color, seg_stats = plantvillage_frames(cfg, fetch_plantvillage(cfg), wanted)

    print(f"[hash] pHash of {len(all_color)} PlantVillage color + {len(plantdoc)} PlantDoc images "
          f"({args.workers} workers)")
    pv_hash = phash(all_color.path.tolist(), args.workers)
    pd_hash = phash(plantdoc.path.tolist(), args.workers)
    np.savez_compressed(processed / "hashes.npz", pv_path=all_color.path.values.astype(str),
                        pv_split=all_color.split.values.astype(str), pv_label=all_color.pv_label.values.astype(str),
                        pv_hash=pv_hash, pd_path=plantdoc.path.values.astype(str),
                        pd_split=plantdoc.split.values.astype(str), pd_hash=pd_hash)

    # 3. PlantVillage photos re-posted inside PlantDoc (any PlantDoc split, any PlantVillage class)
    contam = matches(pd_hash, pv_hash, d)
    contam_df = pd.DataFrame([{"plantdoc_path": plantdoc.path.iloc[i], "plantdoc_split": plantdoc.split.iloc[i],
                               "plantvillage_path": all_color.path.iloc[j], "hamming": dist} for i, j, dist in contam],
                             columns=["plantdoc_path", "plantdoc_split", "plantvillage_path", "hamming"])
    contam_df.to_csv(reports / "contamination_plantvillage_in_plantdoc.csv", index=False)
    keep = np.ones(len(plantdoc), bool)
    keep[[i for i, _, _ in contam]] = False
    print(f"[contamination] PlantDoc images matching a PlantVillage image (hamming <= {d}), removed: "
          f"{contam_df.plantdoc_split.value_counts().to_dict() or 0}")

    # 4. PlantDoc train near-duplicates of PlantDoc test (test kept fixed)
    is_train = (plantdoc.split == "train").values
    tr, te = np.flatnonzero(keep & is_train), np.flatnonzero(keep & ~is_train)
    dups = [(tr[i], te[j], dist) for i, j, dist in matches(pd_hash[tr], pd_hash[te], d)]
    pd.DataFrame([{"train_path": plantdoc.path.iloc[i], "test_path": plantdoc.path.iloc[j], "hamming": dist}
                  for i, j, dist in dups],
                 columns=["train_path", "test_path", "hamming"]).to_csv(reports / "dedup_plantdoc.csv", index=False)
    keep[[i for i, _, _ in dups]] = False
    print(f"[dedup] removed {len(dups)} PlantDoc-train images near-duplicate to test (hamming <= {d})")
    plantdoc = plantdoc[keep]

    # Label space: mapped classes present in PlantVillage train AND (cleaned) PlantDoc test.
    in_pv = set(pv["train"].class_name)
    in_pd_test = set(plantdoc[plantdoc.split == "test"].class_name)
    classes = sorted(c for c in targets if c in in_pv and c in in_pd_test)
    dropped = sorted(set(targets) - set(classes))
    if dropped:
        print(f"[classes] dropped (absent from PV train or PlantDoc test): {dropped}")
    idx = {c: i for i, c in enumerate(classes)}

    def finalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["class_name"].isin(idx)].drop(columns=["split"], errors="ignore").copy()
        df["label"] = df["class_name"].map(idx).astype(int)
        return df.reset_index(drop=True)

    pd_train, pd_val = stratified_val_split(finalize(plantdoc[plantdoc.split == "train"]),
                                            float(cfg.data.val_frac), int(cfg.data.split_seed))
    out = {
        "pv_train": finalize(pv["train"]),
        "pv_test": finalize(pv["test"]),
        "pv_seg_train": finalize(pv["seg_train"]),
        "pd_train": pd_train,
        "pd_val": pd_val,
        "pd_test": finalize(plantdoc[plantdoc.split == "test"]),
    }
    for name, df in out.items():
        df.to_csv(manifests / f"{name}.csv", index=False)

    save_json({"classes": classes, "dropped": dropped}, processed / "classes.json")
    save_json({**seg_stats, "contaminated_removed": len(contam), "near_duplicates_removed": len(dups),
               "unreadable": len(bad)}, processed / "prepare_stats.json")
    counts = (pd.DataFrame({name: df["class_name"].value_counts() for name, df in out.items()})
              .reindex(classes).fillna(0).astype(int))
    summary = (
        f"# Data summary\n\nShared classes (label space): **{len(classes)}**\n\n"
        f"- PlantDoc images matching a PlantVillage image, removed: **{len(contam)}** "
        f"(`reports/contamination_plantvillage_in_plantdoc.csv`)\n"
        f"- PlantDoc-train near-duplicates of test, removed: **{len(dups)}** (`reports/dedup_plantdoc.csv`)\n"
        f"- Unreadable PlantDoc files skipped: **{len(bad)}**\n"
        f"- Compositing leaves, color-train only: **{seg_stats['color_train_with_segmented']}** "
        f"(segmented-train items whose color version is in test, excluded: "
        f"{seg_stats['segmented_train_listed_but_color_test']})\n"
        f"- Dropped mapped classes: {dropped or 'none'}\n\n" + counts.to_markdown() + "\n"
    )
    (reports / "data_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
