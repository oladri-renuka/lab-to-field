"""Build all manifests from the raw datasets. Run once: `python -m src.data.prepare`.

1. PlantVillage (HF repo data.zip): official leaf-grouped color split lists, shared classes only; a 10% leaf-grouped
   lab-only val split is carved from train (checkpoint selection for experiments that use no field labels).
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
from src.data.leaf import grouped_val_split, leaf_groups, load_leaf_map
from src.data.splits import stratified_val_split
from src.utils import Steps, load_config, progress, save_json

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
            for m in progress(members, "extract data.zip", unit="file"):
                z.extract(m, root)
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
    return normalize(parts[2]), Path(parts[3]).stem.lower().replace(SEG_SUFFIX.lower(), "")


def _short_key(rel: str) -> tuple[str, str]:
    cls, stem = leaf_key(rel)
    return cls, stem.split("___")[-1].strip()


class PairIndex:
    """Look up the counterpart of an image in the other config (color <-> segmented).

    Exact (class, stem) match first; otherwise the part after the last '___', used only when unique in the class.
    Needed because e.g. color `Corn___Common_rust_/RS_Rust 1563.JPG` is `<uuid>___RS_Rust 1563_final_masked.jpg`.
    """

    def __init__(self, rels):
        rels = list(rels)
        self.exact = {leaf_key(r): r for r in rels}
        groups: dict[tuple[str, str], list[str]] = {}
        for r in rels:
            groups.setdefault(_short_key(r), []).append(r)
        self.short = {k: v[0] for k, v in groups.items() if len(v) == 1}

    def get(self, rel: str) -> str | None:
        return self.exact.get(leaf_key(rel)) or self.short.get(_short_key(rel))


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

    seg_index = PairIndex(r for lines in seg.values() for r in lines)
    color_index = PairIndex(r for lines in color.values() for r in lines)
    color_split = {r: s for s, lines in color.items() for r in lines}
    shared = all_color[all_color.class_name.notna()]
    out = {s: shared[shared.split == s][["path", "class_name"]].assign(source="pv_color") for s in ("train", "test")}

    # Lab-only validation set: leaf-grouped, stratified carve-out of PlantVillage train. Lets lab-only experiments
    # select checkpoints without touching any field image.
    shared_train = shared[shared.split == "train"].reset_index(drop=True)
    groups = leaf_groups(shared_train.rel.tolist(), load_leaf_map(cfg.data.hf_id))
    tr, va = grouped_val_split(shared_train, groups, float(cfg.data.pv_val_frac), int(cfg.data.split_seed))
    out["train"] = tr[["path", "class_name"]].assign(source="pv_color")
    out["val"] = va[["path", "class_name"]].assign(source="pv_color")

    seg_rows = []
    for r in tr.itertuples():  # compositing leaves only from the (reduced) train split
        rel = seg_index.get(r.rel)
        if rel and (base / rel).exists():
            seg_rows.append({"path": str(base / rel), "class_name": r.class_name, "source": "pv_segmented",
                             "color_path": r.path})
    out["seg_train"] = pd.DataFrame(seg_rows)

    seg_stats = {
        "color_train_shared": len(out["train"]),
        "color_train_with_segmented": len(seg_rows),
        "segmented_total": sum(len(v) for v in seg.values()),
        "segmented_without_color": sum(color_index.get(r) is None for lines in seg.values() for r in lines),
        # leak the naive pipeline would have: segmented-train items whose color version is in color-test
        "segmented_train_listed_but_color_test": sum(color_split.get(color_index.get(r)) == "test" for r in seg["train"]),
    }
    print(f"[plantvillage] shared-class color train={len(out['train'])} test={len(out['test'])}; "
          f"compositing leaves={len(seg_rows)}; segmented-train items with color in test (excluded)="
          f"{seg_stats['segmented_train_listed_but_color_test']}")
    return out, all_color, seg_stats


def read_plantdoc(root: Path, lookup: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    rows, unmapped, bad = [], set(), []
    for split in ("train", "test"):
        for folder in progress(sorted(p for p in (root / split).iterdir() if p.is_dir()), f"check PlantDoc {split}",
                               unit="class"):
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
    step = Steps("prepare", 8)

    processed = Path(cfg.paths.processed_dir)
    manifests = processed / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    cmap = load_class_map(cfg.paths.class_map)
    targets = mapped_plantvillage_classes(cmap)
    wanted = {normalize(c): c for c in targets}

    step("reading PlantDoc + mapping folders to shared classes")
    plantdoc, bad = read_plantdoc(Path(cfg.paths.plantdoc_dir), plantdoc_lookup(cmap))
    pd.Series(bad, name="path", dtype=str).to_csv(reports / "plantdoc_unreadable.csv", index=False)
    step("PlantVillage: download/extract data.zip (skipped if done), official leaf-grouped splits")
    pv, all_color, seg_stats = plantvillage_frames(cfg, fetch_plantvillage(cfg), wanted)

    step(f"perceptual hashing {len(all_color)} PlantVillage + {len(plantdoc)} PlantDoc images ({args.workers} workers)")
    pv_hash = phash(all_color.path.tolist(), args.workers, "hash PlantVillage")
    pd_hash = phash(plantdoc.path.tolist(), args.workers, "hash PlantDoc")
    np.savez_compressed(processed / "hashes.npz", pv_path=all_color.path.to_numpy(dtype=str),
                        pv_split=all_color.split.to_numpy(dtype=str), pv_label=all_color.pv_label.to_numpy(dtype=str),
                        pv_hash=pv_hash, pd_path=plantdoc.path.to_numpy(dtype=str),
                        pd_split=plantdoc.split.to_numpy(dtype=str), pd_hash=pd_hash)

    step("removing PlantVillage test images near-identical to train")
    # PlantVillage test images near-identical to a PlantVillage train image (e.g. "... copy.jpg" duplicates
    # without a leaf id): removed from test so lab accuracy is not inflated.
    is_tr = (all_color.split == "train").to_numpy()
    te_idx, tr_idx = np.flatnonzero(~is_tr), np.flatnonzero(is_tr)
    pv_leak = [(te_idx[i], tr_idx[j], dist)
               for i, j, dist in matches(pv_hash[~is_tr], pv_hash[is_tr], d)]
    pd.DataFrame([{"test_path": all_color.path.iloc[i], "train_path": all_color.path.iloc[j], "hamming": dist}
                  for i, j, dist in pv_leak],
                 columns=["test_path", "train_path", "hamming"]).to_csv(reports / "pv_test_near_dup_train.csv", index=False)
    leaked = {all_color.path.iloc[i] for i, _, _ in pv_leak}
    n_before = len(pv["test"])
    pv["test"] = pv["test"][~pv["test"].path.isin(leaked)]
    # Same rule for the new lab val split vs the reduced lab train split.
    path_idx = {pth: i for i, pth in enumerate(all_color.path)}
    va_i = np.array([path_idx[pth] for pth in pv["val"].path])
    tr_i = np.array([path_idx[pth] for pth in pv["train"].path])
    val_dups = {pv["val"].path.iloc[i] for i, _, _ in matches(pv_hash[va_i], pv_hash[tr_i], d)}
    pv["val"] = pv["val"][~pv["val"].path.isin(val_dups)]
    print(f"[pv-dedup] PlantVillage val images near-identical to a train image, removed: {len(val_dups)}")
    print(f"[pv-dedup] PlantVillage test images near-identical to a train image (hamming <= {d}): {len(pv_leak)} "
          f"(all classes); removed from shared-class test: {n_before - len(pv['test'])}")

    step("removing PlantVillage photos found inside PlantDoc")
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

    step("removing PlantDoc-train near-duplicates of PlantDoc test")
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

    step("label space, stratified PlantDoc val split")
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
        "pv_val": finalize(pv["val"]),
        "pv_test": finalize(pv["test"]),
        "pv_seg_train": finalize(pv["seg_train"]),
        "pd_train": pd_train,
        "pd_val": pd_val,
        "pd_test": finalize(plantdoc[plantdoc.split == "test"]),
    }
    step(f"writing manifests to {manifests} and reports/data_summary.md")
    for name, df in out.items():
        df.to_csv(manifests / f"{name}.csv", index=False)

    save_json({"classes": classes, "dropped": dropped}, processed / "classes.json")
    save_json({**seg_stats, "contaminated_removed": len(contam), "near_duplicates_removed": len(dups),
               "pv_test_near_dup_removed": n_before - len(pv["test"]),
               "unreadable": len(bad)}, processed / "prepare_stats.json")
    counts = (pd.DataFrame({name: df["class_name"].value_counts() for name, df in out.items()})
              .reindex(classes).fillna(0).astype(int))
    summary = (
        f"# Data summary\n\nShared classes (label space): **{len(classes)}**\n\n"
        f"- PlantDoc images matching a PlantVillage image, removed: **{len(contam)}** "
        f"(`reports/contamination_plantvillage_in_plantdoc.csv`)\n"
        f"- PlantDoc-train near-duplicates of test, removed: **{len(dups)}** (`reports/dedup_plantdoc.csv`)\n"
        f"- PlantVillage test images near-identical to a train image, removed from test: "
        f"**{n_before - len(pv['test'])}** (`reports/pv_test_near_dup_train.csv`)\n"
        f"- Unreadable PlantDoc files skipped: **{len(bad)}**\n"
        f"- Compositing leaves, color-train only: **{seg_stats['color_train_with_segmented']}** "
        f"(segmented-train items whose color version is in test, excluded: "
        f"{seg_stats['segmented_train_listed_but_color_test']})\n"
        f"- Dropped mapped classes: {dropped or 'none'}\n\n" + counts.to_markdown() + "\n"
    )
    (reports / "data_summary.md").write_text(summary)
    print(summary)
    step.done("data ready")


if __name__ == "__main__":
    main()
