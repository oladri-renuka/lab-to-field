"""Data audit, run after `src.data.prepare`. Writes reports/audit/ (audit.md + figures + CSVs).

    python -m src.data.audit                 # all sections
    python -m src.data.audit --only leaf files
    python -m src.data.audit --score-labels  # after filling reports/audit/label_review.csv

Sections:
  leaf    PlantVillage leaf grouping: leaf ids shared by train and test (HF leaf-map.json) + pHash near-dups
  counts  per-class PlantDoc test counts, imbalance ratios
  files   verify every file; formats, modes, EXIF rotation, tiny images, resolution/aspect plots
  stats   lab vs field pixel statistics: channel mean/std, brightness, blur (Laplacian variance)
  seg     segmented <-> color pairing and mask quality (area, fragments, alignment), worst-case grid
  labels  contact sheets for manual label review (20 PlantDoc / 10 PlantVillage per class) + CSV template
The background-only shortcut classifier is a separate experiment: `python -m src.shortcut`.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.data.composite import leaf_mask
from src.data.dedup import nearest
from src.data.io import load_image
from src.metrics import bootstrap_accuracy_ci
from src.utils import load_config

OUT = Path("reports/audit")
SECTIONS = ["leaf", "counts", "files", "stats", "seg", "labels"]


def manifests(cfg) -> dict[str, pd.DataFrame]:
    root = Path(cfg.paths.processed_dir) / "manifests"
    return {p.stem: pd.read_csv(p) for p in sorted(root.glob("*.csv"))}


# ---------------------------------------------------------------- leaf grouping
def leaf_id(rel: str, leaf_map: dict) -> str:
    """Replicates the HF repo's loader (plant_village.py) leaf-id logic."""
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


def section_leaf(cfg) -> str:
    from huggingface_hub import hf_hub_download

    from src.data.prepare import split_list

    leaf_map = json.loads(Path(hf_hub_download(cfg.data.hf_id, "leaf_grouping/leaf-map.json", repo_type="dataset")).read_text())
    ids = {s: pd.Series([leaf_id(r, leaf_map) for r in split_list(cfg, "color", s)]) for s in ("train", "test")}
    real = {s: v[~v.str.startswith("fallback_")] for s, v in ids.items()}
    shared = set(real["train"]) & set(real["test"])
    n_img_test_shared = int(real["test"].isin(shared).sum())
    fallback = {s: float(v.str.startswith("fallback_").mean()) for s, v in ids.items()}

    h = np.load(Path(cfg.paths.processed_dir) / "hashes.npz", allow_pickle=False)
    tr, te = h["pv_split"] == "train", h["pv_split"] == "test"
    _, dist = nearest(h["pv_hash"][te], h["pv_hash"][tr])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(dist, bins=np.arange(0, 34) - 0.5)
    ax.axvline(4.5, color="k", ls="--", lw=1)
    ax.set_xlabel("pHash distance, PlantVillage test image → nearest train image")
    ax.set_ylabel("images")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "pv_test_to_train_phash.png", dpi=130)
    plt.close(fig)

    ok = len(shared) == 0
    return (
        "## 1a. PlantVillage leaf grouping\n\n"
        f"- Leaf ids (HF `leaf-map.json`, loader logic) present in **both** train and test: **{len(shared)}** "
        f"({n_img_test_shared} test images). {'✅ grouping holds' if ok else '⚠️ grouping broken for these leaves'}\n"
        f"- Images with no leaf id in the map (fallback, cannot be checked by id): "
        f"train {fallback['train']:.1%}, test {fallback['test']:.1%}\n"
        f"- pHash check, PlantVillage test vs train: distance ≤ 4: **{int((dist <= 4).sum())}** / {len(dist)} test images; "
        f"≤ 10: {int((dist <= 10).sum())}. Distribution: `pv_test_to_train_phash.png`\n"
    )


# ---------------------------------------------------------------- counts
def section_counts(cfg) -> str:
    m = manifests(cfg)
    rows = []
    for name in ("pv_train", "pv_test", "pd_train", "pd_val", "pd_test"):
        vc = m[name].class_name.value_counts()
        rows.append({"split": name, "images": int(vc.sum()), "classes": len(vc), "min": int(vc.min()),
                     "max": int(vc.max()), "imbalance (max/min)": round(vc.max() / vc.min(), 1)})
    small = m["pd_test"].class_name.value_counts()
    small = small[small < 10].sort_values()
    listed = ", ".join(f"{c.replace('___', '/')}={n}" for c, n in small.items())
    return (
        "## 2. Class counts\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\n"
        f"PlantDoc-test classes with < 10 images: **{len(small)}** of {m['pd_test'].class_name.nunique()} ({listed}). "
        "Kept; per-class recall for these is reported with that caveat, and overall accuracy carries a bootstrap CI.\n"
    )


# ---------------------------------------------------------------- files
def _inspect(path: str) -> dict:
    r = {"path": path, "ok": True, "format": None, "mode": None, "w": 0, "h": 0, "exif_orientation": 1}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            r.update(format=im.format, mode=im.mode, w=im.size[0], h=im.size[1],
                     exif_orientation=int(im.getexif().get(0x0112, 1) or 1))
            im.load()
    except Exception as e:
        r.update(ok=False, error=str(e)[:200])
    return r


def section_files(cfg, workers: int) -> str:
    h = np.load(Path(cfg.paths.processed_dir) / "hashes.npz", allow_pickle=False)
    pd_root = Path(cfg.paths.plantdoc_dir)
    pd_all = sorted(str(p) for p in pd_root.rglob("*") if p.is_file() and not p.name.startswith(".") and ".git" not in p.parts)
    seg = pd.read_csv(Path(cfg.paths.processed_dir) / "manifests" / "pv_seg_train.csv").path.tolist()
    groups = {"plantdoc": pd_all, "plantvillage_color": h["pv_path"].tolist(), "plantvillage_segmented": seg}
    tables, frames = [], {}
    with ProcessPoolExecutor(workers) as ex:
        for name, paths in groups.items():
            df = pd.DataFrame(list(ex.map(_inspect, paths, chunksize=256)))
            frames[name] = df
            df.to_csv(OUT / f"files_{name}.csv", index=False)
            good = df[df.ok]
            tables.append({
                "dataset": name, "files": len(df), "unreadable": int((~df.ok).sum()),
                "formats": dict(good.format.value_counts()), "modes": dict(good["mode"].value_counts()),
                "EXIF rotated": int((good.exif_orientation != 1).sum()),
                "min side < 100px": int((good[["w", "h"]].min(1) < 100).sum()),
                "median WxH": f"{int(good.w.median())}x{int(good.h.median())}",
            })

    p = frames["plantdoc"][frames["plantdoc"].ok]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].scatter(p.w, p.h, s=3, alpha=0.4)
    axes[0].set(xscale="log", yscale="log", xlabel="width (px)", ylabel="height (px)", title="PlantDoc resolutions")
    axes[0].axvline(100, color="r", lw=0.8)
    axes[0].axhline(100, color="r", lw=0.8)
    axes[1].hist(np.log2(p.w / p.h), bins=40)
    axes[1].set(xlabel="log2(width / height)", title="PlantDoc aspect ratio")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "plantdoc_resolution.png", dpi=130)
    plt.close(fig)
    p[p[["w", "h"]].min(1) < 100].to_csv(OUT / "plantdoc_tiny.csv", index=False)

    t = pd.DataFrame(tables)
    return (
        "## 5. File sanity\n\n" + t.to_markdown(index=False) + "\n\n"
        "All loading goes through `src/data/io.load_image`: EXIF orientation applied, RGBA/LA/P-with-transparency "
        "composited on white, everything else converted to RGB, first frame of GIFs. Unreadable PlantDoc files are "
        "dropped in `prepare` (`reports/plantdoc_unreadable.csv`). Tiny images: `plantdoc_tiny.csv`; resolution plot: "
        "`plantdoc_resolution.png`.\n"
    )


# ---------------------------------------------------------------- stats
def _pixel_stats(path: str) -> dict:
    img = load_image(path).resize((256, 256), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    g = x @ np.array([0.299, 0.587, 0.114], np.float32)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    return {"r": x[..., 0].mean(), "g": x[..., 1].mean(), "b": x[..., 2].mean(),
            "r_std": x[..., 0].std(), "g_std": x[..., 1].std(), "b_std": x[..., 2].std(),
            "brightness": g.mean(), "saturation": hsv[..., 1].mean(), "blur_lapvar": float(lap.var() * 1e3)}


def section_stats(cfg, workers: int, n: int = 2000) -> str:
    m = manifests(cfg)
    sets = {"PlantVillage train": m["pv_train"], "PlantDoc train+val": pd.concat([m["pd_train"], m["pd_val"]]),
            "PlantDoc test": m["pd_test"]}
    frames = {}
    with ProcessPoolExecutor(workers) as ex:
        for name, df in sets.items():
            paths = df.sample(n=min(n, len(df)), random_state=0).path.tolist()
            frames[name] = pd.DataFrame(list(ex.map(_pixel_stats, paths, chunksize=64)))
    cols = ["r", "g", "b", "r_std", "g_std", "b_std", "brightness", "saturation", "blur_lapvar"]
    table = pd.DataFrame({k: {c: f"{v[c].mean():.3f} ± {v[c].std():.3f}" for c in cols} for k, v in frames.items()}).T
    table.insert(0, "n", [len(v) for v in frames.values()])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for name, v in frames.items():
        axes[0].hist(v.brightness, bins=40, alpha=0.5, density=True, label=name)
        axes[1].hist(v.saturation, bins=40, alpha=0.5, density=True, label=name)
        axes[2].hist(np.log10(v.blur_lapvar + 1e-3), bins=40, alpha=0.5, density=True, label=name)
    for ax, t in zip(axes, ["brightness", "saturation", "log10 Laplacian variance (sharpness)"]):
        ax.set_xlabel(t)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "domain_stats.png", dpi=130)
    plt.close(fig)
    return (
        "## 6. Lab vs field statistics (per image, resized to 256×256; mean ± std across images)\n\n"
        + table.to_markdown() + "\n\nHigher Laplacian variance = sharper. Histograms: `domain_stats.png`.\n"
    )


# ---------------------------------------------------------------- segmented masks
def _mask_quality(args) -> dict:
    from scipy import ndimage

    seg_path, color_path = args
    seg, col = load_image(seg_path), load_image(color_path)
    if seg.size != col.size:
        return {"path": seg_path, "size_mismatch": True}
    mask = np.asarray(leaf_mask(seg)) > 127
    lab, n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    big = int((sizes > 0.005 * mask.size).sum())
    s, c = (np.asarray(x, dtype=np.float32) / 255 for x in (seg, col))
    align = float(np.abs(s - c)[mask].mean()) if mask.any() else 1.0
    border = float(np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean())
    return {"path": seg_path, "color_path": color_path, "size_mismatch": False, "area": float(mask.mean()),
            "components": int(n), "large_components": big, "align_err": align, "border_touch": border}


def section_seg(cfg, workers: int, n: int = 3000) -> str:
    stats = json.loads((Path(cfg.paths.processed_dir) / "prepare_stats.json").read_text())
    seg = pd.read_csv(Path(cfg.paths.processed_dir) / "manifests" / "pv_seg_train.csv")
    sample = seg.sample(n=min(n, len(seg)), random_state=0)
    with ProcessPoolExecutor(workers) as ex:
        q = pd.DataFrame(list(ex.map(_mask_quality, zip(sample.path, sample.color_path), chunksize=64)))
    q.to_csv(OUT / "segmented_mask_quality.csv", index=False)
    ok = q[~q.size_mismatch]
    flags = {
        "size differs from color": int(q.size_mismatch.sum()),
        "mask area < 5% (leaf mostly cut away)": int((ok.area < 0.05).sum()),
        "mask area > 90% (background left in)": int((ok.area > 0.9).sum()),
        "> 1 large fragment (specks / split leaf)": int((ok.large_components > 1).sum()),
        "pixel mismatch vs color > 0.10 (not aligned)": int((ok.align_err > 0.10).sum()),
    }
    bad = ok[(ok.area < 0.05) | (ok.area > 0.9) | (ok.large_components > 1) | (ok.align_err > 0.10)]
    worst = bad.assign(score=bad.large_components + 10 * bad.align_err + (bad.area < 0.05) + (bad.area > 0.9)) \
        .sort_values("score", ascending=False).head(8)
    if len(worst):
        tiles = []
        for r in worst.itertuples():
            pair = Image.new("RGB", (256, 128))
            pair.paste(load_image(r.color_path).resize((128, 128)), (0, 0))
            pair.paste(load_image(r.path).resize((128, 128)), (128, 0))
            tiles.append(pair)
        sheet = Image.new("RGB", (512, 128 * ((len(tiles) + 1) // 2)), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 2) * 256, (i // 2) * 128))
        sheet.save(OUT / "segmented_worst.jpg", quality=90)
    return (
        "## 7. Segmented config\n\n"
        f"- Shared-class color-train images with a segmented counterpart (by filename): "
        f"**{stats['color_train_with_segmented']} / {stats['color_train_shared']}**\n"
        f"- Segmented images with no color counterpart: {stats['segmented_without_color']} of {stats['segmented_total']}\n"
        f"- Segmented-*train* items whose color version is in color-*test*: **{stats['segmented_train_listed_but_color_test']}** "
        "(the two configs were split differently; compositing uses color-train counterparts only, so these never enter training)\n"
        f"- Mask quality on {len(q)} sampled leaves:\n\n"
        + pd.Series(flags, name="images").to_frame().to_markdown() + "\n\n"
        f"Median mask area {ok.area.median():.2f}, median alignment error {ok.align_err.median():.3f}. "
        "Worst cases (color | segmented): `segmented_worst.jpg`; all rows: `segmented_mask_quality.csv`.\n"
    )


# ---------------------------------------------------------------- label review
def _sheet(paths: list[str], title: str, out: Path, tile: int = 160, cols: int = 5) -> None:
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile + 24), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), title, fill="black")
    for i, p in enumerate(paths):
        im = load_image(p)
        im.thumbnail((tile - 4, tile - 4))
        x, y = (i % cols) * tile + 2, (i // cols) * tile + 26
        sheet.paste(im, (x, y))
        d.rectangle([x, y, x + 22, y + 14], fill="yellow")
        d.text((x + 3, y + 1), str(i), fill="black")
    sheet.save(out, quality=88)


def section_labels(cfg) -> str:
    m = manifests(cfg)
    pdoc = pd.concat([m["pd_train"], m["pd_val"], m["pd_test"]])
    rows = []
    for dataset, df, k in (("plantdoc", pdoc, 20), ("plantvillage", m["pv_train"], 10)):
        d = OUT / "labels" / dataset
        d.mkdir(parents=True, exist_ok=True)
        for cls, g in df.groupby("class_name"):
            pick = g.sample(n=min(k, len(g)), random_state=0).path.tolist()
            _sheet(pick, f"{dataset}: {cls}  (mark wrong tiles in label_review.csv)", d / f"{cls.replace(' ', '_')}.jpg")
            rows += [{"dataset": dataset, "class_name": cls, "tile": i, "path": p, "correct": ""} for i, p in enumerate(pick)]
    review = OUT / "label_review.csv"
    if not review.exists():  # never overwrite a review in progress
        pd.DataFrame(rows).to_csv(review, index=False)
    return (
        "## 4. Label quality\n\n"
        "Contact sheets: `reports/audit/labels/{plantdoc,plantvillage}/<class>.jpg` (20 PlantDoc / 10 PlantVillage "
        "per class, numbered tiles). Fill the `correct` column of `label_review.csv` with 1/0, then run "
        "`python -m src.data.audit --score-labels`.\n"
    )


def score_labels() -> None:
    df = pd.read_csv(OUT / "label_review.csv")
    df = df[df.correct.astype(str).isin(["0", "1", "0.0", "1.0"])]
    if df.empty:
        raise SystemExit("label_review.csv has no filled `correct` values yet")
    lines = ["## 4b. Label error estimate (manual review)\n", "| dataset | reviewed | est. label error | 95% CI |", "|---|---|---|---|"]
    for name, g in df.groupby("dataset"):
        wrong = 1 - g.correct.astype(float).values
        lo, hi = bootstrap_accuracy_ci(wrong, 2000)
        lines.append(f"| {name} | {len(g)} | {wrong.mean():.1%} | {lo:.1%} – {hi:.1%} |")
    per_class = df[df.dataset == "plantdoc"].groupby("class_name").correct.apply(lambda s: 1 - s.astype(float).mean())
    lines += ["", "Worst PlantDoc classes (est. error): " + ", ".join(f"{c} {e:.0%}" for c, e in per_class.nlargest(5).items()),
              "", "Accuracy on PlantDoc test is capped near (1 − label error); state this next to the headline number.\n"]
    (OUT / "label_error.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=SECTIONS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--score-labels", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.score_labels:
        score_labels()
        return
    cfg = load_config(args.config, args.overrides)
    run = {"leaf": lambda: section_leaf(cfg), "counts": lambda: section_counts(cfg),
           "files": lambda: section_files(cfg, args.workers), "stats": lambda: section_stats(cfg, args.workers),
           "seg": lambda: section_seg(cfg, args.workers), "labels": lambda: section_labels(cfg)}
    for name in args.only:
        print(f"[audit] {name} ...", flush=True)
        text = run[name]()
        (OUT / f"{name}.md").write_text(text)
        print(text)
    parts = [(OUT / f"{s}.md").read_text() for s in SECTIONS if (OUT / f"{s}.md").exists()]
    (OUT / "audit.md").write_text("# Data audit\n\n" + "\n".join(parts))
    print(f"[audit] wrote {OUT / 'audit.md'}")


if __name__ == "__main__":
    main()
