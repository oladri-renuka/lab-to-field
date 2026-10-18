"""Failure analysis from saved per-image predictions (no model is re-run).

    python -m src.failure      # -> reports/failure_analysis.md + reports/figures/{per_class_recall,confusions,
                               #    failures,reliability}_<model>.png

Uses runs/<exp>/<model>/seed*/preds_primary_pd_test.npz written by src.eval. Seeds are pooled:
recall is averaged over seeds, confusion counts are averaged over seeds, probabilities are averaged per image.
If the manual test-label review exists, failure examples whose label was judged wrong are marked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.io import load_image
from src.report import REVIEW, base_exp, exp_key, model_key

FIG = Path("reports/figures")


def short(c: str) -> str:
    crop, _, dis = c.partition("___")
    return f"{crop.split('_(')[0].split(',')[0]}: {dis.replace('_', ' ').strip()}"


def load_cells(runs: Path) -> dict[tuple[str, str], dict]:
    """(model, exp) -> {probs: [S, N, C], labels: [N], paths: [N]} aligned by path across seeds."""
    cells: dict[tuple[str, str], list] = {}
    for f in sorted(runs.glob("*/*/seed*/preds_primary_pd_test.npz")):
        m = json.loads((f.parent / "metrics.json").read_text())
        p = np.load(f)
        order = np.argsort(p["paths"])
        cells.setdefault((m["model"], base_exp(m["exp_id"])), []).append(
            (p["probs"][order], p["labels"][order], p["paths"][order]))
    out = {}
    for k, lst in cells.items():
        paths = lst[0][2]
        if all(np.array_equal(paths, x[2]) for x in lst):
            out[k] = {"probs": np.stack([x[0] for x in lst]), "labels": lst[0][1], "paths": paths}
    return out


def per_class_recall(cell: dict, n_classes: int) -> np.ndarray:
    rec = np.full((len(cell["probs"]), n_classes), np.nan)
    for s, probs in enumerate(cell["probs"]):
        pred = probs.argmax(1)
        for c in range(n_classes):
            mask = cell["labels"] == c
            if mask.any():
                rec[s, c] = (pred[mask] == c).mean()
    return np.nanmean(rec, 0)


def confusion(cell: dict, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes))
    for probs in cell["probs"]:
        np.add.at(cm, (cell["labels"], probs.argmax(1)), 1)
    return cm / len(cell["probs"])


def recall_heatmap(rec: pd.DataFrame, n_per_class: pd.Series, model: str, path: Path) -> None:
    rec = rec.sort_values(rec.columns[0])
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * rec.shape[1], 0.32 * len(rec) + 1.5))
    im = ax.imshow(rec.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(rec.shape[1]), rec.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rec)), [f"{short(c)} (n={n_per_class[c]})" for c in rec.index], fontsize=7)
    for i in range(rec.shape[0]):
        for j in range(rec.shape[1]):
            v = rec.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v < 0.55 else "black")
    ax.set_title(f"{model}: PlantDoc-test recall per class (mean over seeds; rows sorted by E0)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def confusion_figure(cms: dict[str, np.ndarray], classes: list[str], model: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(cms), figsize=(6.2 * len(cms), 6))
    axes = np.atleast_1d(axes)
    for ax, (exp, cm) in zip(axes, cms.items()):
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1e-9)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{model} {exp}: row-normalised confusion", fontsize=9)
        ax.set_xticks(range(len(classes)), [short(c) for c in classes], rotation=90, fontsize=5)
        ax.set_yticks(range(len(classes)), [short(c) for c in classes], fontsize=5)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def reliability_figure(cells: dict[str, dict], model: str, path: Path, bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    edges = np.linspace(0, 1, bins + 1)
    for exp, cell in cells.items():
        conf = cell["probs"].max(2).ravel()
        corr = (cell["probs"].argmax(2) == cell["labels"][None]).ravel()
        idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
        xs = [conf[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 5]
        ys = [corr[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 5]
        ax.plot(xs, ys, marker="o", ms=3, label=exp)
    ax.set(xlabel="confidence", ylabel="accuracy", title=f"{model}: reliability on PlantDoc test", xlim=(0, 1), ylim=(0, 1))
    ax.axvline(0.5, color="grey", lw=0.6, ls=":")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def failure_grid(cell: dict, classes: list[str], review: dict, model: str, exp: str, path: Path, n: int = 16) -> None:
    mean = cell["probs"].mean(0)
    pred, conf = mean.argmax(1), mean.max(1)
    wrong = np.flatnonzero(pred != cell["labels"])
    worst = wrong[np.argsort(-conf[wrong])][:n]
    cols = 4
    fig, axes = plt.subplots((len(worst) + cols - 1) // cols, cols, figsize=(3.0 * cols, 3.3 * ((len(worst) + cols - 1) // cols)))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), worst):
        ax.imshow(load_image(str(cell["paths"][i])).resize((224, 224)))
        flag = review.get(str(cell["paths"][i]))
        note = "  [label judged WRONG]" if flag == 0 else ""
        ax.set_title(f"true: {short(classes[cell['labels'][i]])}\npred: {short(classes[pred[i]])} ({conf[i]:.2f}){note}",
                     fontsize=6.5, color="darkred")
    fig.suptitle(f"{model} {exp}: most confident errors on PlantDoc test (probabilities averaged over seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_review() -> dict[str, int]:
    if not REVIEW.exists():
        return {}
    df = pd.read_csv(REVIEW, dtype=str).fillna("")
    df = df[df.correct.str.strip().isin(["0", "1", "0.0", "1.0"])]
    return {r.path: int(float(r.correct)) for r in df.itertuples()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--processed-dir", default="data/processed")
    args = ap.parse_args()
    classes = json.loads((Path(args.processed_dir) / "classes.json").read_text())["classes"]
    C = len(classes)
    cells = load_cells(Path(args.runs))
    if not cells:
        raise SystemExit("no preds_primary_pd_test.npz found; run training/eval first")
    review = load_review()
    FIG.mkdir(parents=True, exist_ok=True)
    out_dir = Path("reports/failure")
    out_dir.mkdir(parents=True, exist_ok=True)

    any_cell = next(iter(cells.values()))
    n_per_class = pd.Series(np.bincount(any_cell["labels"], minlength=C), index=classes)
    models = sorted({m for m, _ in cells}, key=model_key)
    md = ["# Failure analysis (PlantDoc test)\n",
          f"Built from saved predictions of {len(cells)} experiment×model cells. Test set: {int(n_per_class.sum())} images, "
          f"{C} classes, {n_per_class.min()}–{n_per_class.max()} per class, so **per-class numbers are noisy**: one image is "
          f"~{100 / n_per_class.median():.0f} points of a class's recall.\n"]
    all_rec = {}
    for model in models:
        exps = sorted({e for m, e in cells if m == model}, key=exp_key)
        rec = pd.DataFrame({e: per_class_recall(cells[(model, e)], C) for e in exps}, index=classes)
        all_rec[model] = rec
        rec.to_csv(out_dir / f"per_class_recall_{model}.csv")
        recall_heatmap(rec, n_per_class, model, FIG / f"per_class_recall_{model}.png")

        md.append(f"## {model}\n")
        md.append(f"![per-class recall](figures/per_class_recall_{model}.png)\n")
        cms = {e: confusion(cells[(model, e)], C) for e in ("e0", "e5_k10") if (model, e) in cells}
        if cms:
            confusion_figure(cms, classes, model, FIG / f"confusions_{model}.png")
            md.append(f"![confusions](figures/confusions_{model}.png)\n")
        if "e0" in cms:
            cm = cms["e0"].copy()
            np.fill_diagonal(cm, 0)
            top = np.dstack(np.unravel_index(np.argsort(-cm.ravel())[:8], cm.shape))[0]
            md.append("Most frequent E0 (lab-only) confusions, images per seed:\n\n| true | predicted as | count |\n|---|---|---|")
            md += [f"| {short(classes[t])} | {short(classes[p])} | {cm[t, p]:.1f} |" for t, p in top if cm[t, p] > 0]
            pred_share = cms["e0"].sum(0) / cms["e0"].sum()
            top3 = np.argsort(-pred_share)[:3]
            md.append(f"\nPrediction collapse (E0): the 3 most-predicted classes take **{pred_share[top3].sum():.0%}** of all "
                      f"predictions ({', '.join(f'{short(classes[c])} {pred_share[c]:.0%}' for c in top3)}); "
                      f"uniform would be {3 / C:.0%}.\n")
            failure_grid(cells[(model, "e0")], classes, review, model, "e0", FIG / f"failures_e0_{model}.png")
            md.append(f"![worst errors](figures/failures_e0_{model}.png)\n")
        rel = {e: cells[(model, e)] for e in ("e0", "e2", "e5_k10", "e7") if (model, e) in cells}
        if rel:
            reliability_figure(rel, model, FIG / f"reliability_{model}.png")
            md.append(f"![reliability](figures/reliability_{model}.png)\n")

    # classes that stay hard even with field data, for every model
    hard_exp = "e5_k25"
    hard = pd.DataFrame({m: r[hard_exp] for m, r in all_rec.items() if hard_exp in r})
    if not hard.empty:
        worst = hard.mean(1).sort_values().head(8)
        md.append(f"## Classes that stay hard even with 25 field images per class ({hard_exp})\n")
        md.append("| class | n test | " + " | ".join(hard.columns) + " |\n|---|---|" + "---|" * len(hard.columns))
        md += [f"| {short(c)} | {n_per_class[c]} | " + " | ".join(f"{hard.loc[c, m]:.2f}" for m in hard.columns) + " |"
               for c in worst.index]
    improves = {m: (r["e5_k10"] - r["e0"]) for m, r in all_rec.items() if {"e0", "e5_k10"} <= set(r.columns)}
    if improves:
        imp = pd.DataFrame(improves).mean(1).sort_values()
        md.append("\n## Where 10 field images per class help least / most (recall change E0 → E5 k=10, mean over models)\n")
        md.append("| least helped | Δ recall | most helped | Δ recall |\n|---|---|---|---|")
        md += [f"| {short(a)} | {imp[a]:+.2f} | {short(b)} | {imp[b]:+.2f} |" for a, b in zip(imp.index[:6], imp.index[::-1][:6])]
    if review:
        md.append(f"\nManual test-label review loaded ({len(review)} images): errors on images whose label was judged wrong are "
                  "marked in the error grids.")
    Path("reports/failure_analysis.md").write_text("\n".join(md) + "\n")
    print(f"[failure] wrote reports/failure_analysis.md and figures for {', '.join(models)}")


if __name__ == "__main__":
    main()
