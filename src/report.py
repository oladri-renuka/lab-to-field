"""Aggregate runs into reports/results.{md,csv}, paired significance tests and figures.

    python -m src.report            # results (mean ± std over seeds), paired tests (Holm), compute tables
    python -m src.report --select   # pick the best of E1-E4 per model by PlantDoc-VAL accuracy (for E6)

Headline metric: PlantDoc test accuracy. The lab-field gap is secondary: fine-tuning on field images also lowers
lab accuracy, which shrinks the gap without any field improvement.
Lab-only rows (E0-E4) select checkpoints on PlantVillage val; the PlantDoc-val-selected number is reported next to
it as an optimistic secondary.
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

EXP_ORDER = ["e0", "e1", "e2", "e3", "e4", "e5_k5", "e5_k10", "e5_k25", "e6", "e7"]
MODEL_ORDER = ["resnet50", "efficientnet_b0", "vit_s16", "vit_s16_in1k"]
REVIEW = Path("reports/audit/test_label_review.csv")
PAIRS = [("e2", "e1"), ("e3", "e2"), ("e4", "e2"), ("e5_k10", "e3"), ("e5_k10", "e4"), ("e5_k10", "e2"),
         ("e6", "e5_k10"), ("e6", "e2")]  # E6 = best of E1-E4 + 10/class: does the combination beat few-shot alone?
# Cross-architecture comparisons (same experiment, same test images). vit_s16_in1k separates architecture from
# pretraining data: vit_s16 vs vit_s16_in1k = 21k-vs-1k pretraining; vit_s16_in1k vs CNNs = architecture at 1k.
MODEL_PAIRS = [("vit_s16", "resnet50"), ("vit_s16", "efficientnet_b0"), ("resnet50", "efficientnet_b0"),
               ("vit_s16_in1k", "resnet50"), ("vit_s16_in1k", "efficientnet_b0"), ("vit_s16", "vit_s16_in1k")]
CROSS_MODEL_EXPS = ["e0", "e1", "e2"]


def base_exp(e: str) -> str:
    return "e6" if e.startswith("e6") else e


def exp_key(e: str) -> int:
    b = base_exp(e)
    return EXP_ORDER.index(b) if b in EXP_ORDER else len(EXP_ORDER)


def model_key(m: str) -> int:
    return MODEL_ORDER.index(m) if m in MODEL_ORDER else 99


# ------------------------------------------------------------------ collection
def reviewed_labels(classes: list[str]) -> dict[str, int | None] | None:
    """path -> corrected label index, or None to drop. Only rows with `correct` filled count as reviewed."""
    if not REVIEW.exists():
        return None
    df = pd.read_csv(REVIEW, dtype=str).fillna("")
    df = df[df.correct.str.strip().isin(["0", "1", "0.0", "1.0"])]
    if df.empty:
        return None
    idx = {c: i for i, c in enumerate(classes)}
    out = {}
    for r in df.itertuples():
        if float(r.correct) == 1:
            out[r.path] = idx[r.class_name]
        else:
            out[r.path] = idx.get(r.corrected_class.strip())  # None (dropped) if empty / not a shared class
    return out


def reviewed_acc(run: Path, review: dict, n_test: int) -> float:
    f = run / "preds_primary_pd_test.npz"
    if review is None or len(review) < n_test or not f.exists():
        return float("nan")  # only once every test image is reviewed
    p = np.load(f)
    keep = [(i, review[str(path)]) for i, path in enumerate(p["paths"]) if review.get(str(path)) is not None]
    idx, lab = np.array([k for k, _ in keep]), np.array([v for _, v in keep])
    return float((p["probs"][idx].argmax(1) == lab).mean())


def collect(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs.glob("*/*/seed*/metrics.json")):
        m = json.loads(f.read_text())
        if "pd_test" not in m:  # e.g. the background-only shortcut run
            continue
        sec = f.with_name("metrics_pd_val_selected.json")
        m2 = json.loads(sec.read_text()) if sec.exists() else None
        val = m.get("val_acc") if isinstance(m.get("val_acc"), dict) else {"pd_val": m.get("pd_val_acc")}
        rows.append({
            "exp": m["exp_id"], "model": m["model"], "seed": m["seed"], "run": str(f.parent),
            "selected_on": m.get("selected_on", "pd_val"), "pd_val_acc": val.get("pd_val"),
            "pd_acc": m["pd_test"]["acc"], "pd_macro_f1": m["pd_test"]["macro_f1"], "pd_ece": m["pd_test"]["ece"],
            "pv_acc": m["pv_test"]["acc"], "gap": m["gap"], "n_classes": m["n_classes"], "n_test": m["pd_test"]["n"],
            "pd_acc_fieldsel": m2["pd_test"]["acc"] if m2 else np.nan,
            "pd_pool_acc": m["pd_pool"]["acc"] if "pd_pool" in m else np.nan,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pd_acc", "pd_macro_f1", "pd_ece", "pv_acc", "gap", "pd_acc_fieldsel", "pd_pool_acc", "pd_acc_reviewed"]
    metrics = [m for m in metrics if m in df]
    g = df.groupby(["exp", "model"])
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    out["selected_on"] = g.selected_on.first()
    out = out.reset_index()
    out["_e"], out["_m"] = out.exp.map(exp_key), out.model.map(model_key)
    return out.sort_values(["_e", "_m"]).drop(columns=["_e", "_m"])


def fmt(r, k: str) -> str:
    m, s = r.get(f"{k}_mean"), r.get(f"{k}_std")
    if m is None or pd.isna(m):
        return "—"
    return f"{m:.3f} ± {0 if pd.isna(s) else s:.3f}"


def to_markdown(s: pd.DataFrame, reviewed: bool) -> str:
    head = ["Exp", "Model", "**PlantDoc acc**", "macro-F1", "ECE", "PlantVillage acc", "Gap (PV−PD)",
            "PlantDoc acc, PD-val-selected†", "selected on", "seeds"]
    if reviewed:
        head.insert(3, "PlantDoc acc, reviewed labels")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in s.iterrows():
        cells = [r.exp, r.model, f"**{fmt(r, 'pd_acc')}**", fmt(r, "pd_macro_f1"), fmt(r, "pd_ece"), fmt(r, "pv_acc"),
                 fmt(r, "gap"), fmt(r, "pd_acc_fieldsel"), r.selected_on, str(r.n_seeds)]
        if reviewed:
            cells.insert(3, fmt(r, "pd_acc_reviewed"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("\n† Optimistic: checkpoint chosen on PlantDoc (field) val. Only reported for rows whose headline "
                 "checkpoint is chosen on PlantVillage (lab) val.")
    return "\n".join(lines)


# ------------------------------------------------------------------ paired tests
def per_image_correct(runs_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean correctness per test image over a cell's seeds, aligned by image path."""
    mats, paths = [], None
    for run in runs_df.run:
        f = Path(run) / "preds_primary_pd_test.npz"
        if not f.exists():
            return None
        p = np.load(f)
        order = np.argsort(p["paths"])
        if paths is None:
            paths = p["paths"][order]
        elif not np.array_equal(paths, p["paths"][order]):
            return None
        mats.append((p["probs"].argmax(1) == p["labels"])[order].astype(float))
    return (np.mean(mats, 0), paths) if mats else None


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float, float, float]:
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(p)


def mcnemar_exact(ca: np.ndarray, cb: np.ndarray) -> float:
    from scipy.stats import binom

    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return 1.0 if b + c == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5)))


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def _compare(cells_a, cells_b, n_a: int, n_b: int, **ids) -> dict | None:
    (ca, pa), (cb, pb) = cells_a, cells_b
    if not np.array_equal(pa, pb):
        return None
    diff, lo, hi, p = paired_bootstrap(ca, cb)
    return {**ids, "ΔPlantDoc acc (A−B)": diff, "95% CI": f"[{lo:+.3f}, {hi:+.3f}]", "p (paired bootstrap)": p,
            "p McNemar": mcnemar_exact(ca >= 0.5, cb >= 0.5), "seeds A/B": f"{n_a}/{n_b}"}


def fmt_p(p: float) -> str:
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def paired_tests(df: pd.DataFrame) -> str:
    key = df.exp.map(base_exp)
    cells, nseeds = {}, {}
    for (model, exp), g in df.groupby([df.model, key]):
        c = per_image_correct(g)
        if c is not None:
            cells[(model, exp)], nseeds[(model, exp)] = c, len(g)
    rows = []
    for model in sorted({m for m, _ in cells}, key=model_key):
        exps = [e for m, e in cells if m == model]
        pairs = [(e, "e0") for e in exps if e != "e0" and "e0" in exps] + [p for p in PAIRS if p[0] in exps and p[1] in exps]
        for a, b in dict.fromkeys(pairs):
            r = _compare(cells[(model, a)], cells[(model, b)], nseeds[(model, a)], nseeds[(model, b)],
                         kind="within model", model=model, A=a, B=b)
            if r:
                rows.append(r)
    for exp in CROSS_MODEL_EXPS:
        for ma, mb in MODEL_PAIRS:
            if (ma, exp) in cells and (mb, exp) in cells:
                r = _compare(cells[(ma, exp)], cells[(mb, exp)], nseeds[(ma, exp)], nseeds[(mb, exp)],
                             kind="across models", model=exp, A=ma, B=mb)
                if r:
                    rows.append(r)
    if not rows:
        return ""
    t = pd.DataFrame(rows)
    t["p Holm"] = holm(t["p (paired bootstrap)"].tolist())
    t["p McNemar Holm"] = holm(t["p McNemar"].tolist())
    t["significant (Holm < 0.05)"] = np.where(t["p Holm"] < 0.05, "yes", "no")
    t.to_csv("reports/paired_tests.csv", index=False)
    shown = t.copy()
    for c in ["p (paired bootstrap)", "p McNemar", "p Holm", "p McNemar Holm"]:
        shown[c] = shown[c].map(fmt_p)
    shown["ΔPlantDoc acc (A−B)"] = shown["ΔPlantDoc acc (A−B)"].map(lambda x: f"{x:+.4f}")
    shown = shown.rename(columns={"model": "model / exp"})
    return ("## Paired comparisons on the same PlantDoc test images\n\n"
            "Per-image correctness averaged over each cell's seeds; paired bootstrap over images (10k resamples) is the "
            "primary test, exact McNemar on majority-vote correctness is supplementary. **One Holm family** covers all "
            f"{len(t)} comparisons (within-model experiment pairs and across-model pairs). `<1e-4` = no bootstrap "
            "resample crossed zero. With 236 test images, differences under ~5 points are unlikely to be detectable.\n\n"
            "Across-model rows: `vit_s16` (ImageNet-21k) vs `vit_s16_in1k` (same model and recipe, ImageNet-1k only) "
            "measures the pretraining-data effect; `vit_s16_in1k` vs the CNNs (also ImageNet-1k) measures architecture "
            "at matched pretraining data.\n\n" + shown.to_markdown(index=False) + "\n")


def shortcut_table(runs: Path) -> str:
    rows = []
    for f in sorted(runs.glob("shortcut_hull/*/seed*/metrics.json")):
        m = json.loads(f.read_text())
        rows.append({"model": m["model"], "acc": m["test_acc"], "majority": m["chance_majority"],
                     "uniform": m["chance_uniform"]})
    if not rows:
        return ""
    d = pd.DataFrame(rows)
    g = d.groupby("model")
    lines = ["## Background-only shortcut test, per architecture (PlantVillage test, leaf-free patches)\n",
             f"Majority-class baseline {d.majority.iloc[0]:.3f}; uniform chance {d.uniform.iloc[0]:.3f}.\n",
             "| Model | Background-only acc | seeds |", "|---|---|---|"]
    for model in sorted(g.groups, key=model_key):
        a = g.get_group(model).acc
        lines.append(f"| {model} | {a.mean():.3f} ± {0 if len(a) < 2 else a.std():.3f} | {len(a)} |")
    lines.append("\nThis measures how well each architecture *can* exploit background statistics. How much a lab-trained "
                 "model *relies* on them is read from the E2−E0 gain and the saliency figures.\n")
    return "\n".join(lines) + "\n"


def recovered_share(s: pd.DataFrame) -> str:
    """(E - E0) / (E7 - E0) on PlantDoc accuracy: share of the achievable field gain each method recovers."""
    piv = s.assign(base=s.exp.map(base_exp)).pivot_table(index="base", columns="model", values="pd_acc_mean")
    if "e0" not in piv.index or "e7" not in piv.index:
        return ""
    models = [m for m in MODEL_ORDER if m in piv.columns]
    rows = [e for e in EXP_ORDER if e in piv.index and e not in ("e0", "e7")]
    lines = ["## Share of the achievable field gain recovered: (E − E0) / (E7 − E0)\n",
             "E0 = lab only, E7 = PlantDoc train only (reference). Computed on mean PlantDoc accuracy; significance is in "
             "the paired-comparison table.\n", "| Exp | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for e in rows:
        cells = []
        for m in models:
            e0, e7, v = piv.loc["e0", m], piv.loc["e7", m], piv.loc[e, m]
            cells.append("—" if pd.isna(v) or e7 == e0 else f"{(v - e0) / (e7 - e0):+.0%}")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ figures + compute
def bar_figure(s: pd.DataFrame, col: str, ylabel: str, title: str, path: Path) -> None:
    exps = list(dict.fromkeys(s.exp))
    models = [m for m in MODEL_ORDER if m in set(s.model)]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(exps)), 3.6))
    w = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        sub = s[s.model == m].set_index("exp").reindex(exps)
        ax.bar([j + i * w for j in range(len(exps))], sub[f"{col}_mean"], w, yerr=sub[f"{col}_std"].fillna(0),
               label=m, capsize=2)
    ax.set_xticks([j + w * (len(models) - 1) / 2 for j in range(len(exps))], exps)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_tables(runs: Path) -> str:
    rows = []
    for f in sorted(runs.glob("compute_*/*/seed*/train_log.json")) + sorted(runs.glob("ddp*/*/seed*/train_log.json")):
        log = json.loads(f.read_text())
        h = log["history"][1:] or log["history"]  # skip epoch 0 (cudnn autotune / warm caches)
        if not h:
            continue
        exp, model = f.parts[-4], f.parts[-3]
        mpath = f.with_name("metrics.json")
        acc = json.loads(mpath.read_text())["pd_test"]["acc"] if mpath.exists() else float("nan")
        rows.append({"run": exp, "model": model, "world": log["world_size"], "global_batch": log["global_batch"],
                     "img_per_s": sum(x["images_per_sec_global"] for x in h) / len(h),
                     "peak_mem_gb": max(x["peak_mem_gb"] for x in h),
                     "epoch_s": sum(x["epoch_time_s"] for x in h) / len(h),
                     "data_wait": sum(x["data_wait_frac"] for x in h) / len(h), "pd_acc": acc})
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    ddp = df[df.run.str.startswith("ddp")]
    if len(ddp) and (ddp.world == 1).any():
        base = ddp[ddp.world == 1].img_per_s.iloc[0]
        df.loc[ddp.index, "scaling_eff"] = ddp.img_per_s / (ddp.world * base)
    return "## Compute experiments\n\n" + df.round(3).to_markdown(index=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    runs = Path(args.runs)
    df = collect(runs)
    if df.empty:
        raise SystemExit(f"no metrics.json under {runs}")
    if args.select:
        cand = df[df.exp.isin(["e1", "e2", "e3", "e4"])].groupby(["model", "exp"]).pd_val_acc.mean().reset_index()
        best = cand.loc[cand.groupby("model").pd_val_acc.idxmax()]
        print("Best of E1–E4 by PlantDoc-VAL accuracy (use as init_experiment for E6):")
        print(best.to_string(index=False))
        return

    review = None
    classes_file = Path(args.processed_dir) / "classes.json"
    if classes_file.exists():
        review = reviewed_labels(json.loads(classes_file.read_text())["classes"])
    df["pd_acc_reviewed"] = [reviewed_acc(Path(r), review, n) for r, n in zip(df.run, df.n_test)]
    has_review = df.pd_acc_reviewed.notna().any()

    s = summarize(df)
    out = Path("reports")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "results.csv", index=False)
    md = (f"# Results (PlantDoc test: {int(df.n_test.iloc[0])} images, {int(df.n_classes.iloc[0])} shared classes)\n\n"
          "Headline: **PlantDoc test accuracy** (mean ± std over seeds). The gap is secondary (fine-tuning on field "
          "images also lowers lab accuracy, which shrinks the gap without any field gain).\n\n" + to_markdown(s, has_review))
    pool = s[s["pd_pool_acc_mean"].notna()] if "pd_pool_acc_mean" in s.columns else s.iloc[0:0]
    if len(pool):
        md += ("\n\n## Secondary field test: PlantDoc train split (E0/E1 only, which never see PlantDoc pixels)\n\n"
               + "\n".join(["| Exp | Model | PlantDoc-train acc | seeds |", "|---|---|---|---|"]
                           + [f"| {r.exp} | {r.model} | {fmt(r, 'pd_pool_acc')} | {r.n_seeds} |" for _, r in pool.iterrows()]))
    md += "\n\n" + recovered_share(s) + "\n" + paired_tests(df) + "\n" + shortcut_table(runs) + "\n" + compute_tables(runs)
    (out / "results.md").write_text(md)
    bar_figure(s, "pd_acc", "PlantDoc test accuracy", "Field accuracy per experiment (mean ± std over seeds)",
               out / "figures" / "pd_acc.png")
    bar_figure(s, "gap", "Lab − field accuracy gap", "Lab-to-field gap (secondary; lower is better)",
               out / "figures" / "gap.png")
    print(md)


if __name__ == "__main__":
    main()
