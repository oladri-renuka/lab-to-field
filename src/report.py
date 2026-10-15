"""Aggregate runs/*/*/seed*/metrics.json into reports/results.{md,csv} and the gap figure.

    python -m src.report            # results table (mean ± std over seeds) + compute tables
    python -m src.report --select   # pick the best of E1-E4 per model by PlantDoc-VAL accuracy (for E6)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXP_ORDER = ["e0", "e1", "e2", "e3", "e4", "e5_k5", "e5_k10", "e5_k25", "e6", "e7"]
MODEL_ORDER = ["resnet50", "efficientnet_b0", "vit_s16"]


def collect(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs.glob("*/*/seed*/metrics.json")):
        m = json.loads(f.read_text())
        rows.append({
            "exp": m["exp_id"], "model": m["model"], "seed": m["seed"], "pd_val_acc": m["pd_val_acc"],
            "pd_acc": m["pd_test"]["acc"], "pd_macro_f1": m["pd_test"]["macro_f1"], "pd_ece": m["pd_test"]["ece"],
            "pv_acc": m["pv_test"]["acc"], "gap": m["gap"], "n_classes": m["n_classes"],
        })
    return pd.DataFrame(rows)


def exp_key(e: str) -> int:
    base = "e6" if e.startswith("e6") else e
    return EXP_ORDER.index(base) if base in EXP_ORDER else len(EXP_ORDER)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pd_acc", "pd_macro_f1", "pd_ece", "pv_acc", "gap"]
    g = df.groupby(["exp", "model"])
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    out = out.reset_index()
    out["_e"], out["_m"] = out.exp.map(exp_key), out.model.map(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    return out.sort_values(["_e", "_m"]).drop(columns=["_e", "_m"])


def to_markdown(s: pd.DataFrame) -> str:
    fmt = lambda r, k: f"{r[k + '_mean']:.3f} ± {0 if pd.isna(r[k + '_std']) else r[k + '_std']:.3f}"
    lines = ["| Exp | Model | PlantDoc acc | PlantDoc macro-F1 | ECE | PlantVillage acc | Gap (PV−PD) ↓ | seeds |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.exp} | {r.model} | {fmt(r, 'pd_acc')} | {fmt(r, 'pd_macro_f1')} | {fmt(r, 'pd_ece')} | "
                     f"{fmt(r, 'pv_acc')} | {fmt(r, 'gap')} | {r.n_seeds} |")
    return "\n".join(lines)


def gap_figure(s: pd.DataFrame, path: Path) -> None:
    exps = list(dict.fromkeys(s.exp))
    models = [m for m in MODEL_ORDER if m in set(s.model)]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(exps)), 3.6))
    w = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        sub = s[s.model == m].set_index("exp").reindex(exps)
        ax.bar([j + i * w for j in range(len(exps))], sub.gap_mean, w, yerr=sub.gap_std.fillna(0), label=m, capsize=2)
    ax.set_xticks([j + w * (len(models) - 1) / 2 for j in range(len(exps))], exps)
    ax.set_ylabel("Lab − field accuracy gap")
    ax.set_title("Lab-to-field gap per experiment (lower is better; mean ± std over seeds)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


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
    s = summarize(df)
    out = Path("reports")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "results.csv", index=False)
    md = f"# Results (PlantDoc test, {int(df.n_classes.iloc[0])} shared classes)\n\n" + to_markdown(s) + "\n\n" + compute_tables(runs)
    (out / "results.md").write_text(md)
    gap_figure(s, out / "figures" / "gap.png")
    print(md)


if __name__ == "__main__":
    main()
