"""Lab vs field validation accuracy over training epochs (from train_log.json; no model is re-run).

    python -m src.curves   # -> reports/figures/val_curves.png + reports/val_curves.md

For lab-only runs (E0, E2), field accuracy tends to peak early and then drift down while lab accuracy keeps
rising: training longer on lab images makes the model worse in the field. The table compares the field-val
accuracy at the lab-selected epoch with the field-val peak (what field-val selection would have picked).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.report import model_key


def curves(runs: Path, exp: str) -> dict[str, list[pd.DataFrame]]:
    out: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(runs.glob(f"{exp}/*/seed*/train_log.json")):
        log = json.loads(f.read_text())
        h = pd.DataFrame(log["history"])
        if {"val_pv_val", "val_pd_val"} <= set(h.columns):
            out.setdefault(f.parts[-3], []).append(h.assign(best=log["best_epoch"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--exps", nargs="+", default=["e0", "e2"])
    args = ap.parse_args()
    runs = Path(args.runs)
    data = {e: curves(runs, e) for e in args.exps}
    models = sorted({m for d in data.values() for m in d}, key=model_key)
    if not models:
        raise SystemExit("no train_log.json with both val curves found")

    fig, axes = plt.subplots(len(args.exps), len(models), figsize=(3.6 * len(models), 2.9 * len(args.exps)), squeeze=False)
    rows = []
    for i, exp in enumerate(args.exps):
        for j, model in enumerate(models):
            ax = axes[i, j]
            for k, h in enumerate(data[exp].get(model, [])):
                ep = h.epoch + 1
                ax.plot(ep, h.val_pv_val, color="tab:blue", alpha=0.8, lw=1, label="lab val (PlantVillage)" if k == 0 else None)
                ax.plot(ep, h.val_pd_val, color="tab:orange", alpha=0.8, lw=1, label="field val (PlantDoc)" if k == 0 else None)
                b = int(h.best.iloc[0])
                ax.scatter([b + 1], [h.val_pd_val.iloc[b]], color="tab:orange", s=14, zorder=3)
                rows.append({"exp": exp, "model": model, "field_val_at_lab_selected": h.val_pd_val.iloc[b],
                             "field_val_peak": h.val_pd_val.max(), "peak_epoch": int(h.val_pd_val.idxmax()) + 1,
                             "lab_selected_epoch": b + 1})
            ax.set_title(f"{exp} · {model}", fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_xlabel("epoch", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=7, loc="center right")
    fig.suptitle("Lab accuracy keeps rising while field accuracy peaks early (dots: field acc at the lab-selected epoch)",
                 fontsize=10)
    fig.tight_layout()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("reports/figures/val_curves.png", dpi=140)

    t = pd.DataFrame(rows).groupby(["exp", "model"]).agg(
        field_val_at_lab_selected=("field_val_at_lab_selected", "mean"), field_val_peak=("field_val_peak", "mean"),
        peak_epoch=("peak_epoch", "mean"), lab_selected_epoch=("lab_selected_epoch", "mean"), seeds=("peak_epoch", "size"))
    t["optimism_if_selected_on_field_val"] = t.field_val_peak - t.field_val_at_lab_selected
    md = ("# Lab vs field validation curves\n\n![curves](figures/val_curves.png)\n\n"
          "Field-val accuracy at the lab-selected checkpoint vs the field-val peak (mean over seeds). The difference is how "
          "much choosing checkpoints on field data would flatter a lab-only model.\n\n" + t.round(3).to_markdown() + "\n")
    Path("reports/val_curves.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
