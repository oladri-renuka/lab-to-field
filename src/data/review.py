"""Fill reports/audit/test_label_review.csv from a short list of WRONG tiles (everything else = correct).

    python -m src.data.review "Apple___Apple_scab:3" "Tomato___Early_blight:5=Tomato___Late_blight" ...
    python -m src.data.review --file wrong.txt        # one entry per line, same format
    python -m src.data.review --show                  # print what is recorded

Entry format: <class as on the sheet title>:<tile number>[=<correct class>]. Leave out "=..." if the image fits no
class or is unusable (it is then dropped from the reviewed-label accuracy). Re-running replaces the whole review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REVIEW = Path("reports/audit/test_label_review.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wrong", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--processed-dir", default="data/processed")
    args = ap.parse_args()
    if not REVIEW.exists():
        raise SystemExit(f"{REVIEW} missing: run `python -m src.data.audit --only testlabels` first")
    df = pd.read_csv(REVIEW, dtype=str).fillna("")
    if args.show:
        bad = df[df.correct == "0"]
        print(f"{(df.correct != '').sum()}/{len(df)} reviewed, {len(bad)} marked wrong")
        print(bad[["class_name", "tile", "corrected_class"]].to_string(index=False) if len(bad) else "")
        return
    classes = json.loads((Path(args.processed_dir) / "classes.json").read_text())["classes"]
    entries = list(args.wrong)
    if args.file:
        entries += [ln.strip() for ln in Path(args.file).read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    df["correct"], df["corrected_class"] = "1", ""
    for e in entries:
        target, _, fix = e.partition("=")
        cls, _, tile = target.rpartition(":")
        hit = (df.class_name == cls.strip()) & (df.tile == tile.strip())
        if not hit.any():
            raise SystemExit(f"no tile {tile!r} in class {cls!r} (check spelling against the sheet title)")
        if fix and fix.strip() not in classes:
            raise SystemExit(f"{fix!r} is not a class name; choose from data/processed/classes.json")
        df.loc[hit, "correct"], df.loc[hit, "corrected_class"] = "0", fix.strip()
    df.to_csv(REVIEW, index=False)
    print(f"[review] {len(df)} test images reviewed, {len(entries)} marked wrong -> {REVIEW}")


if __name__ == "__main__":
    main()
