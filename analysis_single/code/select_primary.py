"""Choice of the primary model from development data only, and the development table of the models.

Rule: among the four trained models, the primary model is the one with the highest mean development AUROC, the mean of the AUROC in the Ambe set and in
the non-herbal set (both restricted to at most 500 Da).  Candidates are Chemprop, MapLight-style CatBoost, TabPFN on descriptors and the random forest;
heavy-atom count and Crippen logP are reference scores, not trained models, and are listed in the development table but not ranked.

The rule is applied to the five-seed means (the scores of every table).  The script recomputes the development AUROCs from the prediction files of the
Ambe set and the non-herbal set and never opens a herb prediction file or herb.csv.  It stops if the rule selects a model other than Chemprop.

python code/select_primary.py [--out results/final]
Writes primary_model_selection.csv and primary_model_selection.json.  dev_table() is also used by run_all.py for dev_single_models.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import BUNDLE, DOMAIN, FINAL, PRIMARY, SEEDS, TRAINED, check_fresh, check_frozen_split, pred_file, seed_mean

RULE = ("Primary model = the stand-alone model with the highest mean development AUROC, where the development AUROC is the mean of the AUROC in the "
        "Ambe set and in the non-herbal set (structures of at most 500 Da, mean over the five seeds of the raw probabilities).")


def load_dev():
    """Only the two development tables are opened (the selection never reads the cohort table)."""
    return {k: pd.read_csv(BUNDLE / f"{k}.csv") for k in ("ambe", "stress")}


def dev_frames(bundle):
    """Development sets restricted to the application domain (at most 500 Da)."""
    return {k: bundle[k][bundle[k].mw <= DOMAIN].reset_index(drop=True) for k in ("ambe", "stress")}


def dev_scores(frames, label):
    if label == "Size":
        return {k: f.heavy.values.astype(float) for k, f in frames.items()}
    if label == "LogP":
        return {k: f.logp.values.astype(float) for k, f in frames.items()}
    return {k: seed_mean(label, k, f.id.values) for k, f in frames.items()}


def dev_row(frames, scores):
    r = {}
    for k in ("ambe", "stress"):
        y = frames[k].label.values
        r[f"{k}_auroc"] = float(roc_auc_score(y, scores[k]))
        r[f"{k}_ap"] = float(average_precision_score(y, scores[k]))
    r["dev_mean_auroc"] = (r["ambe_auroc"] + r["stress_auroc"]) / 2
    return r


def dev_table(bundle=None, with_descriptors=True):
    """One row per model: AUROC and AP in Ambe and in the non-herbal set (up to 500 Da) and their mean AUROC."""
    bundle = bundle or load_dev()
    frames = dev_frames(bundle)
    rows = [dict(variant=lab, kind="single", **dev_row(frames, dev_scores(frames, lab))) for lab in TRAINED]
    if with_descriptors:
        rows += [dict(variant=lab, kind="descriptor", **dev_row(frames, dev_scores(frames, lab))) for lab in ("Size", "LogP")]
    return pd.DataFrame(rows)


def selection_table(bundle=None):
    bundle = bundle or load_dev()
    frames = dev_frames(bundle)
    t = dev_table(bundle, with_descriptors=False).drop(columns="kind").rename(columns={"variant": "model"})
    t.insert(1, "n_ambe", len(frames["ambe"]))
    t.insert(2, "n_ambe_inhibitors", int(frames["ambe"].label.sum()))
    t.insert(3, "n_nonherbal", len(frames["stress"]))
    t.insert(4, "n_nonherbal_inhibitors", int(frames["stress"].label.sum()))
    t = t.rename(columns={"stress_auroc": "nonherbal_auroc", "stress_ap": "nonherbal_ap"})
    t["rank"] = t.dev_mean_auroc.rank(ascending=False, method="min").astype(int)
    top = t.dev_mean_auroc.max()
    if (t.dev_mean_auroc == top).sum() != 1:
        raise SystemExit("the rule does not single out one model (tie in the mean development AUROC)")
    t["selected"] = t.dev_mean_auroc == top
    return t.sort_values("rank").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FINAL))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    check_frozen_split()
    files = [pred_file(lab, k, s) for lab in TRAINED for k in ("ambe", "stress") for s in SEEDS]
    check_fresh(files)
    t = selection_table()
    t.to_csv(out / "primary_model_selection.csv", index=False)
    chosen = t[t.selected].model.iloc[0]
    json.dump(dict(rule=RULE, selected=chosen, expected_primary=PRIMARY, candidates=TRAINED, development_sets=["ambe", "nonherbal"],
                   domain_da=DOMAIN, cohort_used=False, n_prediction_files=len(files)),
              open(out / "primary_model_selection.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    print("selected by the rule:", chosen)
    if chosen != PRIMARY:
        raise SystemExit(f"the rule selects {chosen}, but the analysis is written for {PRIMARY}")


if __name__ == "__main__":
    main()
