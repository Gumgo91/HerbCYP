"""Recompute the headline numbers of the paper from the shipped predictions and labels and compare them with the shipped result tables.

Run from the repository root (needs python 3.9+, numpy, pandas and scikit-learn; about 10 seconds):

    python verify/verify_headline.py

Checks
  1. Cohort (1,028 HERB ingredients up to 500 Da, 171 inhibitors): AUROC, average precision and inhibitors in the top 10% of Chemprop, the
     MapLight-style CatBoost, TabPFN on descriptors, the random forest, heavy-atom count and Crippen logP, from the seed-mean predictions,
     against analysis_single/results/final/estimates.csv.
  2. Score tiers of Chemprop in the cohort (n and inhibitors per tier) against analysis_single/results/final/app/herb_tiers.csv, and testing budgets against
     analysis_single/results/final/app/herb_budget.csv.
  3. Selection of the primary model: mean AUROC in the Ambe set and the non-herbal set (both up to 500 Da) against
     analysis_single/results/final/primary_model_selection.csv.
  4. Ranked list of 15,511 unmeasured ingredients: the score of the ranked list equals the mean of the five seed
     files of Chemprop, and the tier counts.
The confidence intervals (scaffold-cluster bootstrap) are not recomputed here; the code is in analysis_single/code/run_all.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "arch_v4" / "bundle"                       # input tables
PRED = ROOT / "arch_v4" / "results"                        # predictions of the four models, <model>/seed<k>_<set>.csv.gz
RES = ROOT / "analysis_single" / "results" / "final"      # result tables of the paper
MODELS = {"Chemprop": "chemprop", "MapLight": "maplight_cb", "TabPFN-desc": "tabpfn_desc", "RF": "rf"}
SEEDS = (1, 2, 3, 4, 5)
failures = []


def check(name: str, got: float, want: float, tol: float) -> None:
    ok = abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:46s} recomputed {got:10.6f}   table {want:10.6f}")
    if not ok:
        failures.append(name)


def seed_mean(model: str, part: str) -> pd.Series:
    cols = [pd.read_csv(PRED / model / f"seed{s}_{part}.csv.gz").set_index("id")["p"] for s in SEEDS]
    return pd.concat(cols, axis=1).mean(axis=1)


def top_k_hits(score: pd.Series, label: pd.Series, tiebreak: pd.Series, k: int) -> int:
    order = pd.DataFrame({"s": score, "t": tiebreak, "y": label}).sort_values(["s", "t"], ascending=[False, True])
    return int(order["y"].iloc[:k].sum())


est = pd.read_csv(RES / "estimates.csv")


def table_value(analysis: str, model: str, metric: str) -> float:
    r = est[(est.analysis == analysis) & (est.model == model) & (est.metric == metric)]
    assert len(r) == 1, (analysis, model, metric, len(r))
    return float(r.estimate.iloc[0])


# ---------------------------------------------------------------- 1. cohort
herb = pd.read_csv(BUNDLE / "herb.csv").set_index("id")
herb = herb[herb["mw"] <= 500]                         # the file holds the 1,036 measured ingredients; the cohort is the 1,028 of at most 500 Da
scored = pd.read_csv(RES / "herb_scores_all_models.csv").set_index("id")       # carries the canonical SMILES used to break ties
y = herb["label"]
scores = {m: seed_mean(d, "herb").reindex(herb.index) for m, d in MODELS.items()}
scores["Size"] = herb["heavy"].astype(float)
scores["LogP"] = herb["logp"].astype(float)
n, n_pos = len(herb), int(y.sum())
k10 = math.ceil(0.10 * n)
print(f"Cohort: {n} ingredients, {n_pos} inhibitors, top 10% = {k10} assays")
for m, s in scores.items():
    check(f"{m} AUROC", roc_auc_score(y, s), table_value("herb", m, "auroc"), 1e-6)
    check(f"{m} average precision", average_precision_score(y, s), table_value("herb", m, "ap"), 2e-3 if m in ("RF", "Size", "LogP") else 1e-6)
    hits = top_k_hits(s, y, scored["canonical_smiles"].reindex(herb.index), k10)
    check(f"{m} inhibitors in the top 10%", hits, table_value("herb", m, "hits_10"), 2 if m in ("RF", "Size", "LogP") else 0)

# ---------------------------------------------------------------- 2. tiers and budgets
print("Score tiers of Chemprop in the cohort")
edges = [0.0, 0.2, 0.5, 0.8, 1.0000001]
tier = pd.cut(scores["Chemprop"], edges, right=False, labels=["Low", "Moderate", "High", "Very high"])
tiers = pd.read_csv(RES / "app" / "herb_tiers.csv").set_index("tier")
for t in ["Low", "Moderate", "High", "Very high"]:
    m = tier == t
    check(f"{t}: ingredients", int(m.sum()), float(tiers.loc[t, "n"]), 0)
    check(f"{t}: inhibitors", int(y[m].sum()), float(tiers.loc[t, "inhibitors"]), 0)
budget = pd.read_csv(RES / "app" / "herb_budget.csv")
for _, r in budget.iterrows():
    k = int(r["assays"])
    check(f"budget {r['budget']:.0%} ({k} assays): inhibitors found", top_k_hits(scores["Chemprop"], y, scored["canonical_smiles"].reindex(herb.index), k), float(r["found"]), 0)

# ---------------------------------------------------------------- 3. selection of the primary model
print("Selection of the primary model (mean AUROC in the Ambe set and the non-herbal set, both up to 500 Da)")
ambe = pd.read_csv(BUNDLE / "ambe.csv").set_index("id")
stress = pd.read_csv(BUNDLE / "stress.csv").set_index("id")
ambe, stress = ambe[ambe.mw <= 500], stress[stress.mw <= 500]
sel = pd.read_csv(RES / "primary_model_selection.csv").set_index("model")
means = {}
for m, d in MODELS.items():
    a = roc_auc_score(ambe["label"], seed_mean(d, "ambe").reindex(ambe.index))
    s = roc_auc_score(stress["label"], seed_mean(d, "stress").reindex(stress.index))
    means[m] = (a + s) / 2
    check(f"{m} mean development AUROC", means[m], float(sel.loc[m, "dev_mean_auroc"]), 1e-6)
print(f"  selected by the rule: {max(means, key=means.get)}")
if max(means, key=means.get) != "Chemprop":
    failures.append("selected model")

# ---------------------------------------------------------------- 4. ranked list
print("Ranked list of unmeasured ingredients")
s7 = pd.read_csv(RES / "app" / "unmeasured_ranked.csv.gz")
cp = seed_mean("chemprop", "unmeasured").reindex(s7["id"])
check("number of ranked ingredients", len(s7), 15511, 0)
check("largest difference S7 score vs seed mean", float(np.abs(cp.to_numpy() - s7["score"].to_numpy()).max()), 0.0, 1e-9)
counts = s7["tier"].value_counts()
for t, want in (("Low", 9869), ("Moderate", 3247), ("High", 1872), ("Very high", 523)):
    check(f"{t} tier: ingredients", int(counts.get(t, 0)), want, 0)

print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("All checks passed.")
