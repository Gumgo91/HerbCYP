"""Application outputs with Chemprop: testing budgets, risk tiers, ranked list of unmeasured ingredients, herb shortlists.

python code/app_final.py [--out results/final/app]

Score = mean over the five seeds of the raw Chemprop probability.  The tier edges (0.2, 0.5, 0.8) are fixed (app_outputs.py); the HERB cohort (at most
500 Da) is used only for the observed inhibitor rate per tier and the budget table, as in the second analysis.  The ranked list contains the
ingredients of at most 500 Da that have no concordant measurement and no chemical parent in the training pool (15,511 after 109 are set aside as
in-sample).  Two columns are new relative to the second analysis' ranked table: parent_in_evaluation_sets (the ingredient shares a chemical parent
with an ingredient of the evaluation cohort or of the Ambe set) and the score is the Chemprop probability.  The similarity (maxsim_train) and the
applicability class are recomputed from the training rows (union of the five training sets, similarity.py).  From analysis_v3/results only the
split-independent annotations are read (chemotype, methylenedioxyphenyl, sugar, methoxy count; herb names).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from app_outputs import TIERS, budget_table, rank_table, tier_of, tier_table
from common import APP, DOMAIN, PRIMARY, ROOT, V3_RESULTS, check_fresh, check_frozen_split, load_bundle, pred_file, seed_mean, unmeasured_domain, SEEDS
from similarity import maxsim_union
from herbcyp.statistics import cluster_indices  # noqa: E402  (ROOT/src is on the path after common is imported)

N_BOOT, BOOT_SEED = 2000, 2026


def clean_name(s):
    s = "" if pd.isna(s) else str(s)
    s = re.sub("[� ]+", " ", s)
    s = re.sub(r"[^\x20-\x7e]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def herb_names():
    d = pd.read_csv(V3_RESULTS / "a5_herb_coverage.csv")
    return dict(zip(d.herb_id, d.herb_name))


def tier_with_ci(frame, score, group_col="scaffold"):
    """Inhibitor rate per tier with scaffold-cluster bootstrap intervals (same draws as the main analysis)."""
    y = frame.label.values.astype(int)
    t = tier_of(score)
    base = tier_table(y, score)
    draws = list(cluster_indices(frame[group_col], N_BOOT, BOOT_SEED))
    lo, hi = [], []
    for name, _, _ in TIERS:
        m = t == name
        r = []
        for idx in draws:
            k = m[idx]
            if k.sum() >= 5:
                r.append(y[idx][k].mean())
        lo.append(np.percentile(r, 2.5) if len(r) > 100 else np.nan)
        hi.append(np.percentile(r, 97.5) if len(r) > 100 else np.nan)
    base["rate_lo"], base["rate_hi"] = lo, hi
    base["share_of_inhibitors"] = base.inhibitors / max(y.sum(), 1)
    base["share_of_ingredients"] = base.n / len(y)
    return base


def recovery_curve(frame, score, ids):
    y = frame.label.values.astype(int)
    order = np.lexsort((np.asarray(ids, str), -np.asarray(score, float)))
    return pd.DataFrame({"k": np.arange(1, len(y) + 1), "cum_hits": np.cumsum(y[order])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(APP))
    a = ap.parse_args()
    check_frozen_split()
    check_fresh([pred_file(PRIMARY, k, s) for k in ("herb", "ambe", "stress", "unmeasured") for s in SEEDS])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = herb_names()
    bundle = load_bundle()

    # ---------------- HERB cohort: budgets and tiers ----------------
    herb = bundle["herb"].copy()
    herb = herb[herb.mw <= DOMAIN].reset_index(drop=True)
    herb["score"] = seed_mean(PRIMARY, "herb", herb.id.values)
    bt = budget_table(herb.label.values, herb.score.values, herb.smiles.values, budgets=(0.01, 0.05, 0.10, 0.20, 0.30))
    bt.to_csv(out / "herb_budget.csv", index=False)
    tt = tier_with_ci(herb, herb.score.values)
    tt.to_csv(out / "herb_tiers.csv", index=False)
    rc = recovery_curve(herb, herb.score.values, herb.smiles.values)
    rc.to_csv(out / "herb_recovery_curve.csv", index=False)
    herb[["id", "smiles", "name", "label", "score", "mw", "heavy", "scaffold"]].assign(tier=tier_of(herb.score.values)).to_csv(
        out / "herb_cohort_scored.csv", index=False)
    print("HERB cohort, budgets")
    print(bt.round(3).to_string(index=False))
    print("HERB cohort, tiers")
    print(tt.round(3).to_string(index=False))

    # the same tiers in the two development sets (up to 500 Da), for the consistency of the tier definitions
    for k in ("ambe", "stress"):
        d = bundle[k].copy()
        d = d[d.mw <= DOMAIN].reset_index(drop=True)
        d["score"] = seed_mean(PRIMARY, k, d.id.values)
        t_k = tier_with_ci(d, d.score.values)
        t_k.to_csv(out / f"{k}_tiers.csv", index=False)
        print(k)
        print(t_k.round(3).to_string(index=False))

    # ---------------- unmeasured ingredients: ranked list ----------------
    unm = unmeasured_domain(bundle["unmeasured"])
    unm["score"] = seed_mean(PRIMARY, "unmeasured", unm.id.values)
    unm["maxsim_train"] = maxsim_union(["unmeasured"], bundle)["unmeasured"]
    # structures whose chemical parent is in the training library are in-sample: kept apart from the ranked list
    pool = bundle["pool"]
    in_lib = unm.parent_inchikey.isin(set(pool.parent_inchikey))
    lab = pool.groupby("parent_inchikey").y.max()
    ex = unm[in_lib].copy()
    ex["training_label"] = ex.parent_inchikey.map(lab).values
    ex[["id", "name", "smiles", "mw", "score", "training_label", "parent_inchikey"]].sort_values("score", ascending=False).to_csv(
        out / "unmeasured_in_training_library.csv", index=False)
    print(f"{int(in_lib.sum())} unmeasured structures share a parent with the training library and are kept out of the ranked list "
          f"(training inhibitors among them: {int(ex.training_label.sum())})")
    unm = unm[~in_lib].reset_index(drop=True)
    ann = pd.read_csv(V3_RESULTS / "a9_unmeasured_annotated.csv", usecols=["canonical_smiles", "chemotype", "mdp", "has_sugar", "n_ome"])
    ann = ann.drop_duplicates("canonical_smiles").set_index("canonical_smiles")
    for c in ["chemotype", "mdp", "has_sugar", "n_ome"]:
        unm[c] = ann.reindex(unm.smiles)[c].values
    assert unm.maxsim_train.notna().all() and unm.chemotype.notna().all()
    unm["name"] = unm.name.map(clean_name)
    # HERB entries whose name and structure disagree (found while inspecting the top of the ranked list in the first analysis)
    unm["name_flag"] = unm["name"].str.lower().isin(["acetic acid", "8-hydroxyluteolin-8--d-glucopranoside"]).map(
        {True: "name does not match structure", False: ""})
    unm["n_herbs"] = unm.linked_herbs.fillna("").map(lambda s: len([h for h in str(s).split("|") if h]))
    # ingredients that share a chemical parent with an evaluation ingredient (cohort or Ambe): other forms of measured compounds
    herb_all, ambe_all = bundle["herb"], bundle["ambe"]
    in_cohort = unm.parent_inchikey.isin(set(herb_all.parent_inchikey))
    if not in_cohort.equals(unm.parent_inchikey.isin(set(herb_all[herb_all.mw <= DOMAIN].parent_inchikey))):
        raise SystemExit("shared parents with the cohort differ between the 1,036 and the 1,028 ingredients")
    in_ambe = unm.parent_inchikey.isin(set(ambe_all.parent_inchikey))
    unm["parent_in_evaluation_sets"] = in_cohort | in_ambe
    print(f"ranked ingredients that share a parent with an evaluation ingredient: {int(unm.parent_in_evaluation_sets.sum())} "
          f"(cohort {int(in_cohort.sum())}, Ambe {int(in_ambe.sum())}, both {int((in_cohort & in_ambe).sum())})")
    ranked = rank_table(unm, unm.score.values, herb_names=names)
    ranked["applicability"] = np.where(ranked.maxsim_train >= 0.6, "close analogs in training",
                                       np.where(ranked.maxsim_train >= 0.4, "moderate analogs", "remote from training"))
    keep = ["rank", "id", "name", "name_flag", "smiles", "mw", "heavy", "logp", "np", "chemotype", "score", "tier", "maxsim_train", "applicability",
            "n_herbs", "herbs", "linked_herbs", "parent_inchikey", "parent_in_evaluation_sets"]
    ranked[[c for c in keep if c in ranked]].to_csv(out / "unmeasured_ranked.csv.gz", index=False)
    print("ranked", len(ranked), ranked.tier.value_counts().reindex([t[0] for t in TIERS]).to_dict())

    # tier x chemotype, tier x molecular weight, tier x applicability
    ranked["mw_bin"] = pd.cut(ranked.mw, [0, 200, 300, 400, 500], labels=["<=200", "200-300", "300-400", "400-500"])
    pd.crosstab(ranked.chemotype, ranked.tier).reindex(columns=[t[0] for t in TIERS]).assign(total=lambda d: d.sum(axis=1)).sort_values(
        "total", ascending=False).to_csv(out / "unmeasured_tier_by_chemotype.csv")
    pd.crosstab(ranked.mw_bin, ranked.tier).reindex(columns=[t[0] for t in TIERS]).to_csv(out / "unmeasured_tier_by_mw.csv")
    pd.crosstab(ranked.applicability, ranked.tier).reindex(columns=[t[0] for t in TIERS]).to_csv(out / "unmeasured_tier_by_similarity.csv")

    # ---------------- herb view ----------------
    rows = []
    herb_ids = set()
    for s in pd.concat([ranked.linked_herbs, herb.linked_herbs]).dropna():
        herb_ids.update(h for h in str(s).split("|") if h)
    meas = {h: [] for h in herb_ids}
    for _, r in herb.iterrows():
        for h in str(r.linked_herbs).split("|"):
            if h in meas:
                meas[h].append((r.label, r.score))
    cand = {h: [] for h in herb_ids}
    for _, r in ranked.iterrows():
        for h in str(r.linked_herbs).split("|"):
            if h in cand:
                cand[h].append((r["rank"], r["name"], r.score, r.tier))
    for h in sorted(herb_ids):
        m, c = meas[h], cand[h]
        rows.append(dict(herb_id=h, herb=names.get(h, h), n_measured=len(m), n_measured_inhibitors=int(sum(x[0] for x in m)),
                         n_unmeasured=len(c), n_high=sum(x[3] in ("High", "Very high") for x in c),
                         n_very_high=sum(x[3] == "Very high" for x in c),
                         top_candidates="; ".join(f"{x[1]} ({x[2]:.2f})" for x in sorted(c)[:5])))
    hv = pd.DataFrame(rows).sort_values(["n_high", "n_unmeasured"], ascending=False)
    hv["frac_high"] = hv.n_high / hv.n_unmeasured.clip(lower=1)
    hv.to_csv(out / "herb_view.csv", index=False)
    print("herbs", len(hv), "with >=1 High/Very high untested ingredient:", int((hv.n_high > 0).sum()))


if __name__ == "__main__":
    main()
