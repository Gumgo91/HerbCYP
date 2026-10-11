"""Post hoc sensitivity analyses in subsets of the HERB cohort: ingredients without a
methylenedioxyphenyl group, ingredients also present in the Ambe set, ingredients whose chemical parent is in the label-free pretraining corpus,
ring-containing and ring-free ingredients, ingredients with a ring system.  Chemprop is the primary model.

python code/extra_herb.py [--final results/final] [--out results/final]

Reads herb_scores_all_models.csv of run_all.py from --final and writes estimates_extra.csv and paired_extra.csv to --out.
Only the split-independent columns (chemotype, methylenedioxyphenyl) of analysis_v3/results/a4_main_chemotypes.csv are read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ALL_MODELS, BUNDLE, FINAL, PRIMARY, V3_RESULTS, bootstrap_panel, panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=str(FINAL))
    ap.add_argument("--out", default=None, help="default: same folder as --final")
    a = ap.parse_args()
    final, out = Path(a.final), Path(a.out or a.final)
    out.mkdir(parents=True, exist_ok=True)
    herb = pd.read_csv(final / "herb_scores_all_models.csv")
    ann = pd.read_csv(V3_RESULTS / "a4_main_chemotypes.csv", usecols=["canonical_smiles", "mdp", "chemotype"]
                      ).drop_duplicates("canonical_smiles").set_index("canonical_smiles")
    herb["mdp"] = ann.reindex(herb.smiles)["mdp"].values
    herb["chemotype"] = ann.reindex(herb.smiles)["chemotype"].values
    print("MDP in cohort:", int(herb.mdp.fillna(False).sum()), "with", int(herb[herb.mdp == True].label.sum()), "inhibitors")
    models = ALL_MODELS
    est_all, pair_all = [], []
    sub = herb[~(herb.mdp == True)].reset_index(drop=True)
    panel(sub, "herb_no_mdp", est_all, pair_all, models)
    panel(herb[herb.chemotype.notna() & ~herb.chemotype.isin(["Acyclic aliphatic", "Other"])].reset_index(drop=True), "herb_ring_systems", est_all, pair_all, models)
    # ingredients that also occur in the Ambe development set (their labels partly come from Ambe et al.)
    ambe = pd.read_csv(BUNDLE / "ambe.csv")
    ambe = ambe[ambe.mw <= 500]
    herb["parent14"] = herb.parent_inchikey.astype(str).str[:14]
    ov = herb.parent14.isin(set(ambe.parent_inchikey.astype(str).str[:14]))
    print("cohort ingredients also in the Ambe set:", int(ov.sum()), "inhibitors", int(herb[ov].label.sum()))
    panel(herb[~ov].reset_index(drop=True), "herb_no_ambe_overlap", est_all, pair_all, models)
    # ingredients whose chemical parent also occurs in the label-free pretraining corpus (unmeasured HERB structures without a parent in the pool)
    pool = pd.read_csv(BUNDLE / "pool.csv")
    unm = pd.read_csv(BUNDLE / "unmeasured.csv")
    corpus_parents = set(unm.parent_inchikey) - set(pool.parent_inchikey)
    pc = herb.parent_inchikey.isin(corpus_parents)
    print("cohort ingredients whose parent is in the pretraining corpus:", int(pc.sum()), "inhibitors", int(herb[pc].label.sum()))
    panel(herb[~pc].reset_index(drop=True), "herb_no_pretrain_overlap", est_all, pair_all, models)
    panel(herb[~pc & ~ov].reset_index(drop=True), "herb_no_pretrain_no_ambe_overlap", est_all, pair_all, models)
    # ring-free ingredients share the scaffold label '__ACYCLIC__' and form one cluster in the scaffold-cluster bootstrap of the main analysis;
    # for them chemical parents are resampled instead of scaffold clusters.
    cyc = herb.scaffold.ne("__ACYCLIC__")
    panel(herb[cyc].reset_index(drop=True), "herb_ring_containing", est_all, pair_all, models)
    acy = herb[~cyc].reset_index(drop=True)
    est_a, pair_a = bootstrap_panel(acy.assign(scaffold=acy.parent_inchikey), {m: m for m in models}, group_col="scaffold", primary=PRIMARY)
    est_a.insert(0, "analysis", "herb_ring_free_by_parent")
    pair_a.insert(0, "analysis", "herb_ring_free_by_parent")
    est_all.append(est_a)
    pair_all.append(pair_a)
    pd.concat(est_all).to_csv(out / "estimates_extra.csv", index=False)
    pd.concat(pair_all).to_csv(out / "paired_extra.csv", index=False)
    e = pd.concat(est_all)
    for an in ("herb_no_mdp", "herb_ring_systems", "herb_no_ambe_overlap", "herb_no_pretrain_overlap", "herb_no_pretrain_no_ambe_overlap",
               "herb_ring_containing", "herb_ring_free_by_parent"):
        x = e[(e.analysis == an) & (e.metric == "auroc")][["model", "estimate", "ci_lo", "ci_hi", "n", "n_pos"]]
        print(an)
        print(x.round(3).to_string(index=False))
    p = pd.concat(pair_all)
    print(p[(p.metric == "auroc")][["analysis", "reference", "estimate", "ci_lo", "ci_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
