"""Evaluation tables: HERB cohort, development sets and development table (summarizes the prediction files of the four models; nothing is tuned).

python code/run_all.py [--out results/final] [--skip herb dev]

Primary model: Chemprop (rule in select_primary.py, development data only).  Comparators: MapLight-style CatBoost (MapLight), TabPFN on descriptors
(TabPFN-desc) and random forest (RF); reference scores: heavy-atom count (Size) and Crippen logP (LogP).  Scores are the means over the five seeds of the raw probabilities.
The paired differences are Chemprop minus the comparator or reference score (2,000 scaffold-cluster draws, seed 2026, ring-free structures form one cluster).
Cohort labels enter only through the evaluation functions of stats_v3.py (estimates, intervals, stratified AUROC) and the recovery curves.
Every score is the mean of the five seed models (an ensemble), so every AUROC is the AUROC of a five-seed mean; the seed-to-seed variation of single seeds is not summarized here.

Writes estimates.csv, paired.csv, herb_recovery_curves.csv, herb_scores_all_models.csv and dev_single_models.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ALL_MODELS, DOMAIN, FINAL, PRIMARY, SUBSET_MODELS, attach_scores, load_bundle, panel, verify_inputs)
from select_primary import dev_table
from similarity import maxsim_union


def load_frame(bundle, name, sim):
    return attach_scores(bundle[name], name, sim=sim[name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FINAL))
    ap.add_argument("--skip", nargs="*", default=[], help="analysis groups to skip: herb, dev")
    args = ap.parse_args()
    verify_inputs(include_herb="herb" not in args.skip)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    sim = maxsim_union(["ambe", "stress", "herb"], bundle)
    est_all, pair_all = [], []

    # ------------------------------------------------ HERB cohort, final evaluation
    if "herb" not in args.skip:
        herb = load_frame(bundle, "herb", sim)
        herb = herb[herb.mw <= DOMAIN].reset_index(drop=True)
        print(f"HERB cohort at most 500 Da: n={len(herb)}, inhibitors={int(herb.label.sum())}")
        panel(herb, "herb", est_all, pair_all, ALL_MODELS, strata=True)
        panel(herb[herb.maxsim < 0.4], "herb_sim_lt0.4", est_all, pair_all, SUBSET_MODELS)
        panel(herb[(herb.maxsim >= 0.4) & (herb.maxsim < 0.6)], "herb_sim_0.4_0.6", est_all, pair_all, SUBSET_MODELS)
        panel(herb[herb.maxsim >= 0.6], "herb_sim_ge0.6", est_all, pair_all, SUBSET_MODELS)
        panel(herb[herb.mw <= 200], "herb_mw_le200", est_all, pair_all, SUBSET_MODELS)
        panel(herb[(herb.mw > 200) & (herb.mw <= 300)], "herb_mw_200_300", est_all, pair_all, SUBSET_MODELS)
        panel(herb[herb.mw > 300], "herb_mw_300_500", est_all, pair_all, SUBSET_MODELS)
        panel(herb[herb.np >= 0], "herb_np_ge0", est_all, pair_all, SUBSET_MODELS)
        panel(herb[herb.np < 0], "herb_np_lt0", est_all, pair_all, SUBSET_MODELS)
        # recovery curves of all scores
        rows = []
        for m in ALL_MODELS:
            rank = np.lexsort((herb.smiles.to_numpy(str), -herb[m].to_numpy(float)))
            cum = np.cumsum(herb.label.to_numpy(int)[rank])
            rows += [(m, k + 1, int(cum[k])) for k in range(len(herb))]
        pd.DataFrame(rows, columns=["model", "k", "cum_hits"]).to_csv(out / "herb_recovery_curves.csv", index=False)
        herb.to_csv(out / "herb_scores_all_models.csv", index=False)

    # ------------------------------------------------ development sets (Ambe, non-herbal), reported as such
    if "dev" not in args.skip:
        amb = load_frame(bundle, "ambe", sim)
        st = load_frame(bundle, "stress", sim)
        a5 = amb[amb.mw <= DOMAIN].reset_index(drop=True)
        s5 = st[st.mw <= DOMAIN].reset_index(drop=True)
        panel(a5, "ambe", est_all, pair_all, ALL_MODELS, strata=True)
        panel(a5[a5.herb_linked.astype(str).str.lower().isin(["true", "1"])], "ambe_herb_linked", est_all, pair_all, SUBSET_MODELS)
        panel(s5, "stress_le500", est_all, pair_all, ALL_MODELS)
        panel(s5[s5.np >= 0], "stress_le500_np_ge0", est_all, pair_all, SUBSET_MODELS)
        panel(s5[s5.mw < 300], "stress_mw_lt300", est_all, pair_all, SUBSET_MODELS)
        panel(s5[(s5.mw >= 300) & (s5.mw <= 400)], "stress_mw_300_400", est_all, pair_all, SUBSET_MODELS)
        panel(s5[(s5.mw > 400)], "stress_mw_400_500", est_all, pair_all, SUBSET_MODELS)
        # above the application domain (more than 500 Da): reported separately for the non-herbal set
        g5 = st[st.mw > DOMAIN].reset_index(drop=True)
        panel(g5, "stress_gt500", est_all, pair_all, ALL_MODELS)
        panel(g5[g5.np >= 0], "stress_gt500_np_ge0", est_all, pair_all, SUBSET_MODELS)
        panel(st[st.mw > 700].reset_index(drop=True), "stress_gt700", est_all, pair_all, SUBSET_MODELS)

    if est_all:
        pd.concat(est_all, ignore_index=True).to_csv(out / "estimates.csv", index=False)
        pd.concat(pair_all, ignore_index=True).to_csv(out / "paired.csv", index=False)

    # ------------------------------------------------ development table
    if "dev" not in args.skip:
        t = dev_table(bundle)
        t.to_csv(out / "dev_single_models.csv", index=False)
        pd.set_option("display.width", 200)
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
