"""Ranking metrics and paired scaffold-cluster bootstrap.

The resampling uses herbcyp.statistics.cluster_indices, so all models and paired contrasts share the identical draws.
Ties: descending score, then ascending canonical SMILES; labels never break ties.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from herbcyp.statistics import cluster_indices  # noqa: E402

BUDGETS = (0.01, 0.05, 0.10, 0.20)


def metric_names(budgets=BUDGETS):
    names = ["auroc", "ap"]
    for q in budgets:
        p = int(round(q * 100))
        names += [f"recall_{p}", f"precision_{p}", f"ef_{p}", f"hits_{p}", f"k_{p}"]
    return names


def metric_dict(y, scores, ids, budgets=BUDGETS):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    ids = np.asarray(ids, dtype=str)
    n, pos = len(y), int(y.sum())
    neg = n - pos
    out = {}
    if pos == 0 or neg == 0:
        return {k: np.nan for k in metric_names(budgets)}
    out["auroc"] = float((rankdata(scores)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    order = np.argsort(-scores, kind="stable")
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), n - 1]
    tp = np.cumsum(ys)[ends]
    out["ap"] = float(np.sum(np.diff(np.r_[0, tp]) / pos * tp / (ends + 1)))
    rank = np.lexsort((ids, -scores))
    for q in budgets:
        p = int(round(q * 100))
        k = int(np.ceil(q * n))
        hits = int(y[rank[:k]].sum())
        prec = hits / k
        out[f"recall_{p}"] = hits / pos
        out[f"precision_{p}"] = prec
        out[f"ef_{p}"] = prec / (pos / n)
        out[f"hits_{p}"] = float(hits)
        out[f"k_{p}"] = float(k)
    return out


def stratified_auroc(y, scores, strata):
    """Pooled within-stratum concordance: sum of pos>neg pairs over sum of pos*neg pairs, strata with both classes."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    strata = np.asarray(strata)
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        yy, ss = y[m], scores[m]
        p, q = int(yy.sum()), int((1 - yy).sum())
        if p == 0 or q == 0:
            continue
        u = rankdata(ss)[yy == 1].sum() - p * (p + 1) / 2
        num += u
        den += p * q
    return float(num / den) if den else np.nan


def bootstrap_panel(frame, score_cols, group_col="scaffold", n_boot=2000, seed=2026,
                    primary="M1", budgets=BUDGETS, id_col="canonical_smiles", label_col="label",
                    strata_col=None, min_valid=50):
    """Return (estimates_long, paired_long). score_cols: {model_name: column}."""
    frame = frame.reset_index(drop=True)
    y = frame[label_col].to_numpy(int)
    ids = frame[id_col].to_numpy(str)
    models = list(score_cols)
    scores = {m: frame[c].to_numpy(float) for m, c in score_cols.items()}
    strata = frame[strata_col].to_numpy() if strata_col else None
    names = metric_names(budgets) + (["auroc_strat"] if strata is not None else [])

    def panel(idx):
        rows = {}
        for m in models:
            d = metric_dict(y[idx], scores[m][idx], ids[idx], budgets)
            if strata is not None:
                d["auroc_strat"] = stratified_auroc(y[idx], scores[m][idx], strata[idx])
            rows[m] = d
        return rows

    full = panel(np.arange(len(frame)))
    draws = []
    n_groups = frame[group_col].nunique()
    for idx in cluster_indices(frame[group_col], n_boot, seed):
        if np.unique(y[idx]).size < 2:
            continue
        draws.append(panel(idx))
    n_valid = len(draws)
    common = dict(n=len(frame), n_pos=int(y.sum()), prevalence=float(y.mean()), n_clusters=int(n_groups),
                  n_boot_valid=n_valid, n_boot_requested=n_boot)
    est_rows, pair_rows = [], []
    for m in models:
        for name in names:
            arr = np.array([d[m][name] for d in draws], dtype=float)
            ok = n_valid >= min_valid and n_groups >= 2
            lo, hi = (np.nanpercentile(arr, [2.5, 97.5]) if ok else (np.nan, np.nan))
            est_rows.append(dict(model=m, metric=name, estimate=full[m][name], ci_lo=lo, ci_hi=hi, **common))
    if primary in models:
        for m in models:
            if m == primary:
                continue
            for name in names:
                if name.startswith(("hits_", "k_")):
                    continue
                diff = np.array([d[primary][name] - d[m][name] for d in draws], dtype=float)
                ok = n_valid >= min_valid and n_groups >= 2
                lo, hi = (np.nanpercentile(diff, [2.5, 97.5]) if ok else (np.nan, np.nan))
                pair_rows.append(dict(candidate=primary, reference=m, metric=name,
                                      estimate=full[primary][name] - full[m][name],
                                      ci_lo=lo, ci_hi=hi, **common))
    return pd.DataFrame(est_rows), pd.DataFrame(pair_rows)
