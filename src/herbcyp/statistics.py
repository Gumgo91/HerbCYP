"""Paired cluster inference for the fixed HerbCYP evaluation cohort.

Bootstrap sampling concatenates every sampled cluster instance, preserving
its multiplicity. All models and paired contrasts share the same draws.
These intervals are conditional on the saved fitted-model predictions;
they do not include model-refitting or cohort-ascertainment uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

METRICS = ("auroc", "auprc", "recall_10", "recall_20",
           "precision_10", "precision_20", "ef_10", "ef_20")


def align_predictions(cohort, predictions, id_col="canonical_smiles"):
    """One-to-one ID join, rejecting missing, extra, duplicated, or wrong labels."""
    for name, frame in (("cohort", cohort), ("predictions", predictions)):
        if id_col not in frame or frame[id_col].isna().any():
            raise ValueError(f"{name}: missing molecule IDs")
        if frame[id_col].duplicated().any():
            raise ValueError(f"{name}: duplicated molecule IDs")
    want, got = set(cohort[id_col]), set(predictions[id_col])
    if want != got:
        raise ValueError(f"prediction ID mismatch: missing={len(want-got)}, "
                         f"extra={len(got-want)}")
    if "label" not in cohort or not cohort.label.isin([0, 1]).all():
        raise ValueError("cohort labels must be observed binary labels")
    p = predictions.set_index(id_col).loc[cohort[id_col]].reset_index()
    if "label" in p and not np.array_equal(cohort.label.to_numpy(),
                                           p.label.to_numpy()):
        raise ValueError("prediction labels disagree with cohort after ID join")
    if "y_true" in p and not np.array_equal(cohort.label.to_numpy(),
                                            p.y_true.to_numpy()):
        raise ValueError("prediction y_true disagrees with cohort after ID join")
    out = cohort.reset_index(drop=True).copy()
    for col in p.columns:
        if col not in out:
            out[col] = p[col].to_numpy()
    return out


def cluster_indices(groups, n_boot=2000, seed=2026):
    """Yield row indices including repeated instances of a sampled cluster."""
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    by_group = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique), size=len(unique))
        yield np.concatenate([by_group[k] for k in chosen])


def metric_values(y, scores, molecule_ids):
    """AP handles tied scores by whole thresholds; budgets use label-free ties."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    ids = np.asarray(molecule_ids, dtype=str)
    n, positives = len(y), int(y.sum())
    if n == 0:
        return np.full(len(METRICS), np.nan)
    negatives = n - positives
    auroc = (float((rankdata(scores)[y == 1].sum()
                   - positives * (positives + 1) / 2)
                  / (positives * negatives))
             if positives and negatives else np.nan)
    # Average precision, with each tied score treated as one threshold.
    order = np.argsort(-scores, kind="stable")
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), n - 1]
    tp = np.cumsum(ys)[ends]
    ap = (float(np.sum(np.diff(np.r_[0, tp]) / positives
                       * tp / (ends + 1))) if positives and negatives else np.nan)
    rank = np.lexsort((ids, -scores))
    recalls, precisions, enrichments = [], [], []
    for budget in (0.10, 0.20):
        k = int(np.ceil(budget * n))
        recovered = int(y[rank[:k]].sum())
        precision = recovered / k
        recalls.append(recovered / positives if positives else np.nan)
        precisions.append(precision)
        enrichments.append(precision / (positives / n) if positives else np.nan)
    return np.asarray([auroc, ap, *recalls, *precisions, *enrichments])


def paired_cluster_analysis(frame, score_columns, group_col="scaffold",
                            n_boot=2000, seed=2026, primary="M1_chemprop"):
    """Model CIs and paired M1-minus-baseline CIs from identical draws.

    Single-class draws are skipped for *all* models/metrics so paired
    inference remains aligned. Their number is returned explicitly.
    """
    required = {"canonical_smiles", "label", group_col, *score_columns.values()}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required-set(frame.columns))}")
    if frame[group_col].isna().any():
        raise ValueError("missing bootstrap cluster IDs")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("analysis labels must be binary and observed")
    models = list(score_columns)
    ids, y = frame.canonical_smiles.to_numpy(), frame.label.to_numpy(int)
    scores = [frame[col].to_numpy(float) for col in score_columns.values()]
    if any(not np.isfinite(s).all() for s in scores):
        raise ValueError("non-finite score; refusing different model masks")
    if any(((s < 0) | (s > 1)).any() for s in scores):
        raise ValueError("prediction probabilities outside [0,1]")
    points = np.stack([metric_values(y, s, ids) for s in scores])
    draws = []
    if len(frame):
        for idx in cluster_indices(frame[group_col], n_boot, seed):
            if np.unique(y[idx]).size < 2:
                continue
            draws.append(np.stack([metric_values(y[idx], s[idx], ids[idx])
                                   for s in scores]))
    boot = np.stack(draws) if draws else np.empty((0, len(models), len(METRICS)))
    n_valid = len(boot)
    cluster_sizes = frame.groupby(group_col, dropna=False).size()
    common = {"n_eval": len(frame), "n_pos": int(y.sum()),
              "n_neg": int(len(y)-y.sum()), "prevalence": float(y.mean())
              if len(y) else np.nan, "n_clusters": int(len(cluster_sizes)),
              "largest_cluster_n": int(cluster_sizes.max()) if len(frame) else 0,
              "n_boot_requested": n_boot, "n_boot_valid": n_valid,
              "n_boot_single_class": n_boot-n_valid if len(frame) else 0,
              "cluster_unit": group_col, "seed": seed,
              "support_note": "sparse: n<30, clusters<10, or either class<15; descriptive only"
              if len(frame) < 30 or len(cluster_sizes) < 10
              or int(y.sum()) < 15 or int(len(y)-y.sum()) < 15 else "",
              "inference_scope": "conditional on fitted models and observed cohort"}
    rows = []
    for j, name in enumerate(models):
        for k, metric in enumerate(METRICS):
            ci = np.percentile(boot[:, j, k], [2.5, 97.5]) if n_valid >= 50 and len(cluster_sizes) >= 2 \
                else [np.nan, np.nan]
            rows.append({**common, "model": name, "metric": metric,
                         "estimate": points[j, k], "ci_lo": ci[0], "ci_hi": ci[1]})
    deltas = []
    if primary in models:
        j = models.index(primary)
        for baseline in ("B1_rf", "B2_knn"):
            if baseline not in models:
                continue
            b = models.index(baseline)
            for k, metric in enumerate(METRICS):
                delta_boot = boot[:, j, k] - boot[:, b, k]
                ci = np.percentile(delta_boot, [2.5, 97.5]) if n_valid >= 50 and len(cluster_sizes) >= 2 \
                    else [np.nan, np.nan]
                deltas.append({**common, "candidate": primary,
                               "reference": baseline, "metric": metric,
                               "estimate": points[j, k]-points[b, k],
                               "ci_lo": ci[0], "ci_hi": ci[1],
                               "comparison": "candidate minus reference"})
    return pd.DataFrame(rows), pd.DataFrame(deltas)
