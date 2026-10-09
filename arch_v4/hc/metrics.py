"""Metrics used in development and in the final comparison."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def topk_hits(y, s, frac=0.10):
    y, s = np.asarray(y), np.asarray(s)
    k = max(1, int(round(frac * len(y))))
    order = np.argsort(-s, kind="mergesort")
    return int(y[order[:k]].sum()), k


def summarize(y, s):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    hits, k = topk_hits(y, s)
    prev = y.mean()
    return dict(n=len(y), pos=int(y.sum()), auroc=float(roc_auc_score(y, s)), ap=float(average_precision_score(y, s)),
                hits10=hits, k10=k, enrich10=float(hits / k / prev))
