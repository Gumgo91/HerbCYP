"""Application outputs from a score vector: testing budgets, score tiers, ranked table of unmeasured ingredients.

Functions take plain arrays and data frames.  The tier edges (0.2, 0.5, 0.8) are constants; they are applied to the raw seed-mean Chemprop
probability and nothing is recalibrated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = [("Low", 0.0, 0.2), ("Moderate", 0.2, 0.5), ("High", 0.5, 0.8), ("Very high", 0.8, 1.0001)]


def tier_of(p):
    p = np.asarray(p, float)
    out = np.empty(len(p), object)
    for name, lo, hi in TIERS:
        out[(p >= lo) & (p < hi)] = name
    return out


def tier_table(y, p):
    """Inhibitor rate per tier (the 'what to do' table): n, inhibitors, rate."""
    t = tier_of(p)
    rows = []
    for name, lo, hi in TIERS:
        m = t == name
        rows.append(dict(tier=name, rule=f"score {lo:.1f} to {min(hi, 1.0):.1f}", n=int(m.sum()), inhibitors=int(np.asarray(y)[m].sum()),
                         rate=float(np.asarray(y)[m].mean()) if m.any() else np.nan))
    return pd.DataFrame(rows)


def budget_table(y, p, ids, budgets=(0.01, 0.05, 0.10, 0.20, 0.30)):
    """Assays spent, inhibitors found, assays per inhibitor, against random order (expected)."""
    y = np.asarray(y, int)
    order = np.lexsort((np.asarray(ids, str), -np.asarray(p, float)))
    n, pos = len(y), int(y.sum())
    rows = []
    for q in budgets:
        k = int(np.ceil(q * n))
        hits = int(y[order[:k]].sum())
        rows.append(dict(budget=q, assays=k, found=hits, recall=hits / pos, precision=hits / k, enrichment=(hits / k) / (pos / n),
                         assays_per_inhibitor=k / hits if hits else np.nan, random_assays_per_inhibitor=n / pos))
    return pd.DataFrame(rows)


def rank_table(frame, p, herb_names=None, top=None):
    """Ranked list of unmeasured ingredients: score, tier, size, similarity to the training compounds, herbs."""
    t = frame.copy()
    t["score"] = np.asarray(p, float)
    t["tier"] = tier_of(t.score)
    t = t.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    if herb_names is not None and "linked_herbs" in t:
        t["herbs"] = t.linked_herbs.fillna("").map(lambda s: "; ".join(herb_names.get(h, h) for h in str(s).split("|") if h)[:160])
    return t.head(top) if top else t
