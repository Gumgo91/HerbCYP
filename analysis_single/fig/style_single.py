"""Shared style of the figure scripts (Chemprop is the primary model).

Every figure script reads the result tables in ROOT/analysis_single/results/final (and, for the data figure, the split-independent tables of
the bundle) and writes to the figure folder FIGDIR: PDF, EPS, PNG (300 dpi) and TIFF (600 dpi, LZW, 190 mm wide).  Nothing under
ROOT/arch_v4, ROOT/analysis_v3 or ROOT/data is written.  Fig. 6 needs RDKit:

    python run_figures.py   (an environment with RDKit, matplotlib, pandas and Pillow)

One color per model in every figure (Okabe-Ito palette; Chemprop, the primary model, has the strong blue that is also the accent of
the paper), one palette for the four score tiers (blue ramp of the Chemprop probability), neutral greys for the two reference scores.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # ROOT = HerbCYP
FINAL = ROOT / "analysis_single" / "results" / "final"
APP = FINAL / "app"
BUNDLE = ROOT / "arch_v4" / "bundle"                     # input tables (read only)
V3RES = ROOT / "analysis_v3" / "results"                 # split-independent annotation tables (read only)
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4

# one color per model; the two reference scores are greys
COL = {"Chemprop": "#0072B2", "MapLight": "#009E73", "TabPFN-desc": "#56B4E9", "RF": "#CC79A7", "Size": "#6B6B6B", "LogP": "#A9A9A9",
       "random": "#000000"}
LAB = {"Chemprop": "Chemprop", "MapLight": "MapLight-style CatBoost", "TabPFN-desc": "TabPFN on descriptors", "RF": "Random forest",
       "Size": "Heavy-atom count", "LogP": "Crippen logP"}
ORDER = ["Chemprop", "MapLight", "TabPFN-desc", "RF", "Size", "LogP"]
FOUR = ORDER[:4]                                         # the four trained models
TIER = {"Low": "#B7D3E6", "Moderate": "#8DB9D6", "High": "#5E9CC6", "Very high": "#0072B2"}
ACCENT = COL["Chemprop"]
PALE = "#E3EEF7"                                         # pale fill of the accent (cohort box, outputs)
GRID = "#E2E2E2"


def setup():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3, "figure.dpi": 150,
        "savefig.dpi": 600, "ps.fonttype": 42, "pdf.fonttype": 42, "svg.fonttype": "none", "mathtext.default": "regular",
    })


def panel_label(ax, letter, x=-0.12, y=1.04, size=10):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size, fontweight="bold", va="bottom", ha="left")


def save(fig, stem, tiff=True, eps=True):
    """PDF and EPS (vector), 300 dpi PNG preview and 600 dpi LZW TIFF, in FIGDIR."""
    from PIL import Image
    fig.savefig(FIGDIR / f"{stem}.pdf")
    if eps:
        fig.savefig(FIGDIR / f"{stem}.eps")
    fig.savefig(FIGDIR / f"{stem}.png", dpi=300)
    if tiff:
        tmp = FIGDIR / f"{stem}_tmp.png"
        fig.savefig(tmp, dpi=600)
        Image.open(tmp).convert("RGB").save(FIGDIR / f"{stem}.tiff", compression="tiff_lzw", dpi=(600, 600))
        tmp.unlink()


def axis_limits(values, pad: float, step: float, lo_floor=None, hi_cap=None):
    """Axis limits that contain all values with a margin, rounded outward to a multiple of step."""
    import math
    lo = math.floor((min(values) - pad) / step) * step
    hi = math.ceil((max(values) + pad) / step) * step
    if lo_floor is not None:
        lo = max(lo, lo_floor)
    if hi_cap is not None:
        hi = min(hi, hi_cap)
    return round(lo, 6), round(hi, 6)


def est_table() -> pd.DataFrame:
    return pd.read_csv(FINAL / "estimates.csv")


def pair_table() -> pd.DataFrame:
    return pd.read_csv(FINAL / "paired.csv")


def one(df: pd.DataFrame, **kw) -> pd.Series:
    """The single row of df that matches all keyword filters (an error if there is not exactly one)."""
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    sub = df[m]
    if len(sub) != 1:
        raise ValueError(f"{len(sub)} rows for {kw}")
    return sub.iloc[0]


def primary_check() -> None:
    """The figures label Chemprop as the primary model: confirm that the selection file says so."""
    sel = json.load(open(FINAL / "primary_model_selection.json", encoding="utf-8"))
    txt = json.dumps(sel)
    assert "Chemprop" in txt
    t = pd.read_csv(FINAL / "primary_model_selection.csv")
    assert t[t.selected].model.tolist() == ["Chemprop"]


HA_EDGES = [-np.inf, 7, 9, 10, 11, 13, 16, 20, np.inf]       # the eight heavy-atom strata of the size-stratified AUROC (analysis_single/code/common.py EDGES)


def size_matched_yield() -> dict:
    """Yield of the top 10% of the Chemprop ranking of the cohort against the number of inhibitors expected at random within strata: the two weight classes at 300 Da
    (31.5 expected, 2.2 times) and the eight heavy-atom strata of the size-stratified AUROC (39.3 expected, 1.8 times).  Ties are ordered by the canonical SMILES; the strata
    are checked against the size_bin column of herb_scores_all_models.csv and the numbers against the figures typed in Section 3.6 of the manuscript."""
    sc = pd.read_csv(APP / "herb_cohort_scored.csv").sort_values(["score", "smiles"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    bud = pd.read_csv(APP / "herb_budget.csv")
    b10 = bud[np.isclose(bud.budget, 0.10)].iloc[0]
    k, found = int(b10.assays), int(b10.found)
    assert int(sc.label.iloc[:k].sum()) == found
    size_bin = pd.read_csv(FINAL / "herb_scores_all_models.csv", usecols=["id", "size_bin"]).set_index("id").size_bin
    bins = pd.cut(sc.heavy, HA_EDGES)
    assert (bins.astype(str).to_numpy() == size_bin.loc[sc.id].to_numpy()).all(), "the heavy-atom strata no longer equal the size_bin column of herb_scores_all_models.csv"
    prev = sc.groupby(bins, observed=True).label.mean()
    exp_ha = float(sum((bins.iloc[:k] == b).sum() * prev[b] for b in prev.index))
    w = np.where(sc.mw <= 300, "le300", "gt300")
    exp_mw = float(sum((w[:k] == c).sum() * sc.label[w == c].mean() for c in ("le300", "gt300")))
    out = dict(k=k, found=found, exp_ha=exp_ha, enr_ha=found / exp_ha, exp_mw=exp_mw, enr_mw=found / exp_mw, n_le300=int((sc.mw <= 300).sum()), n_strata=len(prev))
    assert (f"{exp_mw:.1f}", f"{out['enr_mw']:.1f}", f"{exp_ha:.1f}", f"{out['enr_ha']:.1f}", out["n_le300"], out["n_strata"]) == ("31.5", "2.2", "39.3", "1.8", 933, 8), out
    return out
