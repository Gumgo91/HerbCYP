"""Figure 1: data and coverage of the evidence (application domain of at most 500 Da).

The panels show data facts (HERB universe, assay evidence, training pool, coverage by molecular weight, descriptor space).  They do not
depend on the split or on the model.  Coverage counts are read from results/final/coverage_by_mw.csv; cohort and pool counts are read from the
bundle; HERB-universe counts are checked against analysis_v3/results/numbers.json.
Panel c plots the coordinates of analysis_v3/results/a5_embedding.csv, a t-SNE embedding of a random sample of 4,000 training-pool structures, one of
7,000 unmeasured HERB structures (seed 2026) and all 1,036 measured HERB structures, each described by 200 normalized RDKit 2D descriptors (the structures
without a complete descriptor vector were dropped: 3,983, 6,995 and 1,036 remain), standardized over these structures, clipped at +-6 SD, reduced to 40
principal components (seed 2026) and embedded by t-SNE (perplexity 60, PCA initialization, seed 2026).  The file is read as it is and is not recomputed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style_single as st  # noqa: E402

st.setup()
ROOT = st.ROOT
cov = pd.read_csv(st.FINAL / "coverage_by_mw.csv")
emb = pd.read_csv(st.V3RES / "a5_embedding.csv")             # t-SNE of pool and HERB structures (split-independent)
NUM = json.load(open(st.V3RES / "numbers.json", encoding="utf-8"))
pool = pd.read_csv(st.BUNDLE / "pool.csv", usecols=["parent_inchikey"])
_tdc_tables = ROOT / "data" / "processed" / "tdc_trainval_std.csv"
if _tdc_tables.exists():                                     # standardized TDC tables present: counts from them
    _tv = pd.read_csv(_tdc_tables)
    N_TV, N_TE = len(_tv), len(pd.read_csv(ROOT / "data" / "processed" / "tdc_test_std.csv"))
    TDC_REMOVED = int((~_tv.parent_inchikey.isin(set(pool.parent_inchikey))).sum())
else:                                                        # otherwise: record counts of the library (tdc_counts.json; the labeled tables are not shipped)
    _cnt = json.load(open(st.V3RES / "tdc_counts.json", encoding="utf-8"))
    N_TV, N_TE = _cnt["n_train_validation_records"], _cnt["n_test_records"]
    TDC_REMOVED = _cnt["n_train_validation_records_removed_for_parent_overlap"]
POOL_REC, POOL_N = N_TV - TDC_REMOVED, len(pool)
unm = pd.read_csv(st.BUNDLE / "unmeasured.csv").drop_duplicates("smiles").set_index("smiles")
emb.loc[emb.group == "unmeasured HERB", "group"] = np.where(
    unm.reindex(emb.loc[emb.group == "unmeasured HERB", "canonical_smiles"]).mw.values <= 500, "unmeasured <=500", "unmeasured >500")
herb = pd.read_csv(st.BUNDLE / "herb.csv")
h5 = herb[herb.mw <= 500]
N_ALL = len(herb)
N5, POS5 = len(h5), int(h5.label.sum())
# HERB-universe counts typed in panel a (evidence tables); the ones in numbers.json are checked
assert int(cov.n_universe.sum()) == NUM["uni_n_universe"] == 23062 and int(cov.n_concordant.sum()) == NUM["uni_n_concordant"] == 1102
assert NUM["herb_herbs_with_structure"] == 7130 and NUM["uni_n_conflict"] == 90 and NUM["n_cohort"] == N_ALL == 1036
assert (N_TV, N_TV + N_TE, TDC_REMOVED, POOL_REC, POOL_N) == (9861, 12328, 119, 9742, 9726)
assert (N5, POS5) == (1028, 171)
# reasons for the removal of the 119 train-validation records: 113 share a chemical parent with a HERB ingredient with eligible evidence, 4 more are identical to a HERB
# ingredient without such evidence, 2 more have contradictory labels; none shares a chemical parent with an Ambe or non-herbal compound
REMOVED_HERB_EVIDENCE, REMOVED_IDENTICAL, REMOVED_CONFLICT = 113, 4, 2
assert REMOVED_HERB_EVIDENCE + REMOVED_IDENTICAL + REMOVED_CONFLICT == TDC_REMOVED


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


W, H = 190, 128
fig = plt.figure(figsize=(W * st.MM, H * st.MM))
ax = fig.add_axes([0.012, 0.585, 0.976, 0.40])
ax.set_xlim(0, 190)
ax.set_ylim(0, 52)
ax.axis("off")
st.panel_label(ax, "a", x=0.0, y=0.97)

BLUE = st.ACCENT
GREY = "#8C8C8C"
INHIB = "#333333"          # inhibitor rate in panel b: charcoal
BOXES = []                 # (patch, [texts]) for the overflow check


def box(x, y, w, h, lines, fc="#F4F4F4", ec="#5F6B73", lw=0.8, bold_first=True, size=7.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.6", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    n = len(lines)
    ts = []
    for i, ln in enumerate(lines):
        ts.append(ax.text(x + w / 2, y + h / 2 + (n / 2 - i - 0.5) * 3.9, ln, ha="center", va="center", fontsize=size,
                          fontweight="bold" if (i == 0 and bold_first) else "normal"))
    BOXES.append((x, y, w, h, ts))


def arrow(x0, y0, x1, y1, color="#5F6B73", lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, shrinkA=0, shrinkB=0))


bw, bh, ytop = 31, 15, 31
GAP = 7.375                                            # equal gaps between the five boxes of the top row (arrows 6.4 mm long)
xs = [6 + i * (bw + GAP) for i in range(4)] + [159.5]
box(xs[0], ytop, bw, bh, [f"{NUM['uni_n_universe']:,} structures", "from 31,164 parseable", f"records of {NUM['herb_herbs_with_structure']:,} herbs"])
box(xs[1], ytop, bw, bh, ["1,192 structures", "with eligible assay", "evidence"])
box(xs[2], ytop, bw, bh, [f"{NUM['uni_n_concordant']:,} structures", "with concordant", "inhibition calls"])
box(xs[3], ytop, bw, bh, [f"{N_ALL:,} structures", "absent from the", "TDC library"])
box(xs[4], ytop, 29.5, bh, ["Evaluation cohort", f"{N5:,} at most 500 Da", f"{POS5} inhibitors"], fc=st.PALE, ec=BLUE, lw=1.2, size=6.6)
for i in range(4):
    arrow(xs[i] + bw + 0.5, ytop + bh / 2, xs[i + 1] - 0.5, ytop + bh / 2)
ax.text(xs[1] - 3.2, ytop + bh + 2.0, "20 PubChem assays and the Ambe et al. dataset", fontsize=6.4, ha="left", va="bottom", color="#444444")
ax.text(xs[1] + bw + GAP / 2, ytop - 1.4, f"{NUM['uni_n_conflict']} conflicting\ncalls removed", fontsize=6.4, ha="center", va="top", color="#8B0000", linespacing=1.05)
ax.text(xs[2] + bw + GAP / 2, ytop - 1.4, "66 also in the\nTDC library", fontsize=6.4, ha="center", va="top", color="#8B0000", linespacing=1.05)
ax.text(xs[4] + 14.75, ytop + bh + 2.0, f"{N_ALL - N5} above 500 Da\nremoved", fontsize=6.4, ha="center", va="bottom", color="#8B0000", linespacing=1.05)

ybot, bh2 = 1.0, 21                                    # bottom row: five-line box for the three reasons of the 119 removals
box(6, ybot, 34, bh2, ["TDC CYP3A4 library", f"{N_TV:,} train-validation", f"records ({N_TV + N_TE:,} in all)"], size=6.8)
box(48, ybot, 80, bh2, [f"Training pool: {POOL_REC:,} records ({POOL_N:,} structures)", f"after removing {TDC_REMOVED} whose chemical parent matches",
                        f"a HERB ingredient with eligible evidence ({REMOVED_HERB_EVIDENCE}),",
                        "a structure identical to a HERB ingredient without",
                        f"evidence ({REMOVED_IDENTICAL}) or contradictory labels ({REMOVED_CONFLICT})"], bold_first=False, size=6.8)
box(133.7, ybot, 55.3, bh2, ["Four models, five seeds", "Chemprop, MapLight-style CatBoost,", "TabPFN on descriptors,", "random forest"], size=6.8)
arrow(40.5, ybot + bh2 / 2, 47.4, ybot + bh2 / 2)
arrow(128.5, ybot + bh2 / 2, 133.1, ybot + bh2 / 2)
arrow(174.25, ybot + bh2 + 0.9, 174.25, ytop - 0.6, color=BLUE, lw=1.1)
ax.text(176.2, (ybot + bh2 + 0.9 + ytop) / 2, "scores", fontsize=6.4, color=BLUE, ha="left", va="center")

# ---------------- (b) coverage by molecular weight ----------------
ax1 = fig.add_axes([0.115, 0.285, 0.365, 0.215])
cats = ["≤200", "200–300", "300–400", "400–500", ">500"]
x = np.arange(len(cats))
pct = cov.coverage_concordant.to_numpy() * 100
ax1.set_yscale("log")
ax1.set_ylim(0.05, 400)
ax1.vlines(x[:4], 0.05, pct[:4], color=BLUE, lw=5.5, zorder=3)
ax1.vlines(x[4:], 0.05, pct[4:], color=GREY, lw=5.5, zorder=3)
ax1.axvspan(3.5, 4.5, color="#F0F0F0", zorder=0)
ax1.text(4.0, 150, "outside\ndomain", ha="center", va="center", fontsize=6.1, color="#666666", linespacing=1.05)
ax1.set_yticks([0.1, 1, 10])
ax1.set_yticklabels(["0.1", "1", "10"])
ax1.set_ylabel("Measured structures (%)")
ax1.set_xlim(-0.55, len(cats) - 0.45)
ax1.set_xticks(x)
ax1.set_xticklabels([])
ax1.tick_params(axis="x", length=0)
ax1.grid(axis="y", lw=0.3, color="#DDDDDD", zorder=0)
for xi, (p, nm, nu) in enumerate(zip(pct, cov.n_concordant, cov.n_universe)):
    ax1.text(xi, p * 1.35, f"{p:.2g}%\n{int(nm):,}/{int(nu):,}" if p < 10 else f"{p:.1f}%\n{int(nm):,}/{int(nu):,}",
             ha="center", va="bottom", fontsize=6.1, color="#222222", linespacing=1.05)
ax1.set_title("Concordant measurements by molecular weight", fontsize=7.2, pad=8, loc="left")
# panel letters b and c share one baseline (the baseline of the two panel titles: top of the axes at 0.50 of the figure height plus the 8 pt title pad)
LETTER_Y = 0.50 + 8 / (H * st.MM * 72)
fig.text(0.0274, LETTER_Y, "b", fontsize=10, fontweight="bold", va="baseline", ha="left")
fig.text(0.522, LETTER_Y, "c", fontsize=10, fontweight="bold", va="baseline", ha="left")

ax1b = fig.add_axes([0.115, 0.115, 0.365, 0.12], sharex=ax1)
ax1b.axvspan(3.5, 4.5, color="#F0F0F0", zorder=0)
# lower chart: the 1,036 evaluation structures (absent from the TDC library), by molecular-weight class; n per class is part of the tick labels
assert int(cov.n_eval.sum()) == NUM["n_cohort"] == 1036
for xi, (k, n) in enumerate(zip(cov.eval_positives, cov.n_eval)):
    lo, hi = wilson(float(k), float(n))
    small = n < 20
    col = GREY if xi == 4 else INHIB
    ax1b.plot([xi, xi], [lo * 100, hi * 100], color=col, lw=1.0)
    ax1b.plot(xi, k / n * 100, "o", ms=3.8, mfc="white" if small else col, mec=col, mew=1.0)
ax1b.set_title("Inhibitors among the 1,036 structures absent from TDC", fontsize=6.4, loc="left", pad=2)
ax1b.set_ylim(-4, 104)
ax1b.set_yticks([0, 50, 100])
ax1b.set_ylabel("Inhibitors (%)")
ax1b.set_xticks(x)
ax1b.set_xticklabels([f"{c}\nn = {int(m):,}" for c, m in zip(cats, cov.n_eval)], linespacing=1.15)   # n per class under the lower chart
ax1.tick_params(axis="x", length=0, labelbottom=False)
ax1b.set_xlabel("Molecular weight (Da)")
ax1b.grid(axis="y", lw=0.3, color="#DDDDDD")

# ---------------- (c) chemical-space map ----------------
ax2 = fig.add_axes([0.545, 0.135, 0.44, 0.365])
order = ["train library", "unmeasured >500", "unmeasured <=500", "measured HERB"]
colr = {"train library": "#BDBDBD", "unmeasured >500": "#F2D49B", "unmeasured <=500": "#E69F00", "measured HERB": BLUE}
sz = {"train library": 3.2, "unmeasured >500": 3.2, "unmeasured <=500": 3.2, "measured HERB": 5.5}
al = {"train library": 0.6, "unmeasured >500": 0.55, "unmeasured <=500": 0.5, "measured HERB": 0.9}
for g in order:
    d = emb[emb.group == g]
    ax2.scatter(d.x, d.y, s=sz[g], c=colr[g], alpha=al[g], lw=0, rasterized=True)
ax2.set_xticks([])
ax2.set_yticks([])
for s in ("left", "bottom"):
    ax2.spines[s].set_visible(False)
ax2.set_xlabel("t-SNE 1", labelpad=1)
ax2.set_ylabel("t-SNE 2", labelpad=1)
ax2.set_title("Chemical space (RDKit descriptors)", fontsize=7.2, pad=8, loc="left")
handles = [Line2D([], [], marker="o", ls="", ms=5, color=colr["train library"], label="Training pool"),
           Line2D([], [], marker="o", ls="", ms=5, color=colr["unmeasured <=500"], label="Unmeasured, ≤500 Da"),
           Line2D([], [], marker="o", ls="", ms=5, color=colr["unmeasured >500"], label="Unmeasured, >500 Da"),
           Line2D([], [], marker="o", ls="", ms=5, color=colr["measured HERB"], label="Measured HERB")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.765, -0.005), ncol=2, frameon=False,
           handletextpad=0.2, columnspacing=1.6, labelspacing=0.3, fontsize=6.4)

# ---------------- text of every box lies inside its box ----------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()
for (x0, y0, w, h, ts) in BOXES:
    for t in ts:
        bb = t.get_window_extent(rend)
        (xa, ya), (xb, yb) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        assert xa >= x0 + 0.4 and xb <= x0 + w - 0.4 and ya >= y0 and yb <= y0 + h, f"text overflows its box: {t.get_text()!r} {xa:.1f} {xb:.1f} in {x0:.1f}..{x0 + w:.1f}"

st.save(fig, "Fig1")
print("Fig1 saved")
