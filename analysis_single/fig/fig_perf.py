"""Figure 3: ranking performance of Chemprop (primary model) and the other models in the HERB evaluation cohort (at most 500 Da).

Everything is read from results/final of the single-model analysis (estimates.csv, paired.csv, herb_recovery_curves.csv); axis limits follow the data.
Panels: (a) recovery curves, (b) AUROC and average precision with 95% scaffold-cluster intervals, (c) paired differences Chemprop minus each comparator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style_single as st  # noqa: E402

st.setup()
st.primary_check()
COL, LAB, ORDER = st.COL, st.LAB, st.ORDER
est, pair = st.est_table(), st.pair_table()
curves = pd.read_csv(st.FINAL / "herb_recovery_curves.csv")
AN = "herb"
a0 = st.one(est, analysis=AN, model="Chemprop", metric="auroc")
N, POS = int(a0.n), int(a0.n_pos)

fig = plt.figure(figsize=(190 * st.MM, 78 * st.MM))
ax_a = fig.add_axes([0.068, 0.17, 0.200, 0.74])
ax_b1 = fig.add_axes([0.400, 0.17, 0.110, 0.74])
ax_b2 = fig.add_axes([0.528, 0.17, 0.095, 0.74])
ax_c = fig.add_axes([0.790, 0.17, 0.195, 0.74])

# ---------------- (a) recovery curves ----------------
styles = {"Chemprop": dict(lw=2.2, ls="-", z=7), "MapLight": dict(lw=1.2, ls="-", z=5), "TabPFN-desc": dict(lw=1.2, ls="-", z=4),
          "RF": dict(lw=1.2, ls="-", z=3), "Size": dict(lw=1.2, ls=(0, (4, 2)), z=2), "LogP": dict(lw=1.2, ls=(0, (4, 2)), z=1)}
kmax = int(0.4 * N)
k10 = int(np.ceil(0.10 * N))
for m, s_ in styles.items():
    d = curves[curves.model == m].sort_values("k")
    d = d[d.k <= kmax]
    ax_a.plot(d.k / N * 100, d.cum_hits / POS * 100, color=COL[m], lw=s_["lw"], ls=s_["ls"], zorder=s_["z"], label=LAB[m])
    h10 = int(curves[(curves.model == m) & (curves.k == k10)].cum_hits.iloc[0])
    ax_a.plot(k10 / N * 100, h10 / POS * 100, "o", color=COL[m], ms=4.2 if m == "Chemprop" else 3.6, zorder=8, mec="white", mew=0.5)
ax_a.plot([0, 40], [0, 40], color="black", lw=0.8, ls=(0, (1, 2)), label="Random order")
ax_a.plot([10, 10], [0, 66], color="#BBBBBB", lw=0.6, zorder=0)       # 10% budget; stops below the legend so that it does not run through the legend text
ax_a.set_xlim(0, 40)
ax_a.set_ylim(0, 102)
ax_a.set_yticks([0, 20, 40, 60, 80, 100])
ax_a.set_xlabel(f"Ingredients tested (% of {N:,})")
ax_a.set_ylabel(f"Inhibitors found (% of {POS})")
ax_a.set_xticks([0, 10, 20, 30, 40])
h10m = int(curves[(curves.model == "Chemprop") & (curves.k == k10)].cum_hits.iloc[0])
ax_a.text(39.3, 3.0, f"Circles: top 10%\n({k10} ingredients).\nChemprop: {h10m} of {POS}", fontsize=6.4, color=COL["Chemprop"],
          ha="right", va="bottom", linespacing=1.15, zorder=6)    # below the diagonal, clear of the curves
ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, handlelength=1.8, labelspacing=0.15, borderaxespad=0.1, fontsize=6.2)
st.panel_label(ax_a, "a", x=-0.21, y=1.04)


# ---------------- (b) forest plots ----------------
def wrapped(m):
    """Model name as in the legend of panel a, broken over two lines where it is long (the words do not change)."""
    return {"MapLight": "MapLight-style\nCatBoost", "TabPFN-desc": "TabPFN on\ndescriptors", "Size": "Heavy-atom\ncount"}.get(m, LAB[m])


ys = np.arange(len(ORDER))[::-1]
for ax, metric, title in ((ax_b1, "auroc", "AUROC"), (ax_b2, "ap", "Average precision")):
    sub = est[(est.analysis == AN) & est.model.isin(ORDER) & (est.metric == metric)]
    xlim = st.axis_limits(list(sub.ci_lo) + list(sub.ci_hi), pad=0.02, step=0.05 if metric == "ap" else 0.02)
    for y, m in zip(ys, ORDER):
        r = st.one(est, analysis=AN, model=m, metric=metric)
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=COL[m], lw=1.7 if m == "Chemprop" else 1.3, zorder=3)
        ax.plot(r.estimate, y, "o", color=COL[m], ms=4.8 if m == "Chemprop" else 3.8, zorder=4, mec="white", mew=0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([wrapped(m) for m in ORDER] if ax is ax_b1 else [], fontsize=6.4, linespacing=1.0)
    for t, m in zip(ax.get_yticklabels(), ORDER):
        if m == "Chemprop":
            t.set_fontweight("bold")
            t.set_color(COL["Chemprop"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.3, color=st.GRID, zorder=0)
    ax.set_xlabel(title)
    if metric == "ap":
        ax.axvline(POS / N, color="black", lw=0.7, ls=(0, (1, 2)), zorder=1)      # prevalence: AP of a random order
st.panel_label(ax_b1, "b", x=-1.1, y=1.04)

# ---------------- (c) paired differences ----------------
mets = [("auroc", "AUROC", "o"), ("ap", "AP", "s"), ("recall_10", "Recall, top 10%", "^"), ("auroc_strat", "Size-stratified AUROC", "D")]
refs = [m for m in ORDER if m != "Chemprop"]
ypos = np.arange(len(refs))[::-1]
off = np.linspace(-0.27, 0.27, len(mets))
for j, (metric, lab, mk) in enumerate(mets):
    for y, ref in zip(ypos, refs):
        r = st.one(pair, analysis=AN, candidate="Chemprop", reference=ref, metric=metric)
        sig = (r.ci_lo > 0) or (r.ci_hi < 0)
        ax_c.plot([r.ci_lo, r.ci_hi], [y + off[j]] * 2, color="#555555", lw=0.8, zorder=3)
        ax_c.plot(r.estimate, y + off[j], mk, color=COL["Chemprop"] if sig else "white", mec=COL["Chemprop"], mew=0.9, ms=3.6, zorder=4,
                  label=lab if y == ypos[0] else None)
ax_c.axvline(0, color="black", lw=0.8, zorder=1)
ax_c.set_yticks(ypos)
ax_c.set_yticklabels([wrapped(m) for m in refs], fontsize=6.4, linespacing=1.0)
ax_c.tick_params(axis="y", length=0)
ax_c.set_ylim(-0.6, len(refs) - 0.4)
ax_c.set_xlabel("Chemprop minus comparator")
ax_c.grid(axis="x", lw=0.3, color=st.GRID, zorder=0)
ax_c.legend(loc="upper right", frameon=False, fontsize=6.2, handletextpad=0.2, labelspacing=0.25, borderaxespad=0.0)
st.panel_label(ax_c, "c", x=-0.5, y=1.04)
fig.text(0.985, 0.012, "Filled symbols: 95% interval excludes zero", ha="right", va="bottom", fontsize=6.2, color="#555555")

st.save(fig, "Fig3")
print("Fig3 saved")
