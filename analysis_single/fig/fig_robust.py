"""Figure 4: AUROC in strata of the cohort and in the development sets, selection of the primary model, and subsets of the cohort.

All values are read from results/final (estimates.csv, dev_single_models.csv / primary_model_selection.csv,
paired_extra.csv, estimates_extra.csv).  One color per model as in every figure (Chemprop, the primary model, has the strong blue).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style_single as st  # noqa: E402

st.setup()
st.primary_check()
COL, LAB = st.COL, st.LAB
est = st.est_table()
sel = pd.read_csv(st.FINAL / "primary_model_selection.csv")
dev = pd.read_csv(st.FINAL / "dev_single_models.csv")
pair_x = pd.read_csv(st.FINAL / "paired_extra.csv")
est_x = pd.read_csv(st.FINAL / "estimates_extra.csv")
MODELS = st.FOUR

fig = plt.figure(figsize=(190 * st.MM, 128 * st.MM))
ax_a = fig.add_axes([0.058, 0.615, 0.500, 0.315])
ax_b = fig.add_axes([0.628, 0.615, 0.357, 0.315])
ax_c = fig.add_axes([0.215, 0.085, 0.235, 0.345])
ax_d = fig.add_axes([0.745, 0.085, 0.240, 0.345])


def grouped(ax, groups, models=MODELS, ylim=(0.5, 1.0)):
    n = len(models)
    w = 0.8 / n
    for gi, (tag, _) in enumerate(groups):
        for mi, m in enumerate(models):
            r = est[(est.analysis == tag) & (est.model == m) & (est.metric == "auroc")]
            if r.empty:
                raise ValueError(f"no estimate for {tag} {m}")
            r = r.iloc[0]
            assert r.ci_lo > ylim[0], f"interval below the axis: {tag} {m} {r.ci_lo:.3f}"
            x = gi + (mi - (n - 1) / 2) * w
            ax.plot([x, x], [r.ci_lo, r.ci_hi], color=COL[m], lw=1.4 if m == "Chemprop" else 1.1, zorder=3)
            ax.plot(x, r.estimate, "o", color=COL[m], ms=4.2 if m == "Chemprop" else 3.3, mec="white", mew=0.4, zorder=4)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(len(groups)))
    lab = []
    for tag, nm in groups:
        r = est[(est.analysis == tag) & (est.model == "Chemprop") & (est.metric == "auroc")].iloc[0]
        lab.append(nm + f"\n{int(r.n):,} ({int(r.n_pos):,})")
    ax.set_xticklabels(lab, fontsize=6.2, linespacing=1.15)
    ax.tick_params(axis="x", length=0)
    ax.axhline(0.5, color="black", lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.grid(axis="y", lw=0.3, color=st.GRID, zorder=0)
    ax.set_ylabel("AUROC")


# (a) HERB cohort strata
grouped(ax_a, [("herb_sim_lt0.4", "Similarity\n<0.4"), ("herb_sim_0.4_0.6", "Similarity\n0.4–0.6"), ("herb_sim_ge0.6", "Similarity\n≥0.6"),
               ("herb_mw_le200", "≤200\nDa"), ("herb_mw_200_300", "200–300\nDa"), ("herb_mw_300_500", "300–500\nDa"),
               ("herb_np_ge0", "NP-like"), ("herb_np_lt0", "Not\nNP-like")])
ax_a.axvline(2.5, color="#CCCCCC", lw=0.6)
ax_a.axvline(5.5, color="#CCCCCC", lw=0.6)
ax_a.set_title("HERB evaluation cohort (ingredients, inhibitors in parentheses)", fontsize=7.0, loc="left", pad=3)
st.panel_label(ax_a, "a", x=-0.105, y=1.10)

# (b) development sets and outside the domain
grouped(ax_b, [("ambe", "Ambe\nset"), ("stress_le500", "Non-herbal\n≤500 Da"), ("stress_le500_np_ge0", "Non-herbal\nNP-like\n≤500 Da"),
               ("stress_gt500", "Non-herbal\n>500 Da"), ("stress_gt700", "Non-herbal\n>700 Da")])
ax_b.axvspan(2.5, 4.6, color="#F2F2F2", zorder=0)
ax_b.text(3.55, 0.995, "outside the domain", ha="center", va="top", fontsize=6.2, color="#666666")
ax_b.set_title("Development sets", fontsize=7.0, loc="left", pad=3)
st.panel_label(ax_b, "b", x=-0.14, y=1.10)
h = [Line2D([], [], marker="o", ls="", color=COL[m], ms=4, label=LAB[m]) for m in MODELS]
fig.legend(handles=h, loc="center", bbox_to_anchor=(0.50, 0.522), ncol=4, frameon=False, fontsize=6.3, handletextpad=0.1, columnspacing=1.4)

# (c) selection of the primary model: AUROC in the two development sets (up to 500 Da) and their mean
cand = sel.sort_values("rank").model.tolist()                                       # the four candidates in the order of the rule
assert cand == MODELS and sel[sel.selected].model.tolist() == ["Chemprop"]
base = ["Size", "LogP"]                                                             # reference scores, not ranked by the rule
ypos = {m: y for m, y in zip(cand + base, [6.0, 5.0, 4.0, 3.0, 1.2, 0.2])}
dv = dev.set_index("variant")
XMEAN = 0.903                                                                       # right edge of the column of printed means
assert max(float(dv.loc[m, c_]) for m in cand + base for c_ in ("ambe_auroc", "stress_auroc", "dev_mean_auroc")) < XMEAN - 0.045, "a marker reaches the column of means"
for m in cand + base:
    y = ypos[m]
    a_, s_, mean_ = float(dv.loc[m, "ambe_auroc"]), float(dv.loc[m, "stress_auroc"]), float(dv.loc[m, "dev_mean_auroc"])
    ax_c.plot(a_, y + 0.24, "^", mfc="white", mec=COL[m], mew=1.0, ms=4.2, zorder=3)
    ax_c.plot(s_, y - 0.24, "s", mfc="white", mec=COL[m], mew=1.0, ms=3.7, zorder=3)
    ax_c.plot(mean_, y, "o", color=COL[m], ms=6.0 if m == "Chemprop" else 4.8, mec="white", mew=0.5, zorder=4)
    ax_c.text(XMEAN, y, f"{mean_:.3f}", va="center", ha="right", fontsize=6.2, fontweight="bold" if m == "Chemprop" else "normal",
              color=COL["Chemprop"] if m == "Chemprop" else "#222222")     # printed values are the means (column headed "Mean")
ax_c.text(XMEAN, 6.55, "Mean", fontsize=6.2, fontweight="bold", color="#222222", ha="right", va="center")
ax_c.axhline(2.1, color="#BBBBBB", lw=0.6, ls=(0, (1, 2)))
ax_c.text(0.905, 2.2, "reference scores", fontsize=6.0, color="#666666", ha="right", va="bottom")
ax_c.set_yticks([ypos[m] for m in cand + base])
ax_c.set_yticklabels([LAB[m] for m in cand + base], fontsize=6.2)
for t, m in zip(ax_c.get_yticklabels(), cand + base):
    if m == "Chemprop":
        t.set_fontweight("bold")
        t.set_color(COL["Chemprop"])
ax_c.tick_params(axis="y", length=0)
ax_c.set_ylim(-0.4, 6.7)
ax_c.set_xlim(0.68, 0.905)
ax_c.set_xticks([0.7, 0.75, 0.8, 0.85, 0.9])
ax_c.set_xlabel("AUROC, Ambe and non-herbal sets (≤500 Da)")
ax_c.grid(axis="x", lw=0.3, color=st.GRID, zorder=0)
ax_c.set_title("Selection of the primary model", fontsize=7.0, loc="left", pad=3)
st.panel_label(ax_c, "c", x=-0.60, y=1.10)
key = [Line2D([], [], marker="o", ls="", color="#555555", ms=4.6, label="Mean of the two sets"),
       Line2D([], [], marker="^", ls="", mfc="white", mec="#555555", ms=4.2, label="Ambe set"),
       Line2D([], [], marker="s", ls="", mfc="white", mec="#555555", ms=3.8, label="Non-herbal set")]
ax_c.legend(handles=key, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=6.0, handletextpad=0.1, labelspacing=0.25,
            borderaxespad=0.1)

# (d) post hoc subsets of the cohort: paired AUROC difference, Chemprop minus MapLight-style CatBoost and minus TabPFN on descriptors
subsets = [("herb", "All ingredients"), ("herb_ring_containing", "Ring-containing"), ("herb_ring_free_by_parent", "Ring-free, by parent"),
           ("herb_no_ambe_overlap", "Without Ambe overlap"), ("herb_no_mdp", "Without methylenedioxyphenyl"),
           ("herb_ring_systems", "Ring-system chemotypes")]
refs = [("MapLight", "o"), ("TabPFN-desc", "D")]
pair_all = st.pair_table()
yy = np.arange(len(subsets))[::-1]
for y, (tag, nm) in zip(yy, subsets):
    for j, (ref, mk) in enumerate(refs):
        src = pair_all if tag == "herb" else pair_x
        r = st.one(src, analysis=tag, candidate="Chemprop", reference=ref, metric="auroc")
        sig = (r.ci_lo > 0) or (r.ci_hi < 0)
        yj = y + (0.16 if j == 0 else -0.16)
        ax_d.plot([r.ci_lo, r.ci_hi], [yj, yj], color=COL[ref], lw=1.1, zorder=3)
        ax_d.plot(r.estimate, yj, mk, color=COL[ref] if sig else "white", mec=COL[ref], mew=1.0, ms=3.9, zorder=4)
ax_d.axvline(0, color="black", lw=0.8, zorder=1)
labs = []
for tag, nm in subsets:
    r = st.one(est if tag == "herb" else est_x, analysis=tag, model="Chemprop", metric="auroc")
    labs.append(f"{nm}\n{int(r.n):,} ({int(r.n_pos):,})")
ax_d.set_yticks(yy)
ax_d.set_yticklabels(labs, fontsize=6.0, linespacing=1.1)
ax_d.tick_params(axis="y", length=0)
ax_d.set_ylim(-0.6, len(subsets) + 0.95)
ax_d.set_xlim(-0.14, 0.14)
ax_d.set_xlabel("Chemprop minus comparator, AUROC")
ax_d.grid(axis="x", lw=0.3, color=st.GRID, zorder=0)
ax_d.set_title("Post hoc subsets of the cohort", fontsize=7.0, loc="left", pad=3)
st.panel_label(ax_d, "d", x=-0.80, y=1.10)
key = [Line2D([], [], marker="o", ls="-", color=COL["MapLight"], ms=3.9, lw=1.1, label="MapLight-style CatBoost"),
       Line2D([], [], marker="D", ls="-", color=COL["TabPFN-desc"], ms=3.9, lw=1.1, label="TabPFN on descriptors"),
       Line2D([], [], marker="o", ls="", color="#555555", ms=3.9, label="Filled: interval excludes zero")]
lg = ax_d.legend(handles=key, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=6.0, handletextpad=0.2, labelspacing=0.25,
                  borderaxespad=0.0, facecolor="white", edgecolor="none", framealpha=1.0)
lg.set_zorder(10)

st.save(fig, "Fig4")
print("Fig4 saved")
