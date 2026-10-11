"""Figure 5: what a laboratory can do with the Chemprop scores (score tiers, testing budgets, overview of the ranked list of unmeasured ingredients).

Read from results/final/app (herb_tiers, herb_budget, unmeasured_tier_by_mw, unmeasured_tier_by_chemotype,
unmeasured_tier_by_similarity).  Tiers apply to the Chemprop probability (edges 0.2, 0.5, 0.8)."""
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
tiers = pd.read_csv(st.APP / "herb_tiers.csv")
bud = pd.read_csv(st.APP / "herb_budget.csv")
by_mw = pd.read_csv(st.APP / "unmeasured_tier_by_mw.csv", index_col=0)
by_ct = pd.read_csv(st.APP / "unmeasured_tier_by_chemotype.csv", index_col=0)
by_sim = pd.read_csv(st.APP / "unmeasured_tier_by_similarity.csv", index_col=0)
TC = dict(st.TIER)
TN = list(TC)
assert tiers.tier.tolist() == TN and int(tiers.n.sum()) == 1028 and int(tiers.inhibitors.sum()) == 171
N_RANKED = int(by_sim.to_numpy().sum())
assert N_RANKED == int(by_ct.total.sum()) == int(by_mw.to_numpy().sum()) == 15511

fig = plt.figure(figsize=(190 * st.MM, 122 * st.MM))
ax_a = fig.add_axes([0.062, 0.610, 0.235, 0.300])
ax_b = fig.add_axes([0.395, 0.610, 0.270, 0.300])
ax_c = fig.add_axes([0.770, 0.610, 0.215, 0.300])
ax_d = fig.add_axes([0.318, 0.085, 0.300, 0.310])
ax_e = fig.add_axes([0.800, 0.085, 0.185, 0.310])

# ---------------- (a) inhibitor rate by tier ----------------
x = np.arange(len(tiers))
cohort_rate = tiers.inhibitors.sum() / tiers.n.sum() * 100
for xi, (_, r) in zip(x, tiers.iterrows()):
    ax_a.bar(xi, r.rate * 100, color=TC[r.tier], width=0.72, edgecolor="#6A6A6A", lw=0.4, zorder=3)
    has_ci = r.n >= 5 and (r.rate_hi - r.rate_lo) > 1e-9      # interval for tiers with at least five ingredients
    if has_ci:
        ax_a.plot([xi, xi], [r.rate_lo * 100, r.rate_hi * 100], color="#222222", lw=0.8, zorder=4)
    ytxt = max(r.rate_hi * 100, r.rate * 100) + 3 if has_ci else min(r.rate * 100 + 4, 106)
    if abs(ytxt - cohort_rate) < 4:                                   # keep the count clear of the dotted cohort line
        ytxt = cohort_rate + 3.5
    ax_a.text(xi, ytxt, f"{int(r.inhibitors)}/{int(r.n)}", ha="center", va="bottom", fontsize=6.2, color="#222222")
ax_a.axhline(cohort_rate, color="black", lw=0.7, ls=(0, (1, 2)), zorder=1)
ax_a.set_xticks(x)
ax_a.set_xticklabels([f"{t}\n{r}" for t, r in zip(tiers.tier, ["<0.2", "0.2–0.5", "0.5–0.8", "≥0.8"])], fontsize=6.3)
ax_a.set_ylim(0, 118)
ax_a.set_yticks([0, 25, 50, 75, 100])
ax_a.set_ylabel("Inhibitor rate in tier (%)")
ax_a.set_xlabel("Score range of the tier")
ax_a.grid(axis="y", lw=0.3, color=st.GRID, zorder=0)
ax_a.text(3.55, cohort_rate, "cohort\nrate", fontsize=6.2, color="#444444", ha="left", va="center", clip_on=False, linespacing=1.0)
ax_a.set_title("Score tiers, HERB cohort", fontsize=7.2, loc="left", pad=3)
st.panel_label(ax_a, "a", x=-0.225, y=1.10)

# ---------------- (b) budgets ----------------
xb = np.arange(len(bud))
w = 0.38
prev = tiers.inhibitors.sum() / tiers.n.sum()
ax_b.bar(xb - w / 2, bud.found, width=w, color=st.COL["Chemprop"], zorder=3, label="Ranked by Chemprop")
ax_b.bar(xb + w / 2, bud.assays * prev, width=w, color="#BDBDBD", zorder=3, label="Random order (expected)")
for xi, (_, r) in zip(xb, bud.iterrows()):
    ax_b.text(xi - w / 2, r.found + 2, f"{int(r.found)}", ha="center", va="bottom", fontsize=6.2, color=st.COL["Chemprop"], fontweight="bold")
ax_b.set_xticks(xb)
ax_b.set_xticklabels([f"{int(round(q * 100))}%\n({int(a)})" for q, a in zip(bud.budget, bud.assays)], fontsize=6.3)
ax_b.set_xlabel("Testing budget (assays)")
ax_b.set_ylabel("Inhibitors found")
ax_b.set_ylim(0, bud.found.max() * 1.30)
ax_b.grid(axis="y", lw=0.3, color=st.GRID, zorder=0)
ax_b.legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=1.2, borderaxespad=0.0)
b10 = bud[np.isclose(bud.budget, 0.10)].iloc[0]
SM = st.size_matched_yield()                      # the cohort is mostly small ingredients: yield of the top 10% within the eight heavy-atom strata
ax_b.text(0.02, 0.78, f"At 10%: {b10.assays_per_inhibitor:.1f} assays per inhibitor\n(random order: {b10.random_assays_per_inhibitor:.1f})\n"
                      f"{SM['enr_ha']:.1f} times random yield within\nheavy-atom strata", transform=ax_b.transAxes,
          fontsize=6.2, color="#444444", ha="left", va="top", linespacing=1.15)
ax_b.set_title("Testing budgets, HERB cohort", fontsize=7.2, loc="left", pad=3)
st.panel_label(ax_b, "b", x=-0.17, y=1.10)

# ---------------- (c) unmeasured ingredients by molecular weight ----------------
mw_labels = ["≤200", "200–300", "300–400", "400–500"]
bottom = np.zeros(len(by_mw))
for t in TN:
    v = by_mw[t].to_numpy(float)
    ax_c.bar(range(len(by_mw)), v, bottom=bottom, color=TC[t], width=0.74, edgecolor="#6A6A6A", lw=0.3, zorder=3)
    bottom += v
ax_c.set_xticks(range(len(by_mw)))
ax_c.set_xticklabels(mw_labels, fontsize=6.0)
ax_c.set_xlabel("Molecular weight (Da)")
ax_c.set_ylabel("Unmeasured ingredients")
ax_c.grid(axis="y", lw=0.3, color=st.GRID, zorder=0)
ax_c.set_title("Ranked list, by size", fontsize=7.2, loc="left", pad=3)
st.panel_label(ax_c, "c", x=-0.25, y=1.10)

# ---------------- (d) tiers by chemotype ----------------
top = by_ct.sort_values("total", ascending=False).head(10).iloc[::-1]
share = top[TN].div(top.total, axis=0) * 100
left = np.zeros(len(share))
for t in TN:
    ax_d.barh(range(len(share)), share[t], left=left, color=TC[t], height=0.72, edgecolor="#6A6A6A", lw=0.3, zorder=3)
    left += share[t].to_numpy()
ax_d.set_yticks(range(len(share)))
ax_d.set_yticklabels([f"{c} ({int(n):,})" for c, n in zip(share.index, top.total)], fontsize=6.0)
ax_d.set_xlim(0, 100)
ax_d.set_xlabel("Share of ingredients (%)")
ax_d.tick_params(axis="y", length=0)
ax_d.set_title("Ranked list, by chemotype (number in parentheses)", fontsize=7.2, loc="left", pad=3)
st.panel_label(ax_d, "d", x=-0.80, y=1.10)

# ---------------- (e) tiers by similarity to the training structures ----------------
order = [("remote from training", "<0.4"), ("moderate analogs", "0.4–0.6"), ("close analogs in training", "≥0.6")]
sh = by_sim.loc[[o for o, _ in order]]
tot = sh.sum(axis=1)
sh_pct = sh.div(tot, axis=0) * 100
yy = np.arange(len(order))[::-1]
left = np.zeros(len(order))
for t in TN:
    ax_e.barh(yy, sh_pct[t].to_numpy(), left=left, color=TC[t], height=0.66, edgecolor="#6A6A6A", lw=0.3, zorder=3)
    left += sh_pct[t].to_numpy()
ax_e.set_yticks(yy)
ax_e.set_yticklabels([f"{lab}\n({int(n):,})" for (o, lab), n in zip(order, tot)], fontsize=6.0, linespacing=1.1)
ax_e.set_xlim(0, 100)
ax_e.set_ylim(-0.55, len(order) - 0.45)
ax_e.tick_params(axis="y", length=0)
ax_e.set_xlabel("Share of ingredients (%)")
ax_e.set_title("Ranked list, by similarity\nto the training structures", fontsize=7.2, loc="left", pad=3, linespacing=1.1)
st.panel_label(ax_e, "e", x=-0.50, y=1.22)

handles = [plt.Rectangle((0, 0), 1, 1, fc=TC[t], ec="#6A6A6A", lw=0.3) for t in TN]
fig.legend(handles, TN, loc="center", bbox_to_anchor=(0.50, 0.468), ncol=4, frameon=False, fontsize=6.4, handlelength=1.2,
           handletextpad=0.4, columnspacing=1.6, title="Score tier (Chemprop)", title_fontsize=6.4, alignment="center")

print("ranked ingredients in panels c and e:", N_RANKED, "; chemotype panel shows", int(top.total.sum()), "of", int(by_ct.total.sum()))
st.save(fig, "Fig5")
print("Fig5 saved")
