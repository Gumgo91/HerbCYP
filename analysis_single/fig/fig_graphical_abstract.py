"""Graphical abstract (190 x 88 mm), drawn entirely by script: one panel with the flow Chemprop -> ranking of the 1,028 cohort ingredients -> result -> 15,511 ranked ingredients.

Every length is computed from the result files: the strip shows the positions of all 171 inhibitors
along the Chemprop ranking of the cohort, the two bars have lengths 1.47 : 6.01 and the tier bar has the shares of the ranked list.  The line under the two bars
gives the yield of the top 10% within the eight heavy-atom strata of the size-stratified AUROC (70 inhibitors found against 39.3 expected at random in the
same strata: 1.8 times random), because the cohort is mostly small ingredients and the 4.1-fold yield of the plain comparison is partly a size effect.  The
example structure is depicted by RDKit.  Needs RDKit (herbperturb environment).  Writes GraphicalAbstract.pdf, .eps, .png (300 dpi) and .tiff (600 dpi, LZW).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style_single as st  # noqa: E402

st.setup()
st.primary_check()
BLUE, GREY = st.COL["Chemprop"], "#BDBDBD"
TIERC = list(st.TIER.values())
est = st.est_table()
a0 = st.one(est, analysis="herb", model="Chemprop", metric="auroc")
N_COH, N_POS, AUROC = int(a0.n), int(a0.n_pos), float(a0.estimate)
bud = pd.read_csv(st.APP / "herb_budget.csv")
b10 = bud[np.isclose(bud.budget, 0.10)].iloc[0]
K10, FOUND10 = int(b10.assays), int(b10.found)
APK, APK_RAND = float(b10.assays_per_inhibitor), float(b10.random_assays_per_inhibitor)
scored = pd.read_csv(st.APP / "herb_cohort_scored.csv").sort_values(["score", "smiles"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
curve = pd.read_csv(st.FINAL / "herb_recovery_curves.csv")
assert len(scored) == N_COH and int(scored.label.sum()) == N_POS
assert int(scored.label.iloc[:K10].sum()) == FOUND10 == int(curve[(curve.model == "Chemprop") & (curve.k == K10)].cum_hits.iloc[0])
by_sim = pd.read_csv(st.APP / "unmeasured_tier_by_similarity.csv", index_col=0)
tier_n = by_sim.sum(axis=0)
N_RANK = int(tier_n.sum())
assert N_RANK == 15511 and (N_COH, N_POS, FOUND10, K10) == (1028, 171, 70, 103)
assert f"{APK:.1f}" == "1.5" and f"{APK_RAND:.1f}" == "6.0" and f"{AUROC:.3f}" == "0.869"
# yield of the top 10% within the eight heavy-atom strata of the size-stratified AUROC (style_single.size_matched_yield: the number of inhibitors expected at random is the
# sum over strata of (ingredients of the stratum among the top 103) x (inhibitor prevalence of the stratum); 39.3 expected, 1.8 times; two weight classes at 300 Da: 31.5, 2.2 times)
SM = st.size_matched_yield()
EXP_STRATA, ENR_STRATA = SM["exp_ha"], SM["enr_ha"]
assert SM["found"] == FOUND10 and SM["k"] == K10
N_POOL = len(pd.read_csv(st.BUNDLE / "pool.csv", usecols=["parent_inchikey"]))
assert N_POOL == 9726

W, H = 190.0, 88.0
fig = plt.figure(figsize=(W * st.MM, H * st.MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(8, 8 + H)
ax.axis("off")


def rbox(x, y, w, h, fc="white", ec="#9A9A9A", lw=0.8, r=1.6, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def txt(x, y, s, size=7.0, color="#222222", ha="center", va="center", bold=False, ls=1.15, z=6):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight="bold" if bold else "normal", linespacing=ls, zorder=z)


def arrow(x0, y0, x1, y1, color="#555555", lw=1.6, ms=10):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, zorder=3, shrinkA=0, shrinkB=0))


YB, HB = 10.0, 84.0                                          # blocks span y 10 to 94 (the axes show y 8 to 96)
X1, X2, X3, X4 = (1.0, 40.0), (47.0, 44.0), (97.0, 52.0), (155.0, 34.0)

# ---------------------------------------------------------------- block 1: example structure and Chemprop
rbox(X1[0], YB, X1[1], HB, fc="white", ec="#C8C8C8")
SMILES = "COc1ccc2c(c1OC)C(=O)OC2C1c2cc3c(cc2CCN1C)OCO3"
mol = Chem.MolFromSmiles(SMILES)
rdDepictor.SetPreferCoordGen(True)
rdDepictor.Compute2DCoords(mol)
conf = mol.GetConformer()
xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
c0 = xy - xy.mean(0)
_, _, vt = np.linalg.svd(c0, full_matrices=False)
for i, (x_, y_) in enumerate(c0 @ vt.T):
    conf.SetAtomPosition(i, (float(x_), float(y_), 0.0))
MOL = (4.0, 38.0, 64.0, 88.0)                                # extent in mm: x0, x1, y0, y1
PXW = 1400
PXH = int(round(PXW * (MOL[3] - MOL[2]) / (MOL[1] - MOL[0])))
d = rdMolDraw2D.MolDraw2DCairo(PXW, PXH)
o = d.drawOptions()
o.clearBackground = True
o.bondLineWidth = 4
o.padding = 0.05
o.fixedFontSize = int(round(7.4 * 25.4 / 72 * PXW / (MOL[1] - MOL[0])))   # atom labels of 7.4 pt at print size
d.DrawMolecule(mol)
d.FinishDrawing()
img = np.array(Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB"))
ax.imshow(img, extent=(MOL[0], MOL[1], MOL[2], MOL[3]), aspect="auto", zorder=3, interpolation="lanczos")
ax.set_xlim(0, W)
ax.set_ylim(8, 8 + H)
txt(X1[0] + X1[1] / 2, 60.6, "ingredient of the\nHERB database", size=6.6, color="#444444", ls=1.1)
arrow(X1[0] + X1[1] / 2, 56.8, X1[0] + X1[1] / 2, 51.5)
rbox(6.0, 36.0, 30.0, 14.0, fc=BLUE, ec=BLUE, r=1.4)
txt(21.0, 43.0, "Chemprop", size=10.0, color="white", bold=True)
txt(21.0, 25.0, f"trained on a pool of\n{N_POOL:,} public\nscreening compounds", size=7.2, color="#333333", ls=1.2)

arrow(41.4, 50.0, 46.6, 50.0)

# ---------------------------------------------------------------- block 2: ranking of the cohort with the 171 inhibitors
rbox(X2[0], YB, X2[1], HB, fc="white", ec="#C8C8C8")
cx = X2[0] + X2[1] / 2
txt(cx, 88.6, f"{N_COH:,} mostly small\nHERB ingredients", size=8.0, bold=True, ls=1.15)
txt(cx, 82.2, "ranked by Chemprop score", size=7.0, color="#444444")
SX0, SW, STOP, SH = 52.0, 13.0, 78.0, 55.0                  # strip: x0, width, top y, height (mm)
ax.add_patch(Rectangle((SX0, STOP - SH), SW, SH, fc="#F4F4F4", ec="#9A9A9A", lw=0.6, zorder=2))
ax.add_patch(Rectangle((SX0, STOP - SH * K10 / N_COH), SW, SH * K10 / N_COH, fc=st.PALE, ec=BLUE, lw=0.9, zorder=3))
pos = np.flatnonzero(scored.label.to_numpy() == 1)
for r_ in pos:
    y_ = STOP - (r_ + 0.5) / N_COH * SH
    ax.plot([SX0 + 0.8, SX0 + SW - 0.8], [y_, y_], color=BLUE if r_ < K10 else "#6F6F6F", lw=0.4, solid_capstyle="butt", zorder=4)
ytop10 = STOP - SH * K10 / N_COH
ax.plot([SX0 + SW + 1.0, SX0 + SW + 2.2, SX0 + SW + 2.2, SX0 + SW + 1.0], [STOP, STOP, ytop10, ytop10], color=BLUE, lw=0.9, zorder=4, solid_joinstyle="miter")
txt(SX0 + SW + 4.0, (STOP + ytop10) / 2 + 3.4, "Top 10%", size=7.6, bold=True, color=BLUE, ha="left")
txt(SX0 + SW + 4.0, (STOP + ytop10) / 2 - 0.6, f"{K10} ingredients\ntested", size=6.6, color="#333333", ha="left", ls=1.15)
txt(SX0 + SW + 4.0, STOP - SH * 0.5, "rank 1 at the top,\nrank 1,028 at\nthe bottom", size=6.3, color="#666666", ha="left", ls=1.15)
ax.plot([SX0, SX0 + 3.0], [STOP - SH - 4.2, STOP - SH - 4.2], color="#6F6F6F", lw=0.9)
txt(SX0 + 4.0, STOP - SH - 4.2, "each line: one inhibitor", size=6.4, color="#444444", ha="left")

arrow(91.4, 50.0, 96.6, 50.0)

# ---------------------------------------------------------------- block 3: result
rbox(X3[0], YB, X3[1], HB, fc="white", ec=BLUE, lw=1.6)
cx = X3[0] + X3[1] / 2
txt(cx, 84.5, f"{FOUND10} of {N_POS}", size=22, bold=True, color=BLUE)
txt(cx, 74.2, "inhibitors of cytochrome P450 3A4\n(CYP3A4) in the top 10% of the ranking", size=6.9, color="#333333", ls=1.2)
BX0, BMAX = 102.0, 40.0
scale = BMAX / APK_RAND
txt(BX0, 62.0, "Chemprop ranking", size=6.8, color="#333333", ha="left")
rbox(BX0, 55.0, APK * scale, 5.4, fc=BLUE, ec=BLUE, r=0.5)
txt(BX0 + APK * scale + 1.5, 57.7, f"{APK:.1f} assays per inhibitor", size=7.0, bold=True, color=BLUE, ha="left")
txt(BX0, 49.0, "Random order", size=6.8, color="#333333", ha="left")
rbox(BX0, 42.0, APK_RAND * scale, 5.4, fc=GREY, ec=GREY, r=0.5)
txt(BX0 + 1.5, 44.7, f"{APK_RAND:.1f} assays per inhibitor", size=7.0, bold=True, color="#333333", ha="left")
ax.plot([BX0 - 0.8, BX0 - 0.8], [40.5, 61.0], color="#555555", lw=0.8, zorder=4)
txt(cx, 37.6, f"{ENR_STRATA:.1f} times random yield", size=7.4, bold=True, color=BLUE)
txt(cx, 33.6, "within heavy-atom strata", size=6.6, color="#333333")
txt(cx, 26.0, f"AUROC {AUROC:.3f}", size=10.5, bold=True)
txt(cx, 22.1, "(area under the ROC curve)", size=6.4, color="#555555")
txt(cx, 15.8, f"in {N_COH:,} ingredients whose chemical parents\nare absent from the training data", size=6.4, color="#555555", ls=1.2)

arrow(149.4, 50.0, 154.6, 50.0)

# ---------------------------------------------------------------- block 4: ranked list of unmeasured ingredients
rbox(X4[0], YB, X4[1], HB, fc="white", ec="#C8C8C8")
cx = X4[0] + X4[1] / 2
txt(cx, 85.0, f"{N_RANK:,}\nunmeasured\ningredients ranked", size=8.0, bold=True, ls=1.2)
widths = [float(tier_n[t]) / N_RANK for t in ("Low", "Moderate", "High", "Very high")]
xx, TW = 158.0, 28.0
for w_, c in zip(widths, TIERC):
    ax.add_patch(Rectangle((xx, 55.0), w_ * TW, 7.0, fc=c, ec="#6A6A6A", lw=0.3, zorder=3))
    xx += w_ * TW
txt(cx, 66.0, "four score tiers", size=6.8, color="#444444")
for k, (nm_, c_) in enumerate(zip(("Low", "Moderate", "High", "Very high"), TIERC)):
    yk = 46.0 - k * 5.4
    ax.add_patch(Rectangle((159.5, yk - 1.7), 3.4, 3.4, fc=c_, ec="#6A6A6A", lw=0.3, zorder=3))
    txt(164.5, yk, nm_, size=7.0, color="#333333", ha="left")
txt(cx, 17.5, "for new assays", size=6.8, color="#444444")

st.save(fig, "GraphicalAbstract")
# text of the graphical abstract for the submission form, with the numbers of the figure
GA_TEXT = (f"Herbal ingredients can inhibit cytochrome P450 3A4 (CYP3A4) and raise exposure to co-administered drugs, yet few are measured. Chemprop, "
           f"trained on {N_POOL:,} screening compounds, ranks ingredients up to 500 Da. Among {N_COH:,} mostly small ingredients of the HERB database, "
           f"the top-ranked 10% held {FOUND10} of {N_POS} inhibitors (area under the ROC curve {AUROC:.3f}): {APK:.1f} assays per inhibitor against {APK_RAND:.1f} "
           f"in random order, {ENR_STRATA:.1f} times random yield within heavy-atom strata. The ranking sorts {N_RANK:,} unmeasured ingredients into four "
           f"score tiers.\n")
N_WORDS = len(GA_TEXT.split())
assert 50 <= N_WORDS <= 80, f"the text of the graphical abstract has {N_WORDS} words; the journal asks for 50 to 80"
(st.FIGDIR / "GraphicalAbstract_text.txt").write_text(GA_TEXT, encoding="utf-8", newline="\n")
print("saved GraphicalAbstract", N_WORDS, "words;", FOUND10, K10, round(APK, 2), round(APK_RAND, 2), round(AUROC, 3), N_RANK, [round(w, 3) for w in widths])
