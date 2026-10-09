"""Fig. 2 (study flow chart), drawn entirely by script with matplotlib.

Four columns joined by arrows: three source tables with their counts, the four models and two reference scores, the evaluation, and the outputs.
The training pool leads into the frame of the models by an arrow labeled "fit"; the evaluation cohort and the unmeasured ingredients join in one arrow labeled
"score" (the trained models score them; TabPFN on descriptors has no predictions for the unmeasured ingredients, which its box states).
No generative AI tool and no image editor is involved: every box, line and word is a primitive of this script, and the counts of the three source tables
are read from the input and result files (training pool, evaluation cohort, ranked list) when the figure is drawn.  Colors are those of the other figures
(style_single.py): the strong blue of the primary model, one color per model, grey for the two reference scores, pale blue for the cohort and the outputs.

The drawing uses a 1600-unit canvas for 190 mm (one unit = 0.119 mm = 0.34 pt): box text 24 units (8.1 pt), column headings 30 units (10.1 pt), captions
22 units (7.4 pt), the second caption of the primary model 21 units (7.1 pt).  Text is measured with the renderer and must stay at least 16 units inside its box.
Writes Fig2.pdf, Fig2.eps (vector), Fig2.png (300 dpi) and Fig2.tiff (600 dpi, LZW) in manuscript/final_v5/figures through style_single.save().
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style_single as st  # noqa: E402

st.setup()
st.primary_check()

# ---------------------------------------------------------------- numbers of the source tables, read from the files
est = st.est_table()
_a = st.one(est, analysis="herb", model="Chemprop", metric="auroc")
N_COH, N_POS = int(_a.n), int(_a.n_pos)                                          # evaluation cohort (at most 500 Da) and its inhibitors
N_POOL = len(pd.read_csv(st.BUNDLE / "pool.csv", usecols=["parent_inchikey"]))   # training pool
N_RANK = len(pd.read_csv(st.APP / "unmeasured_ranked.csv.gz", usecols=["rank"]))  # ranked unmeasured ingredients
assert (N_POOL, N_COH, N_POS, N_RANK) == (9726, 1028, 171, 15511), "a count of the source tables differs from the manuscript"
_pred = st.ROOT / "arch_v4" / "results"                                          # prediction files of the four models (the data package restores this layout with setup_layout.py)
assert not list((_pred / "tabpfn_desc").glob("seed*_unmeasured.csv.gz")) and all(len(list((_pred / d).glob("seed*_unmeasured.csv.gz"))) == 5 for d in ("chemprop", "maplight_cb", "rf")), \
    "TabPFN on descriptors scores the unmeasured ingredients, or another model does not: revise the box text of the TabPFN row and the legend of Fig. 2"

# ---------------------------------------------------------------- canvas (units = pixels of a 1600 px wide page of 190 mm)
W, H = 1600, 775
W_MM = 190.0
PT = W_MM / 25.4 * 72 / W                                       # points per unit
fig = plt.figure(figsize=(W_MM / 25.4, W_MM / 25.4 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")

INK, LINE = "#222222", "#555555"
GREYBOX = "#F4F4F4"
TXT = []                                                         # (artist, left limit, right limit) for the fit check


def box(x, y, w, h, fc, ec, lw, rx=18, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rx}", fc=fc, ec=ec, lw=lw * PT, zorder=z)
    ax.add_patch(p)
    return p


def label(cx, y, s, size, bold=False, italic=False, color=INK, limits=None):
    t = ax.text(cx, y, s, fontsize=size * PT, fontweight="bold" if bold else "normal", fontstyle="italic" if italic else "normal", color=color,
                ha="center", va="baseline", zorder=6)
    if limits:
        TXT.append((t, *limits))
    return t


def line(x0, y0, x1, y1, w=2):
    ax.plot([x0, x1], [y0, y1], color=LINE, lw=w * PT, solid_capstyle="butt", zorder=2)


def head(x_tail, x_tip, y, half=12):
    ax.add_patch(Polygon([(x_tail, y - half), (x_tip, y), (x_tail, y + half)], closed=True, fc=LINE, ec="none", zorder=3))


def thick_arrow(x0, x1, y):
    line(x0, y, x1 - 18, y, w=5)
    head(x1 - 20, x1, y)


# ---------------------------------------------------------------- column headings and captions
for cx, s in ((195, "Source tables"), (625, "Models compared"), (1035, "Evaluation"), (1435, "Outputs")):
    label(cx, 70, s, 30, bold=True)
label(625, 110, "four trained models, five seeds each", 22, italic=True)
label(1035, 110, f"on the {N_COH:,} cohort ingredients", 22, italic=True)

YC = 440                                                         # vertical centre of the figure body

# ---------------------------------------------------------------- column 1: source tables (x 30 to 350)
BX0, BX1 = 30, 348
BUS = 368                                                        # x of the vertical line that joins the cohort and the unmeasured ingredients
src = [(190, GREYBOX, LINE, 1.5, [("Training pool", True), (f"{N_POOL:,} public", False), ("screening compounds", False)]),
       (375, st.PALE, st.ACCENT, 2, [("Evaluation cohort", True), (f"{N_COH:,} HERB ingredients", False), (f"{N_POS} inhibitors", False)]),
       (560, GREYBOX, LINE, 1.5, [("Unmeasured ingredients", True), (f"{N_RANK:,} HERB ingredients", False)])]
centres = []
for y0, fc, ec, lw, rows in src:
    box(BX0, y0, BX1 - BX0, 130, fc, ec, lw)
    base = y0 + 65 + 8 - 16 * (len(rows) - 1)                    # block of lines centred in the box (32 units between lines)
    for k, (s, b) in enumerate(rows):
        label((BX0 + BX1) / 2, base + 32 * k, s, 24, bold=b, limits=(BX0, BX1))
    centres.append(y0 + 65)
# the training pool fits the models ("fit"); the cohort and the unmeasured ingredients are scored by them ("score"): two arrows into the frame
FX_IN = 440                                                      # left edge of the models frame
YFIT, YSCORE = centres[0], (centres[1] + centres[2]) / 2
line(BX1, centres[0], BUS, centres[0])
thick_arrow(BUS, FX_IN, YFIT)
ARROW_LABELS = [label((BUS + FX_IN) / 2, YFIT - 22, "fit", 22, italic=True, color=LINE)]
line(BX1, centres[1], BUS, centres[1])
line(BX1, centres[2], BUS, centres[2])
line(BUS, centres[1], BUS, centres[2])
thick_arrow(BUS, FX_IN, YSCORE)
ARROW_LABELS.append(label((BUS + FX_IN) / 2, YSCORE - 22, "score", 22, italic=True, color=LINE))

# ---------------------------------------------------------------- column 2: models compared (frame x 440 to 810)
FX0, FX1 = 440, 810
box(FX0, 130, FX1 - FX0, 625, "#FFFFFF", LINE, 1.5, rx=22)
MX0, MW = 460, 330                                               # inner boxes
box(MX0, 150, MW, 112, st.COL["Chemprop"], st.COL["Chemprop"], 0, rx=16, z=3)
label(MX0 + MW / 2, 188, "Chemprop", 24, bold=True, color="#FFFFFF", limits=(MX0, MX0 + MW))
label(MX0 + MW / 2, 218, "(primary model)", 24, bold=True, color="#FFFFFF", limits=(MX0, MX0 + MW))
label(MX0 + MW / 2, 248, "selected on development sets", 21, italic=True, color="#FFFFFF", limits=(MX0, MX0 + MW))
bars = [("MapLight", 276), ("TabPFN-desc", 358), ("RF", 440), ("Size", 576), ("LogP", 658)]
for key, y0 in bars:
    p = box(MX0, y0, MW, 70, "#FFFFFF", LINE, 1, rx=16, z=3)
    stripe = plt.Rectangle((MX0, y0), 14, 70, fc=st.COL[key], ec="none", zorder=4)
    ax.add_patch(stripe)
    stripe.set_clip_path(p)
    if key == "TabPFN-desc":                                     # no predictions for the unmeasured ingredients: the second line says so
        label(MX0 + 14 + (MW - 14) / 2, y0 + 30, st.LAB[key], 24, limits=(MX0 + 14, MX0 + MW))
        label(MX0 + 14 + (MW - 14) / 2, y0 + 58, "scores the cohort only", 21, italic=True, color=LINE, limits=(MX0 + 14, MX0 + MW))
    else:
        label(MX0 + 14 + (MW - 14) / 2, y0 + 43, st.LAB[key], 24, limits=(MX0 + 14, MX0 + MW))
line(MX0, 538, MX0 + MW, 538, w=1.5)
label(MX0 + MW / 2, 566, "Reference scores", 22, italic=True)

# ---------------------------------------------------------------- column 3: evaluation (x 890 to 1180), a bracket on each side
EX0, EX1 = 890, 1180
ev = [(250, ["AUROC"]), (350, ["Average precision (AP)"]), (450, ["Testing yield"]), (550, ["Paired scaffold-cluster", "bootstrap"])]
ev_c = []
for y0, rows in ev:
    box(EX0, y0, EX1 - EX0, 80, GREYBOX, LINE, 1.5)
    c = y0 + 40
    base = [c + 8] if len(rows) == 1 else [c - 8, c + 24]
    for s, b in zip(rows, base):
        label((EX0 + EX1) / 2, b, s, 24, limits=(EX0, EX1))
    ev_c.append(c)
LB, RB = 860, 1210
line(LB, ev_c[0], LB, ev_c[-1])
line(RB, ev_c[0], RB, ev_c[-1])
for yc in ev_c:
    line(LB, yc, EX0, yc)
    line(EX1, yc, RB, yc)
thick_arrow(FX1, LB - 2, YC)
thick_arrow(RB, 1258, YC)

# ---------------------------------------------------------------- column 4: outputs (x 1290 to 1580)
OX0, OX1 = 1290, 1580
outs = [(245, ["Score tiers"]), (385, ["Testing budgets"]), (525, ["Ranked list of", f"{N_RANK:,} ingredients"])]
out_c = []
for y0, rows in outs:
    box(OX0, y0, OX1 - OX0, 110, st.PALE, st.ACCENT, 2)
    c = y0 + 55
    base = [c + 8] if len(rows) == 1 else [c - 8, c + 24]
    for s, b in zip(rows, base):
        label((OX0 + OX1) / 2, b, s, 24, limits=(OX0, OX1))
    out_c.append(c)
OB = 1260
line(OB, out_c[0], OB, out_c[-1])
for yc in out_c:
    line(OB, yc, OX0, yc)

# ---------------------------------------------------------------- fit check: every line of box text at least 16 units inside its box
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()
for t, lo, hi in TXT:
    bb = t.get_window_extent(rend)
    (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    assert x0 >= lo + 16 and x1 <= hi - 16, f"text too wide for its box: {t.get_text()!r} {x0:.0f}..{x1:.0f} in {lo}..{hi}"
for t in ARROW_LABELS:                                           # the two arrow labels stay between the bus line and the frame of the models
    bb = t.get_window_extent(rend)
    (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    assert x0 >= BUS + 6 and x1 <= FX_IN - 6, f"arrow label too wide: {t.get_text()!r} {x0:.0f}..{x1:.0f} in {BUS}..{FX_IN}"
for t in ax.texts:
    assert t.get_fontsize() >= 6.0, f"text below 6 pt: {t.get_text()!r}"

st.save(fig, "Fig2")
print("Fig2 saved", (N_POOL, N_COH, N_POS, N_RANK))
