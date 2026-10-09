"""Figure 6: the highest-ranked unmeasured ingredients (structures, Chemprop scores, natural-product likeness, herbs).  Needs RDKit (herbperturb environment).

Read from results/final/app/unmeasured_ranked.csv.gz of the single-model analysis; names, ranks, scores, tiers, similarity, natural-product likeness and herbs
come from the file.  The likeness score is printed for every panel so that a reader sees which top-ranked structures are not natural-product-like (score below 0;
some HERB records may be annotation errors).
Rows that HERB labels with a name that does not match the structure (column name_flag) are skipped.  Long names are wrapped over three lines
and cut with an ellipsis (the complete name is in the ranked table).  The methylenedioxyphenyl group is drawn in orange where it is present.
The names of the ranked table are the HERB names without the characters outside the ASCII set (app_final.py, clean_name; arch_v4/bundle/unmeasured.csv holds the names as
recorded): the note under the panels says so, and the script checks the shown names against the recorded ones (in the ranks shown only the name of rank 10 loses
characters: its (±) reads as empty parentheses)."""
from __future__ import annotations

import io
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style_single as st  # noqa: E402

st.setup()
st.primary_check()
rk = pd.read_csv(st.APP / "unmeasured_ranked.csv.gz")
TIERC = dict(st.TIER)
NSHOW = int(sys.argv[1]) if len(sys.argv) > 1 else 12
assert rk["rank"].is_monotonic_increasing and rk.score.is_monotonic_decreasing
flagged = rk[rk["name_flag"].notna()]
top = rk[rk["name_flag"].isna()].head(NSHOW).reset_index(drop=True)
print("ranks shown:", top["rank"].tolist())
print("flagged entries (skipped, not in the top rows):", flagged[["rank", "name"]].values.tolist())
assert int(flagged["rank"].min()) > int(top["rank"].max()), "a flagged entry lies among the ranks shown"
assert (top.parent_in_evaluation_sets == False).all(), "a shown ingredient shares a parent with a cohort or Ambe ingredient"  # noqa: E712

MDP = Chem.MolFromSmarts("c1cc2OCOc2cc1")
HAS_MDP = [Chem.MolFromSmiles(x).HasSubstructMatch(MDP) for x in top.smiles]
print(f"{sum(HAS_MDP)} of {len(top)} shown ingredients contain a methylenedioxyphenyl group (ranks {[int(r) for r, h in zip(top['rank'], HAS_MDP) if h]})")
print("applicability of the shown ingredients:", top.applicability.value_counts().to_dict(), "; similarity range", round(top.maxsim_train.min(), 2), round(top.maxsim_train.max(), 2))


def display_name(raw: str) -> str:
    raw = str(raw).strip()
    if not raw.isascii():
        print("warning: non-ASCII HERB name", repr(raw))
    return raw[:1].upper() + raw[1:]


# names as recorded in HERB (inputs of the second analysis) against the names of the ranked table, which have lost every character outside the ASCII set
RECORDED = pd.read_csv(st.BUNDLE / "unmeasured.csv", usecols=["id", "name"]).set_index("id").name
assert top.id.isin(RECORDED.index).all(), "a shown ingredient is not in the input table of the unmeasured ingredients"
LOST = [int(r) for r, i in zip(top["rank"], top.id) if not str(RECORDED.loc[i]).isascii()]
print("shown names that lost characters outside the ASCII set (ranks):", LOST)
assert LOST == [10], f"the note under the panels was written for the loss in the name of rank 10 only, not for ranks {LOST}"
_lost_rows = top[top["rank"].isin(LOST)]
assert all(str(RECORDED.loc[i]).replace("±", "").strip() == str(n).strip() for i, n in zip(_lost_rows.id, _lost_rows["name"])), \
    "the name of rank 10 differs from the recorded name by more than the missing (±)"


def wrap_name(nm: str, width: int = 33, lines: int = 3) -> list[str]:
    w = textwrap.wrap(nm, width=width, break_long_words=True, break_on_hyphens=True)
    if len(w) > lines:
        w = w[:lines]
        w[-1] = w[-1][: width - 1].rstrip("-, ") + "…"
    return w


PX_PER_MM = 27                           # 686 dpi at print size
SW_MM, SH_MM = 44.0, 26.0                # structure panel on the page; the bitmap has the same aspect ratio


def draw(smi, w=int(SW_MM * PX_PER_MM), h=int(SH_MM * PX_PER_MM)):
    mol = Chem.MolFromSmiles(smi)
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.clearBackground = True
    o.bondLineWidth = 3
    o.padding = 0.08
    o.fixedFontSize = int(round(7.4 * 25.4 / 72 * PX_PER_MM))      # atom labels of 7.4 pt at print size (their subscripts, for example in NH2, are smaller)
    atoms = sorted({a for m in mol.GetSubstructMatches(MDP) for a in m})
    bonds = [b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    orange = (0.98, 0.83, 0.55)
    d.DrawMolecule(mol, highlightAtoms=atoms, highlightBonds=bonds, highlightAtomColors={a: orange for a in atoms},
                   highlightBondColors={b: orange for b in bonds})
    d.FinishDrawing()
    return np.array(Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB"))


ncol, nrow = 4, int(np.ceil(NSHOW / 4))
cw, ch = 46.0, 53.0
W, H = 190.0, 13 + ch * nrow
fig = plt.figure(figsize=(W * st.MM, H * st.MM))
TRUNC = []
for i, r in top.iterrows():
    row, col = divmod(i, ncol)
    x0 = 3 + col * (cw + 1.0)
    y0 = H - 1.5 - (row + 1) * ch                        # bottom of the cell (the page keeps 13 mm at the bottom for three lines of notes)
    ax = fig.add_axes([x0 / W, (y0 + ch - 1.0 - SH_MM) / H, (cw - 2) / W, SH_MM / H])
    ax.imshow(draw(r.smiles))
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=TIERC[r.tier], lw=1.3, clip_on=False))
    nm = display_name(r["name"])
    lines = wrap_name(nm)
    if lines[-1].endswith("…"):
        TRUNC.append((int(r["rank"]), nm))
    ytxt = y0 + ch - 1.0 - SH_MM - 1.6                   # top of the text block
    fig.text((x0 + 0.5) / W, ytxt / H, f"{int(r['rank'])}. {lines[0]}", fontsize=6.3, fontweight="bold", va="top", ha="left")
    for k, ln in enumerate(lines[1:], start=1):
        fig.text((x0 + 0.5) / W, (ytxt - 2.55 * k) / H, ln, fontsize=6.3, fontweight="bold", va="top", ha="left")
    yy = ytxt - 2.55 * 3 - 0.4
    fig.text((x0 + 0.5) / W, yy / H, f"Score {r.score:.2f}  |  {r.tier}", fontsize=6.3, va="top", ha="left", color="#222222")
    fig.text((x0 + 0.5) / W, (yy - 3.0) / H, f"Similarity to training {r.maxsim_train:.2f}", fontsize=6.2, va="top", ha="left", color="#555555")
    fig.text((x0 + 0.5) / W, (yy - 6.0) / H, "NP-likeness " + f"{r.np:.2f}".replace("-", "−"), fontsize=6.2, va="top", ha="left", color="#555555")
    herbs = str(r.herbs) if isinstance(r.herbs, str) else ""
    first = herbs.split("; ")[0] if herbs else ""
    one = first if r.n_herbs <= 1 else f"{first} and {int(r.n_herbs) - 1} more"
    herbs = one if len(one) <= 34 else (first[:32] + "…" + (f"\nand {int(r.n_herbs) - 1} more" if r.n_herbs > 1 else ""))
    fig.text((x0 + 0.5) / W, (yy - 9.0) / H, herbs, fontsize=6.2, va="top", ha="left", color="#555555", linespacing=1.05)
empty_par = [int(r) for r, n in zip(top["rank"], top["name"]) if "()" in str(n)]
assert empty_par == LOST, f"empty parentheses in the names of ranks {empty_par}, lost characters in the names of ranks {LOST}"
NOTES = [
    "Orange: methylenedioxyphenyl group, where present. Frame color: score tier. Similarity: maximum Tanimoto similarity to the training structures.",
    "NP-likeness: natural-product likeness score (below 0: not natural-product-like). Names are the HERB names without characters outside the ASCII set (Greek letters, ±, primes).",
    "The herb shown is the first herb that lists the ingredient, in the order of the HERB identifiers."
    + (f" The empty parentheses in the name of rank {', '.join(map(str, empty_par))} stand for (±)." if empty_par else ""),
]
note_texts = [fig.text(3 / W, y / H, s, fontsize=6.0, color="#555555", va="bottom") for y, s in zip((7.8, 4.9, 2.0), NOTES)]
fig.canvas.draw()
for t in note_texts:                                   # every note line stays inside the page (2 mm margin)
    assert t.get_window_extent().x1 <= fig.bbox.x1 - 2 / 25.4 * fig.dpi, f"note line too long: {t.get_text()!r}"
if TRUNC:
    print("names cut with an ellipsis in the figure:", TRUNC)
st.save(fig, "Fig6", eps=False)
(st.FIGDIR / "Fig6.eps").unlink(missing_ok=True)      # the structures are bitmaps, so an EPS would be about 49 MB; Fig. 6 is submitted as TIFF (and PDF)
print("Fig6 saved")
