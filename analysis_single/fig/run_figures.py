"""Regenerate every figure of the single-model analysis in manuscript/final_v5/figures.

Run from ROOT/analysis_single/fig with the herbperturb interpreter (Fig. 6 and the graphical abstract need RDKit):

    set PYTHONIOENCODING=utf-8
    python run_figures.py   (an environment with RDKit, matplotlib, pandas and Pillow)

All figures, including the flow chart of Fig. 2 (fig2_flowchart.py), are drawn by these scripts from the result files; no figure is made or altered with a
generative AI tool or an image editor.  The checks (check_figures.py) run with the default Python (PyMuPDF).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
STEPS = ["fig_data.py", "fig2_flowchart.py", "fig_perf.py", "fig_robust.py", "fig_app.py", "fig_gallery.py", "fig_graphical_abstract.py"]
for s in STEPS:
    print("==", s, flush=True)
    r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, env=ENV)
    if r.returncode:
        sys.exit(f"{s} failed with exit code {r.returncode}")
print("all figures written")
