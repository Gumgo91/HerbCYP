"""Run one expert over the five seeds and write long-format predictions.

An expert object needs: name, fit(bundle, store, tr, sel, seed) and predict(store, frame) -> probabilities.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SEEDS, SETS, eval_frames, split
from .metrics import summarize


def run_expert(expert, bundle, store, out_dir, seeds=SEEDS, sets=SETS, log=print):
    out_dir = Path(out_dir) / expert.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        todo = [n for n in sets if not (out_dir / f"seed{seed}_{n}.csv.gz").exists()]
        if not todo:
            log(f"[{expert.name}] seed {seed} already done")
            continue
        t0 = time.time()
        tr, sel, cal = split(bundle["pool"], seed)
        expert.fit(bundle, store, tr, sel, seed)
        frames = eval_frames(bundle, seed)
        for name in todo:
            p = np.asarray(expert.predict(store, frames[name], name), dtype=float)
            pd.DataFrame({"id": frames[name].id.values, "p": p}).to_csv(out_dir / f"seed{seed}_{name}.csv.gz", index=False)
            # the HERB cohort is scored and saved here; the log line below summarizes the other labeled sets
            if name != "herb" and ("label" in frames[name] or name == "calibration"):
                y = frames[name].label.values if "label" in frames[name] else frames[name].y.values
                m = summarize(y, p)
                log(f"[{expert.name}] seed {seed} {name:11s} AUROC {m['auroc']:.3f} AP {m['ap']:.3f} top10 {m['hits10']}/{m['k10']}")
        (out_dir / f"seed{seed}.done").write_text(f"{time.time() - t0:.1f}s sets={'+'.join(todo)}\n")
        log(f"[{expert.name}] seed {seed} finished in {time.time() - t0:.0f}s")
