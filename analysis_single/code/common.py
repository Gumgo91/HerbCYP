"""Shared helpers of the single-model analysis (Chemprop as the primary model).

The analysis re-summarizes the prediction files of the second analysis (ROOT/arch_v4/results/<model>/seed{k}_{set}.csv.gz) for the four stand-alone
models Chemprop, MapLight-style CatBoost, TabPFN on descriptors and the random forest, plus the two descriptor baselines (heavy-atom count, Crippen
logP).  Nothing is trained or tuned here, and nothing under ROOT/arch_v4, ROOT/arch_v5 or ROOT/analysis_v3 is written.

Design (unchanged from the second analysis): cohort = HERB ingredients up to 500 Da, paired scaffold-cluster bootstrap with 2,000 draws (seed 2026,
analysis_v3/stats_v3.py bootstrap_panel), scores = mean over the five seeds of the raw probabilities.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True      # never leave __pycache__ folders in read-only trees (arch_v4, analysis_v3, src)

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]           # ROOT/analysis_single
ROOT = HERE.parent                                   # ROOT = HerbCYP
V4 = ROOT / "arch_v4"
BUNDLE = V4 / "bundle"
RESULTS = V4 / "results"                             # <model>/seed{k}_{set}.csv.gz of the second analysis (read only)
V3_RESULTS = ROOT / "analysis_v3" / "results"        # split-independent annotations (read only)
CACHE = HERE / "cache"
FINAL = HERE / "results" / "final"
APP = FINAL / "app"
LOGS = HERE / "results" / "logs"
CHEMBL_SRC = ROOT / "data" / "raw" / "chembl" / "cyp3a4_activities.jsonl"

sys.path.insert(0, str(ROOT / "analysis_v3"))
sys.path.insert(0, str(ROOT / "src"))
from stats_v3 import bootstrap_panel  # noqa: E402  (ROOT/analysis_v3/stats_v3.py, read only)

SEEDS = [1, 2, 3, 4, 5]
DOMAIN = 500.0                                       # application domain (Da)
PRIMARY = "Chemprop"
# model label -> folder of the second analysis; the label is the column name in every table
MODEL_DIR = {"Chemprop": "chemprop", "MapLight": "maplight_cb", "TabPFN-desc": "tabpfn_desc", "RF": "rf"}
MODEL_NAME = {"Chemprop": "Chemprop", "MapLight": "MapLight-style CatBoost", "TabPFN-desc": "TabPFN on descriptors", "RF": "Random forest",
              "Size": "Heavy-atom count", "LogP": "Crippen logP"}
TRAINED = list(MODEL_DIR)                            # candidates of the selection rule
REF = ["MapLight", "TabPFN-desc", "RF", "Size", "LogP"]       # references of the paired differences
ALL_MODELS = [PRIMARY] + REF
SUBSET_MODELS = [PRIMARY, "MapLight", "TabPFN-desc", "RF"]    # models of the stratum and sensitivity analyses
EDGES = [-np.inf, 7, 9, 10, 11, 13, 16, 20, np.inf]          # heavy-atom bins of the size-stratified AUROC
ALLOW_OLD_ENV = "HERBCYP_ALLOW_OLD_RESULTS"


class StaleResultsError(RuntimeError):
    """Prediction files older than the frozen split, or ids that do not match the frames of the current split."""


# ------------------------------------------------------------------ bundle and predictions
def load_bundle(d=BUNDLE):
    d = Path(d)
    return {k: pd.read_csv(d / f"{k}.csv") for k in ("pool", "ambe", "stress", "herb", "unmeasured")}


def unmeasured_domain(df):
    return df[df.mw <= DOMAIN].reset_index(drop=True)


def check_frozen_split(bundle_dir=BUNDLE):
    """The pool must be the one whose hash is recorded in SPLIT_V2_MANIFEST.json of the second analysis."""
    d = Path(bundle_dir)
    want = json.loads((d / "SPLIT_V2_MANIFEST.json").read_text(encoding="utf-8"))["pool_sha256"]
    got = hashlib.sha256((d / "pool.csv").read_bytes()).hexdigest()
    if got != want:
        raise ValueError(f"{d / 'pool.csv'} (sha256 {got[:12]}) is not the frozen pool of SPLIT_V2_MANIFEST.json ({want[:12]})")
    return d / "pool.csv"


def check_fresh(files, bundle_dir=BUNDLE):
    """Refuse prediction files older than the frozen pool (leftovers of the first analysis)."""
    if os.environ.get(ALLOW_OLD_ENV) == "1":
        return
    t_ref = check_frozen_split(bundle_dir).stat().st_mtime
    old = [str(f) for f in files if Path(f).stat().st_mtime < t_ref]
    if old:
        raise StaleResultsError(f"{len(old)} prediction files are older than the frozen pool, e.g. {old[:3]}")


def read_pred(path, ids):
    """Probabilities of one prediction file for the given ids (all ids present, probabilities finite and within [0, 1])."""
    d = pd.read_csv(path)
    if not {"id", "p"} <= set(d.columns):
        raise ValueError(f"{path}: columns id and p expected, found {list(d.columns)}")
    if d.id.duplicated().any():
        raise ValueError(f"{path}: duplicated ids")
    ids = np.asarray(ids)
    miss = ~pd.Index(ids).isin(d.id)
    if miss.any():
        raise StaleResultsError(f"{path}: {int(miss.sum())} of {len(ids)} ids of the current frame are missing")
    p = d.set_index("id").loc[ids, "p"].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError(f"{path}: probabilities must be finite and within [0, 1]")
    return p


def pred_file(label, sset, seed):
    return RESULTS / MODEL_DIR[label] / f"seed{seed}_{sset}.csv.gz"


def sets_of(label):
    """Evaluation sets a model has prediction files for (TabPFN on descriptors has none for the unmeasured ingredients)."""
    return [k for k in ("ambe", "stress", "herb", "unmeasured") if all(pred_file(label, k, s).exists() for s in SEEDS)]


def seed_mean(label, sset, ids):
    """Mean over the five seeds of the raw probabilities of one model for the rows with the given ids."""
    return np.mean([read_pred(pred_file(label, sset, s), ids) for s in SEEDS], axis=0)


def required_files(include_herb=True):
    """Every prediction file the analysis reads."""
    files = []
    for lab in TRAINED:
        sets = ["ambe", "stress"] + (["herb"] if include_herb else [])
        if lab != "TabPFN-desc":
            sets = sets + ["unmeasured"]
        files += [pred_file(lab, k, s) for k in sets for s in SEEDS]
    return files


def verify_inputs(include_herb=True):
    """Stop before any table is written if a file is missing, older than the frozen pool, or does not carry the ids of the current frames."""
    check_frozen_split()
    files = required_files(include_herb)
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(files)} required prediction files are missing, e.g. {missing[:5]}")
    check_fresh(files)
    b = load_bundle()
    frames = {"ambe": b["ambe"], "stress": b["stress"], "herb": b["herb"], "unmeasured": unmeasured_domain(b["unmeasured"])}
    for f in files:
        sset = f.name[:-len(".csv.gz")].split("_")[-1]
        read_pred(f, frames[sset].id.values)
    print(f"inputs checked: {len(files)} prediction files present, newer than the frozen pool, with the ids of the current frames", flush=True)
    return files


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ evaluation frames
def attach_scores(frame, sset, models=TRAINED, sim=None):
    """Score columns (one per model), maxsim (if given), canonical_smiles, Size, LogP and size_bin, in the column order of the second analysis."""
    f = frame.copy()
    for lab in ("Chemprop", "RF"):
        if lab in models:
            f[lab] = seed_mean(lab, sset, f.id.values)
    if sim is not None:
        f["maxsim"] = sim
    f["canonical_smiles"] = f.smiles
    f["Size"], f["LogP"] = f.heavy.values.astype(float), f.logp.values
    for lab in ("MapLight", "TabPFN-desc"):
        if lab in models:
            f[lab] = seed_mean(lab, sset, f.id.values)
    f["size_bin"] = pd.cut(f.heavy, EDGES).astype(str)
    return f


def panel(frame, tag, out_est, out_pair, models, strata=False, quiet=False):
    """Estimates with scaffold-cluster bootstrap intervals and paired differences of the primary model (2,000 draws, seed 2026)."""
    est, pair = bootstrap_panel(frame, {m: m for m in models}, group_col="scaffold", primary=PRIMARY, strata_col="size_bin" if strata else None)
    est.insert(0, "analysis", tag)
    pair.insert(0, "analysis", tag)
    out_est.append(est)
    out_pair.append(pair)
    if not quiet:
        a = est[(est.model == PRIMARY) & (est.metric == "auroc")].iloc[0]
        print(f"{tag:34s} n={a.n:5d} pos={a.n_pos:4d} clusters={a.n_clusters:4d} {PRIMARY} AUROC {a.estimate:.3f} ({a.ci_lo:.3f}, {a.ci_hi:.3f})", flush=True)
    return est, pair
