"""Command line entry: python run_expert.py --expert chemprop --bundle bundle --cache cache/feats.npz --out results [--seeds 1 2] [--limit N]

--limit N trains on the first N training rows and scores only the first N rows of each evaluation set (smoke tests).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hc.data import SEEDS, SETS, load_bundle  # noqa: E402
from hc.feats import build_store  # noqa: E402
from hc.runner import run_expert  # noqa: E402

# The registry lists the four models of the paper.
REGISTRY = {
    "maplight_cb": ("experts.e_catboost", "MapLightCB", {}),
    "tabpfn_desc": ("experts.e_tabpfn", "TabPFNDesc", {}),
    "chemprop": ("experts.e_chemprop", "ChempropExpert", {}),
    "rf": ("experts.e_rf", "RandomForestExpert", {}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--bundle", default=str(HERE / "bundle"))
    ap.add_argument("--cache", default=str(HERE / "cache" / "feats.npz"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sets", nargs="*", default=None, choices=SETS, help="subset of calibration ambe stress herb unmeasured")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    manifest = Path(a.bundle) / "SPLIT_V2_MANIFEST.json"
    if not manifest.exists():
        raise SystemExit("bundle/SPLIT_V2_MANIFEST.json is missing: refusing to train without the split manifest")
    if hashlib.sha256((Path(a.bundle) / "pool.csv").read_bytes()).hexdigest() != json.loads(manifest.read_text(encoding="utf-8"))["pool_sha256"]:
        raise SystemExit("bundle/pool.csv differs from the split of SPLIT_V2_MANIFEST.json")
    bundle = load_bundle(a.bundle)
    if a.limit:
        pool = bundle["pool"]
        keep = np.r_[np.where(pool.role_s1 == "train")[0][:a.limit], np.where(pool.role_s1 != "train")[0][:a.limit]]
        bundle["pool"] = pool.iloc[np.sort(keep)].reset_index(drop=True)
        for k in ("ambe", "stress", "herb", "unmeasured"):
            bundle[k] = bundle[k].iloc[:a.limit].reset_index(drop=True)
    Path(a.cache).parent.mkdir(parents=True, exist_ok=True)
    store = build_store(bundle, a.cache)
    mod, cls, kw = REGISTRY[a.expert]
    if a.device:
        kw = dict(kw, device=a.device)
    expert = getattr(importlib.import_module(mod), cls)(**kw)
    expert.name = a.expert
    expert.out_dir = Path(a.out) / a.expert   # lets an expert write per-seed diagnostics (chemprop: seed{k}_selection.json) next to its predictions
    run_expert(expert, bundle, store, a.out, seeds=a.seeds, sets=a.sets or SETS, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
