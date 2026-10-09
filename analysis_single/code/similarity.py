"""Similarity of the evaluation structures to the training structures (annotation only; it never enters a score).

maxsim = maximum Tanimoto similarity (Morgan radius 2, 2048 bits) of a structure to the union of the five training sets of the split of the second
analysis (pool rows with role 'train' in at least one of role_s1 ... role_s5).  This equals the maximum over the five seeds of the per-seed maxima
that the second analysis stored in arch_v4/cache; it is recomputed here from the bundle and cached in analysis_single/cache/maxsim_union.npz
together with hashes of the pool (ids, SMILES, role columns) and of every set, and a cache that does not match the bundle is rebuilt.
tests/test_pipeline.py compares the result with arch_v4/cache (read only).

python code/similarity.py [--rebuild]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np

from common import CACHE, SEEDS, load_bundle, unmeasured_domain

SETS = ("ambe", "stress", "herb", "unmeasured_all")
FORMAT = 1


def _sha(parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def pool_signature(pool):
    cols = [pool.id.astype(str), pool.smiles.astype(str)] + [pool[f"role_s{s}"].astype(str) for s in SEEDS]
    return _sha("|".join(c) for c in cols)


def set_signature(smiles):
    return _sha(["|".join(map(str, smiles))])


def frame_of(bundle, k):
    return bundle["unmeasured"] if k == "unmeasured_all" else bundle[k]


def _fingerprints(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"SMILES {i} cannot be parsed: {s}")
        out.append(gen.GetFingerprint(m))
    return out


def compute(bundle, sets=SETS):
    from rdkit import DataStructs
    pool = bundle["pool"]
    train = np.any([pool[f"role_s{s}"].values == "train" for s in SEEDS], axis=0)
    train_fps = _fingerprints(pool.smiles.values[train])
    out = {}
    for k in sets:
        t0 = time.time()
        q = _fingerprints(frame_of(bundle, k).smiles.values)
        out[k] = np.array([max(DataStructs.BulkTanimotoSimilarity(f, train_fps)) for f in q])
        print(f"maxsim {k}: {len(q)} structures against {len(train_fps)} training structures ({time.time() - t0:.0f} s)", flush=True)
    return out


def maxsim_union_all(bundle=None, cache=CACHE, rebuild=False):
    """{set: array} for 'ambe', 'stress', 'herb' and 'unmeasured_all' (all 21,864 unmeasured structures, in the order of unmeasured.csv)."""
    bundle = bundle or load_bundle()
    f = cache / "maxsim_union.npz"
    meta = dict(format=FORMAT, pool=pool_signature(bundle["pool"]), sets={k: set_signature(frame_of(bundle, k).smiles.values) for k in SETS})
    if f.exists() and not rebuild:
        with np.load(f, allow_pickle=False) as z:
            if json.loads(str(z["_meta"])) == meta:
                print("similarity cache matches the bundle (pool, role columns and SMILES of every set)", flush=True)
                return {k: z[k] for k in SETS}
        print("similarity cache does not match the bundle: rebuilding", flush=True)
    out = compute(bundle)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, _meta=np.array(json.dumps(meta)), **out)
    return out


def maxsim_union(sets, bundle=None, cache=CACHE):
    """{set: array} for the sets asked for with 'unmeasured' = the structures up to 500 Da, in the order of the domain frame."""
    bundle = bundle or load_bundle()
    allm = maxsim_union_all(bundle, cache)
    res = {}
    for k in sets:
        if k == "unmeasured":
            res[k] = allm["unmeasured_all"][(bundle["unmeasured"].mw <= 500).values]
            assert len(res[k]) == len(unmeasured_domain(bundle["unmeasured"]))
        else:
            res[k] = allm[k]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    r = maxsim_union_all(rebuild=a.rebuild)
    print({k: (len(v), round(float(np.median(v)), 4)) for k, v in r.items()})
