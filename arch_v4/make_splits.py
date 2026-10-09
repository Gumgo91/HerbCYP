"""Corrected scaffold splits of the training pool (second analysis, see PROTOCOL_REDO_SPLIT.md).

The first analysis used a deterministic largest-group-first split, which put all 264 ring-free structures into the selection subset in every
seed and the largest groups into the same subset in every seed. The corrected split

  * forms groups = connected components of structures that share a stereo-free Murcko scaffold or a chemical parent; ring-free structures
    (empty scaffold) are grouped by Butina clustering on Morgan fingerprints (radius 2, 2,048 bits, Tanimoto similarity >= 0.5) and parents,
    so that they are no longer a single group;
  * assigns groups to training / selection / calibration (70 / 15 / 15 %) at random, one random generator per seed, choosing among 5,000 random
    assignments the one whose subset sizes are closest to the targets (label-free criterion).

The group id of every structure is stored in the column gid (used for the grouped cross-validation of the gate).

Usage: python make_splits.py            writes bundle/pool.csv (role_s1..role_s5) and bundle/SPLIT_V2_MANIFEST.json
       python make_splits.py --dry-run  prints the quality report only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
RATIOS = (0.70, 0.15, 0.15)
ROLES = ("train", "selection", "calibration")
N_TRIALS = 5000
ACYCLIC_SIMILARITY = 0.5


def groups_of(pool: pd.DataFrame) -> np.ndarray:
    """Group id per row (union of shared scaffold, shared parent and, for ring-free structures, Butina cluster)."""
    n = len(pool)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    scaf = pool.scaffold.fillna("").astype(str).values
    ring_free = scaf == ""
    first = {}
    for i in range(n):
        for key in ((("s", scaf[i]),) if not ring_free[i] else ()) + (("p", pool.parent_inchikey.iloc[i]),):
            if key in first:
                union(i, first[key])
            else:
                first[key] = i
    idx = np.where(ring_free)[0]
    fps = []
    for i in idx:
        m = Chem.MolFromSmiles(pool.smiles.iloc[i])
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    dists = []
    for a in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[a], fps[:a])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(fps), 1.0 - ACYCLIC_SIMILARITY, isDistData=True)
    for c in clusters:
        for k in c[1:]:
            union(int(idx[c[0]]), int(idx[k]))
    roots = np.array([find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def assign(sizes: np.ndarray, seed: int):
    n_total = sizes.sum()
    target = np.array(RATIOS) * n_total
    rng = np.random.RandomState(1000 + seed)
    best, best_dev = None, np.inf
    for _ in range(N_TRIALS):
        order = rng.permutation(len(sizes))
        load = np.zeros(3)
        bins = np.zeros(len(sizes), dtype=int)
        for g in order:
            room = np.maximum(target - load, 0.0)
            if room.sum() <= 0:
                b = int(np.argmax(target - load))
            else:
                b = int(rng.choice(3, p=room / room.sum()))
            bins[g] = b
            load[b] += sizes[g]
        dev = float(np.abs(load - target).max())
        if dev < best_dev:
            best, best_dev = bins.copy(), dev
            if dev <= 1.0:
                break
    return best, best_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    bundle = HERE / "bundle"
    pool = pd.read_csv(bundle / "pool.csv")
    gid = groups_of(pool)
    sizes = np.bincount(gid)
    ring_free = pool.scaffold.isna().values
    print(f"{len(pool)} structures, {len(sizes)} groups, largest groups {sorted(sizes)[-5:]}; ring-free structures {ring_free.sum()} in "
          f"{len(set(gid[ring_free]))} groups (largest {np.bincount(gid[ring_free]).max()})")
    roles = {}
    for seed in range(1, 6):
        bins, dev = assign(sizes, seed)
        roles[seed] = np.array(ROLES)[bins[gid]]
        print(f"seed {seed}: sizes {[int((roles[seed] == r).sum()) for r in ROLES]} (target {[round(x * len(pool)) for x in RATIOS]}), max deviation {dev:.0f}")
    pool_new = pool.copy()
    pool_new["gid"] = gid
    for seed in range(1, 6):
        pool_new[f"role_s{seed}"] = roles[seed]
    # quality report (label-free criteria first, labels only descriptively)
    rep = {}
    for seed in range(1, 6):
        r = pool_new[f"role_s{seed}"]
        rep[seed] = {role: dict(n=int((r == role).sum()), ring_free=int(((r == role) & ring_free).sum()), positives=int(pool.y[r == role].sum()),
                                prevalence=round(float(pool.y[r == role].mean()), 3), gt500=int(((r == role) & (pool.mw > 500)).sum())) for role in ROLES}
    print(json.dumps(rep, indent=1))
    same = pool_new[[f"role_s{s}" for s in range(1, 6)]].nunique(axis=1).eq(1)
    print("same role in all five seeds:", int(same.sum()), f"({100 * same.mean():.1f}%)", "(first analysis: 2,873, 29.5%)")
    # disjointness checks
    for seed in range(1, 6):
        r = pool_new[f"role_s{seed}"]
        for key, label in (("parent_inchikey", "parent"), ("scaffold", "scaffold")):
            sub = pool_new[pool_new[key].notna()]
            roles_per = sub.groupby(key)[f"role_s{seed}"].nunique()
            assert (roles_per == 1).all(), (seed, label, int((roles_per > 1).sum()))
    print("parents and scaffolds are never shared between subsets in any seed")
    if a.dry_run:
        return
    pool_new.to_csv(bundle / "pool.csv", index=False)
    manifest = dict(algorithm="make_splits.py (random group assignment, best of %d trials per seed)" % N_TRIALS, ratios=RATIOS,
                    acyclic_similarity=ACYCLIC_SIMILARITY, seeds=list(range(1, 6)), n_groups=int(len(sizes)),
                    pool_sha256=hashlib.sha256((bundle / "pool.csv").read_bytes()).hexdigest(), report={str(k): v for k, v in rep.items()},
                    same_role_all_seeds=int(same.sum()))
    (bundle / "SPLIT_V2_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("wrote bundle/pool.csv and SPLIT_V2_MANIFEST.json")


if __name__ == "__main__":
    main()
