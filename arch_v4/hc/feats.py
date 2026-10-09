"""Molecular features for all structures of the bundle, computed once and cached as a compressed npz.

Views: RDKit 2D descriptors, Morgan (ECFP-like) counts, feature-Morgan (FCFP-like) counts, Avalon counts, ErG, MACCS keys.
"""
from __future__ import annotations

import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_GEN = {}


def _init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _gens():
    if not _GEN:
        from rdkit.Chem import rdFingerprintGenerator as G
        _GEN["ecfp"] = G.GetMorganGenerator(radius=2, fpSize=1024)
        _GEN["fcfp"] = G.GetMorganGenerator(radius=2, fpSize=1024, atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen())
    return _GEN


def _one(smi):
    from rdkit import Chem
    from rdkit.Avalon import pyAvalonTools
    from rdkit.Chem import Descriptors, MACCSkeys, rdReducedGraphs
    g = _gens()
    m = Chem.MolFromSmiles(smi)
    nd = len(Descriptors._descList)
    if m is None:
        return (np.full(nd, np.nan, np.float32), np.zeros(1024, np.uint8), np.zeros(1024, np.uint8), np.zeros(1024, np.uint8),
                np.zeros(315, np.float32), np.zeros(167, np.uint8))
    desc = np.empty(nd, np.float64)
    for i, (_, fn) in enumerate(Descriptors._descList):
        try:
            desc[i] = fn(m)
        except Exception:
            desc[i] = np.nan
    desc = np.clip(np.nan_to_num(desc, nan=np.nan, posinf=1e9, neginf=-1e9), -1e9, 1e9).astype(np.float32)
    ecfp = np.minimum(g["ecfp"].GetCountFingerprintAsNumPy(m), 255).astype(np.uint8)
    fcfp = np.minimum(g["fcfp"].GetCountFingerprintAsNumPy(m), 255).astype(np.uint8)
    av = np.zeros(1024, np.uint8)
    for k, v in pyAvalonTools.GetAvalonCountFP(m, nBits=1024).GetNonzeroElements().items():
        av[k] = min(v, 255)
    try:
        erg = np.asarray(rdReducedGraphs.GetErGFingerprint(m), dtype=np.float32)
    except Exception:
        erg = np.zeros(315, np.float32)
    maccs = np.asarray(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.uint8)
    return desc, ecfp, fcfp, av, erg, maccs


def descriptor_names():
    from rdkit.Chem import Descriptors
    return [n for n, _ in Descriptors._descList]


class Store:
    def __init__(self, smiles, arrays):
        self.index = {s: i for i, s in enumerate(smiles)}
        self.arrays = arrays

    def get(self, key, smiles):
        return self.arrays[key][[self.index[s] for s in smiles]]


def build_store(bundle, cache: str | Path, n_jobs: int | None = None) -> Store:
    cache = Path(cache)
    smiles = list(dict.fromkeys(s for k in bundle for s in bundle[k].smiles))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["smiles"]) == smiles:
            return Store(smiles, {k: z[k] for k in ("desc", "ecfp", "fcfp", "avalon", "erg", "maccs")})
    n_jobs = n_jobs or min(os.cpu_count() or 4, 24)
    with Pool(n_jobs, initializer=_init) as pool:
        rows = pool.map(_one, smiles, chunksize=64)
    arrays = dict(desc=np.stack([r[0] for r in rows]), ecfp=np.stack([r[1] for r in rows]), fcfp=np.stack([r[2] for r in rows]),
                  avalon=np.stack([r[3] for r in rows]), erg=np.stack([r[4] for r in rows]), maccs=np.stack([r[5] for r in rows]))
    np.savez_compressed(cache, smiles=np.array(smiles, dtype=object), **arrays)
    return Store(smiles, arrays)


def clean_descriptors(train_desc: np.ndarray):
    """Column mask and medians from training rows only: drops constant or mostly missing descriptors."""
    ok = np.isfinite(train_desc).mean(0) > 0.95
    med = np.nanmedian(np.where(np.isfinite(train_desc), train_desc, np.nan), axis=0)
    sd = np.nanstd(np.where(np.isfinite(train_desc), train_desc, np.nan), axis=0)
    ok &= sd > 0
    return ok, np.where(np.isfinite(med), med, 0.0)


def apply_descriptors(desc, ok, med):
    x = desc[:, ok].astype(np.float64)
    bad = ~np.isfinite(x)
    x[bad] = np.broadcast_to(med[ok], x.shape)[bad]
    return x
