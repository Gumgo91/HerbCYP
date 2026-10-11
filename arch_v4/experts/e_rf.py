"""Random forest model for the expert interface of arch_v4.

500 trees, n_jobs=-1, random_state=seed, scikit-learn defaults otherwise, on 2,048-bit Morgan bit vectors of radius 2.
Fitted on the training rows only; the selection rows are not used by this model.
The output is the raw predict_proba of the positive class.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

RDLogger.DisableLog("rdApp.*")

_GEN = {}


def morgan_fp(smiles, radius=2, nbits=2048):
    """Morgan bit vector as int8. Bit-identical to AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) used by
    src/herbcyp/featurize.py (checked on all 39,814 SMILES of the bundle); the generator API is the supported one."""
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    if (radius, nbits) not in _GEN:
        _GEN[(radius, nbits)] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return _GEN[(radius, nbits)].GetFingerprintAsNumPy(m).astype(np.int8)


class RandomForestExpert:
    name = "rf"
    out_dir = None   # set by run_expert.py; unused by this expert

    def __init__(self, n_estimators=500, n_jobs=-1, radius=2, nbits=2048, **_ignored):   # device is accepted and ignored
        self.n_estimators, self.n_jobs, self.radius, self.nbits = n_estimators, n_jobs, radius, nbits
        self._fps = {}

    def _x(self, smiles):
        out = []
        for s in smiles:
            s = str(s)
            if s not in self._fps:
                fp = morgan_fp(s, self.radius, self.nbits)
                if fp is None or fp.shape != (self.nbits,):
                    raise ValueError(f"Cannot featurize structure: {s}")
                self._fps[s] = fp
            out.append(self._fps[s])
        return np.stack(out)

    def fit(self, bundle, store, tr, sel, seed):
        pool = bundle["pool"]
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=self.n_jobs, random_state=seed)
        self.model.fit(self._x(pool.smiles.values[tr]), pool.y.to_numpy()[tr])
        assert list(self.model.classes_) == [0, 1]

    def predict(self, store, frame, set_name=None):
        smiles = frame.smiles.values
        if len(smiles) == 0:
            return np.array([], dtype=float)
        return self.model.predict_proba(self._x(smiles))[:, 1]
