"""In-context tabular learner (TabPFN) on RDKit descriptors, MACCS keys and compressed Morgan counts."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from hc.feats import apply_descriptors, clean_descriptors


def make_tabpfn(seed, n_estimators=8, device="cuda"):
    from tabpfn import TabPFNClassifier
    try:   # keep the openly licensed v2 weights if the package offers several versions
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2, n_estimators=n_estimators, device=device,
                                                           random_state=seed, ignore_pretraining_limits=True)
    except Exception:
        return TabPFNClassifier(n_estimators=n_estimators, device=device, random_state=seed, ignore_pretraining_limits=True)


class TabPFNDesc:
    name = "tabpfn_desc"

    def __init__(self, n_estimators=8, pca=64, device="cuda", chunk=2000):
        self.n_estimators, self.pca_dim, self.device, self.chunk = n_estimators, pca, device, chunk

    def _x(self, store, smiles):
        d = apply_descriptors(store.get("desc", smiles), self.ok, self.med)
        pc = self.pca.transform(np.log1p(store.get("ecfp", smiles).astype(np.float32)))
        return np.hstack([d, store.get("maccs", smiles), pc]).astype(np.float32)

    def fit(self, bundle, store, tr, sel, seed):
        pool = bundle["pool"]
        s_tr = pool.smiles.values[tr]
        self.ok, self.med = clean_descriptors(store.get("desc", s_tr))
        self.pca = PCA(self.pca_dim, random_state=seed).fit(np.log1p(store.get("ecfp", s_tr).astype(np.float32)))
        self.clf = make_tabpfn(seed, self.n_estimators, self.device)
        self.clf.fit(self._x(store, s_tr), pool.y.values[tr])

    def predict(self, store, frame, set_name=None):
        x = self._x(store, frame.smiles.values)
        return np.concatenate([self.clf.predict_proba(x[i:i + self.chunk])[:, 1] for i in range(0, len(x), self.chunk)])
