"""MapLight-style baseline: CatBoost on Morgan counts, Avalon counts, ErG and RDKit descriptors."""
from __future__ import annotations

import os

import numpy as np

from hc.feats import apply_descriptors, clean_descriptors


class MapLightCB:
    name = "maplight_cb"

    def __init__(self, iterations=4000, lr=0.05, depth=6, gpu=False):
        self.iterations, self.lr, self.depth, self.gpu = iterations, lr, depth, gpu

    def _x(self, store, smiles):
        d = apply_descriptors(store.get("desc", smiles), self.ok, self.med)
        return np.hstack([store.get("ecfp", smiles), store.get("avalon", smiles), store.get("erg", smiles), d]).astype(np.float32)

    def fit(self, bundle, store, tr, sel, seed):
        from catboost import CatBoostClassifier
        pool = bundle["pool"]
        s_tr, s_sel = pool.smiles.values[tr], pool.smiles.values[sel]
        self.ok, self.med = clean_descriptors(store.get("desc", s_tr))
        kw = dict(task_type="GPU") if self.gpu else dict(thread_count=os.cpu_count())
        self.model = CatBoostClassifier(iterations=self.iterations, learning_rate=self.lr, depth=self.depth, eval_metric="AUC",
                                        random_seed=seed, od_type="Iter", od_wait=200, verbose=False, **kw)
        self.model.fit(self._x(store, s_tr), pool.y.values[tr], eval_set=(self._x(store, s_sel), pool.y.values[sel]),
                       use_best_model=True)

    def predict(self, store, frame, set_name=None):
        return self.model.predict_proba(self._x(store, frame.smiles.values))[:, 1]
