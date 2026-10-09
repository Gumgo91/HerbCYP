"""Chemprop comparator of the first analysis (scripts/revision_train.py: train_cp, new_mpnn, seed_model, scale_descriptors,
molecule_dataset, cp_predict, Features.desc), ported to the arch_v4 expert interface. Configuration: runs/revision_v1/MODEL_CONFIG.json.

Chemprop 2.3.1 D-MPNN (BondMessagePassing depth 3, hidden 300, MeanAggregation, batch norm), 200 normalized RDKit descriptors
(chemprop registry v1_rdkit_2d_normalized, train-only StandardScaler, missing values -> 0) concatenated before a 2-layer FFN
(hidden 300, BinaryClassificationFFN), BCE, Adam with warm-up 2 epochs (1e-4 -> 1e-3 -> 1e-4), batch size 64, 50 epochs without
early stopping. The checkpoint with the lowest selection loss (val_loss, earliest epoch on ties) is used. The output is the raw
probability of the selected model (the first analysis averaged the raw scores of the five seeds).

HERBCYP_CHEMPROP_EPOCHS overrides the number of epochs for smoke tests only. The default is 50 and must stay 50 for the analysis;
the override changes the learning-rate schedule (cool-down is defined over max_epochs) and is recorded in seed{k}_selection.json.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before the first CUDA call, as in the first analysis

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

EPOCHS = 50
GRAPH_CACHE = True      # runtime-only featurization cache of the E2 stage of the first analysis (equality audited there)
DESC_DIM = 200
_FG = []                # one featurizer instance per process


# --- descriptor featurizer (src/herbcyp/featurize.py: rdkit_2d_normalized, copied; the featurizer is built once per process) ---
def rdkit_2d_normalized(smiles):
    """200-d normalized RDKit descriptors via chemprop's v1 registry."""
    if not _FG:
        from chemprop.featurizers import MoleculeFeaturizerRegistry
        _FG.append(MoleculeFeaturizerRegistry["v1_rdkit_2d_normalized"]())
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    return np.asarray(_FG[0](m), dtype=np.float32)


def _init_worker():
    RDLogger.DisableLog("rdApp.*")


def _desc_worker(smiles):
    return rdkit_2d_normalized(smiles)


# --- model construction and determinism (revision_train.py, unchanged) ---
def seed_model(seed):
    """Seed every RNG BEFORE graph/predictor/module initialization."""
    from lightning import pytorch as pl
    import torch
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def new_mpnn(seed, desc_dim=DESC_DIM):
    from chemprop import models, nn
    seed_model(seed)
    predictor = nn.BinaryClassificationFFN(n_tasks=1, input_dim=300 + desc_dim, hidden_dim=300, n_layers=2)
    return models.MPNN(nn.BondMessagePassing(d_h=300, depth=3), nn.MeanAggregation(), predictor, batch_norm=True)


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def molecule_dataset(smiles, descriptors, labels=None, cache=GRAPH_CACHE):
    from chemprop import data
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    points = [data.MoleculeDatapoint.from_smi(
        str(s), x_d=desc, **({"y": np.array([float(labels[i])])} if labels is not None else {}))
        for i, (s, desc) in enumerate(zip(smiles, descriptors))]
    dataset = data.MoleculeDataset(points, SimpleMoleculeMolGraphFeaturizer())
    dataset.cache = cache
    return dataset


def scale_descriptors(scaler, descriptors):
    """Missing descriptor -> train mean; equivalent to zero after standardization (revision_train.py, unchanged)."""
    transformed = scaler.transform(descriptors).astype(np.float32)
    if np.isinf(transformed).any():
        raise ValueError("Infinite transformed descriptor")
    return np.nan_to_num(transformed, nan=0.0)


def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


class ChempropExpert:
    name = "chemprop"
    out_dir = None      # set by run_expert.py to <out>/<name>; seed{k}_selection.json is written there

    def __init__(self, device="auto", epochs=None, n_jobs=None):
        env = os.environ.get("HERBCYP_CHEMPROP_EPOCHS")
        self.epochs = int(epochs if epochs is not None else (env if env else EPOCHS))
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.epochs != EPOCHS:
            print(f"[chemprop] WARNING: {self.epochs} epochs instead of {EPOCHS}; smoke test only", file=sys.stderr, flush=True)
        self.device = (device or "auto").lower()
        self.n_jobs = n_jobs
        self._desc_cache = {}
        self.last_selection = None

    # -- descriptors, computed lazily per SMILES and kept in memory (identical values in any process) --
    def _desc(self, smiles):
        smiles = [str(s) for s in smiles]
        missing = list(dict.fromkeys(s for s in smiles if s not in self._desc_cache))
        if missing:
            for s, d in zip(missing, self._compute(missing)):
                if d is None:
                    raise ValueError(f"Cannot featurize structure: {s}")
                if np.isinf(d).any() or d.shape != (DESC_DIM,):
                    raise ValueError(f"Malformed/infinite descriptors: {s}")
                self._desc_cache[s] = d
        return np.stack([self._desc_cache[s] for s in smiles]) if smiles else np.zeros((0, DESC_DIM), np.float32)

    def _compute(self, smiles):
        n_jobs = self.n_jobs or min(os.cpu_count() or 1, 8)
        if len(smiles) >= 500 and n_jobs > 1:
            try:
                import multiprocessing as mp
                with mp.get_context("spawn").Pool(n_jobs, initializer=_init_worker) as pool:
                    return pool.map(_desc_worker, smiles, chunksize=64)
            except Exception as exc:   # fall back to one process; the values do not depend on the process
                print(f"[chemprop] parallel descriptor computation failed ({exc!r}); using one process", file=sys.stderr, flush=True)
        return [rdkit_2d_normalized(s) for s in smiles]

    def _accelerator(self):
        d = self.device
        if d in ("auto", ""):
            return "auto", 1
        if d == "cpu":
            return "cpu", 1
        if d in ("gpu", "cuda"):
            return "gpu", 1
        if d.startswith("cuda:") and d[5:].isdigit():
            return "gpu", [int(d[5:])]
        raise ValueError(f"Unknown device {d!r}; use auto, cpu, gpu, cuda or cuda:N")

    def _trainer(self, **kw):
        from lightning import pytorch as pl
        acc, dev = self._accelerator()
        return pl.Trainer(accelerator=acc, devices=dev, deterministic="warn", enable_progress_bar=False,
                          enable_model_summary=False, logger=False, **kw)

    def fit(self, bundle, store, tr, sel, seed):
        from chemprop import data, models
        from lightning import pytorch as pl
        from lightning.pytorch.callbacks import Callback, ModelCheckpoint
        import torch

        pool = bundle["pool"]
        self._desc(pool.smiles.values)          # every pool structure is used in some role in some seed: compute once, in parallel
        s_tr, s_sel = pool.smiles.values[tr], pool.smiles.values[sel]
        y_tr, y_sel = pool.y.to_numpy()[tr], pool.y.to_numpy()[sel]
        assert set(np.unique(np.r_[y_tr, y_sel])) <= {0, 1}
        d_tr, d_sel = self._desc(s_tr), self._desc(s_sel)

        scaler = StandardScaler().fit(d_tr)
        mpnn = new_mpnn(seed)
        initialization_hash = state_hash(mpnn)
        # Initialization is stable even after unrelated RNG use: verify before fitting.
        second = new_mpnn(seed)
        assert state_hash(second) == initialization_hash
        del second
        # Reset once more so the verification does not change training RNG state.
        mpnn = new_mpnn(seed)
        assert state_hash(mpnn) == initialization_hash

        trds = molecule_dataset(s_tr.tolist(), scale_descriptors(scaler, d_tr), y_tr)
        vads = molecule_dataset(s_sel.tolist(), scale_descriptors(scaler, d_sel), y_sel)
        tr_loader = data.build_dataloader(trds, batch_size=64, shuffle=True, seed=seed, num_workers=0)
        va_loader = data.build_dataloader(vads, batch_size=256, shuffle=False, num_workers=0)

        class History(Callback):
            def __init__(self):
                self.val_loss, self.val_roc = [], []

            def on_validation_end(self, trainer, pl_module):    # same hook at which ModelCheckpoint reads val_loss
                if trainer.sanity_checking:
                    return
                m = trainer.callback_metrics
                self.val_loss.append(float(m["val_loss"]))
                self.val_roc.append(float(m["val/roc"]) if "val/roc" in m else None)

        history = History()
        with tempfile.TemporaryDirectory(prefix="hc_chemprop_", ignore_cleanup_errors=True) as tmp:
            callback = ModelCheckpoint(dirpath=tmp, filename="selected-epoch{epoch:02d}", monitor="val_loss", mode="min",
                                       save_top_k=1, save_last=False, auto_insert_metric_name=False)
            trainer = self._trainer(max_epochs=self.epochs, callbacks=[callback, history], default_root_dir=tmp,
                                    log_every_n_steps=50, num_sanity_val_steps=0)
            trainer.fit(mpnn, tr_loader, va_loader)
            self.model = models.MPNN.load_from_checkpoint(callback.best_model_path, map_location=torch.device("cpu"))
            ckpt = torch.load(callback.best_model_path, map_location="cpu", weights_only=False)
            best_epoch = int(ckpt["epoch"])
            best_loss = float(callback.best_model_score.cpu())
        self.scaler = scaler
        self.pred_trainer = self._trainer(enable_checkpointing=False)

        by_epoch = history.val_loss
        assert len(by_epoch) == self.epochs, (len(by_epoch), self.epochs)
        first_min = int(np.argmin(by_epoch))     # earliest epoch on ties
        if first_min != best_epoch:
            raise RuntimeError(f"checkpoint epoch {best_epoch} differs from the earliest minimum of val_loss {first_min}")
        self.last_selection = {
            "seed": seed, "n_train": int(len(tr)), "n_selection": int(len(sel)),
            "selected_epoch_zero_based": best_epoch, "selected_epoch_one_based": best_epoch + 1, "selected_val_loss": best_loss,
            "epochs_completed": self.epochs, "epochs_default": EPOCHS, "initial_state_sha256": initialization_hash,
            "graph_cache": GRAPH_CACHE, "tie_rule": "earliest epoch (strictly lower loss replaces best)",
            "descriptor_missing_rule": "zero after train-only StandardScaler (train-mean imputation)",
            "train_descriptor_missing_rows": int(np.isnan(d_tr).any(axis=1).sum()),
            "selection_descriptor_missing_rows": int(np.isnan(d_sel).any(axis=1).sum()),
            "train_descriptor_all_missing_columns": int(np.isnan(d_tr).all(axis=0).sum()),
            "val_loss_by_epoch": by_epoch, "val_roc_by_epoch": history.val_roc,
            "device": self.device, "resolved_device": str(trainer.strategy.root_device),
            "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() and trainer.strategy.root_device.type == "cuda" else None),
            "versions": {p: _version(p) for p in ("torch", "chemprop", "lightning", "rdkit", "scikit-learn", "numpy")},
        }
        if self.out_dir is not None:
            out = Path(self.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"seed{seed}_selection.json").write_text(json.dumps(self.last_selection, indent=1), encoding="utf8")

    def predict(self, store, frame, set_name=None):
        from chemprop import data
        import torch
        smiles = [str(s) for s in frame.smiles.values]
        if not smiles:
            return np.array([], dtype=float)
        xd = scale_descriptors(self.scaler, self._desc(smiles))
        ds = molecule_dataset(smiles, xd, cache=False)      # used once: no need to keep the graphs
        loader = data.build_dataloader(ds, batch_size=256, shuffle=False, num_workers=0)
        p = torch.cat(self.pred_trainer.predict(self.model, loader)).detach().cpu().numpy().ravel().astype(float)
        if len(p) != len(smiles) or not np.isfinite(p).all():
            raise RuntimeError("chemprop produced a wrong number of predictions or non-finite values")
        return p
