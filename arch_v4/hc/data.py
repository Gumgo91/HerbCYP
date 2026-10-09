"""Bundle access and the frozen scaffold splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [1, 2, 3, 4, 5]
EXTERNAL = ["ambe", "stress", "herb", "unmeasured"]
SETS = ["calibration"] + EXTERNAL
LABELLED = ["calibration", "ambe", "stress", "herb"]


def load_bundle(d) -> dict[str, pd.DataFrame]:
    d = Path(d)
    return {k: pd.read_csv(d / f"{k}.csv") for k in ["pool"] + EXTERNAL}


def split(pool: pd.DataFrame, seed: int):
    r = pool[f"role_s{seed}"].values
    return np.where(r == "train")[0], np.where(r == "selection")[0], np.where(r == "calibration")[0]


UNM_MAX_MW = 500.0   # the application domain: predictions are made for structures of at most 500 Da


def unmeasured_domain(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.mw <= UNM_MAX_MW].reset_index(drop=True)


def eval_frames(bundle: dict, seed: int) -> dict[str, pd.DataFrame]:
    """All frames an expert must score for one seed. Calibration rows come from the pool."""
    _, _, cal = split(bundle["pool"], seed)
    out = {"calibration": bundle["pool"].iloc[cal].reset_index(drop=True)}
    for k in EXTERNAL:
        out[k] = unmeasured_domain(bundle[k]) if k == "unmeasured" else bundle[k]
    return out
