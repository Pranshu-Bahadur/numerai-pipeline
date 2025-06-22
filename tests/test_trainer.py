from pathlib import Path
import pandas as pd
import numpy as np
import types
from src import trainer, data


# ---- helpers --------------------------------------------------------------
def _fake_df(rows=50, feats=5):
    cols = [f"feature{i}" for i in range(feats)]
    df   = pd.DataFrame(np.random.randn(rows, feats), columns=cols)
    df["target"] = np.random.randn(rows)
    return df

FAKE_TRAIN = _fake_df(80)
FAKE_VAL   = _fake_df(20)
FAKE_FEATURES = [c for c in FAKE_TRAIN.columns if c.startswith("feature")]


def _patch(monkeypatch):
    # 1) stub download_data to return a Path with the slice name in it
    def _dummy_dl(version="v5.0", file_name="train", cache_dir=None):
        return Path(f"/tmp/{file_name}.parquet")
    monkeypatch.setattr(data, "download_data", _dummy_dl)

    # 2) stub pandas.read_parquet: decide train vs val by path name
    def _dummy_read(path):
        return FAKE_TRAIN if path.name.startswith("train") else FAKE_VAL
    monkeypatch.setattr(pd, "read_parquet", _dummy_read)

    # 3) stub feature list
    monkeypatch.setattr(data, "get_feature_names",
                        lambda v, s: FAKE_FEATURES)


# ---- tests ----------------------------------------------------------------
def _check(monkeypatch, cfg):
    _patch(monkeypatch)
    p = trainer.train_single(cfg)
    assert p.exists() and p.stat().st_size > 0

def test_xgb_A(monkeypatch):
    _check(monkeypatch, "configs/xgb_A.json")

def test_xgb_B(monkeypatch):
    _check(monkeypatch, "configs/xgb_B.json")

