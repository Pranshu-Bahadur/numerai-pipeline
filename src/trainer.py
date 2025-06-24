from pathlib import Path
import json, pandas as pd, xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from structlog import get_logger
from src import data

ROOT       = Path(__file__).resolve().parents[1]
MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
LOG = get_logger()


def load_cfg(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def train_single(cfg_path: str | Path) -> Path:
    cfg = load_cfg(cfg_path)

    fetch = lambda t: pd.read_parquet(
        data.download_data(cfg["data_version"], file_name=t))
    feats = data.get_feature_names(cfg["data_version"], cfg["feature_set"])

    df_tr = fetch("train")
    df_va = fetch("validation").dropna(subset=["target"])   # v5.0 has NaNs

    X_tr, y_tr = df_tr[feats], df_tr["target"]
    X_va, y_va = df_va[feats], df_va["target"]

    model = xgb.XGBRegressor(**cfg["params"])
    model.fit(X_tr, y_tr)

    preds   = model.predict(X_va)
    corr, _ = spearmanr(preds, y_va)
    mae     = mean_absolute_error(y_va, preds)
    LOG.info("train_done", model=cfg["model_name"], corr=float(corr), mae=float(mae))

    out = MODEL_DIR / f"{cfg['model_name']}.json"   # pure XGB model file
    model.save_model(out)                           # portable across runtimes
    return out


def train_all() -> None:
    for cfg in ("configs/xgb_A.json", "configs/xgb_B.json"):
        train_single(cfg)


if __name__ == "__main__":
    train_all()

