from pathlib import Path
import json, joblib, pandas as pd, xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from structlog import get_logger
from src import data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
LOG = get_logger()

def load_cfg(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)

def train_single(cfg_path: str | Path) -> Path:
    cfg = load_cfg(cfg_path)
    _get_df  = lambda dataset_type: pd.read_parquet(data.download_data(version=cfg["data_version"], file_name = dataset_type))
    feats = data.get_feature_names(cfg["data_version"], cfg["feature_set"])
    df_train = _get_df('train')
    df_val   = _get_df('validation').dropna(subset='target') #Numerai Validation Data Contains NaN Target Values.

    X_tr, y_tr = df_train[feats], df_train["target"]
    X_val, y_val = df_val[feats], df_val["target"]

    model = xgb.XGBRegressor(**cfg["params"])
    model.fit(X_tr, y_tr)

    val_pred = model.predict(X_val)
    corr, _  = spearmanr(val_pred, y_val)
    mae      = mean_absolute_error(y_val, val_pred)

    LOG.info("train_done",
             model   = cfg["model_name"],
             version = cfg["data_version"],
             feature_set = cfg["feature_set"],
             corr    = float(corr),
             mae     = float(mae))

    out = MODEL_DIR / f"{cfg['model_name']}.json"
    model.save_model(out)
    return out

def train_all():
    for cfg in ("configs/xgb_A.json", "configs/xgb_B.json"):
        train_single(cfg)

if __name__ == "__main__":
    train_all()

