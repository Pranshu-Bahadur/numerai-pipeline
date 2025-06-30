from pathlib import Path
import json, pandas as pd, xgboost as xgb, glob
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


def train_single(cfg_path: str | Path, df_tr : pd.DataFrame, df_va : pd.DataFrame, feats : list = None) -> Path:
    cfg = load_cfg(cfg_path)
    if not feats:
        feats = [c for c in df_tr.columns if c.startswith('feature')]
    X_tr, y_tr, eras_tr = df_tr[feats], df_tr["target"], df_tr["era"]
    X_va, y_va, eras_va = df_va[feats], df_va["target"], df_va["era"]

    model = xgb.XGBRegressor(**cfg["params"])
    model.fit(X_tr, y_tr)

    preds   = model.predict(X_va)
    df_val = pd.DataFrame(
        {"preds": preds, "target": y_va.values, "era": eras_va.values}
    )

    def _era_metrics(era_df: pd.DataFrame) -> tuple[float, float]:
        corr, _ = spearmanr(era_df["preds"], era_df["target"])
        mae = mean_absolute_error(era_df["target"], era_df["preds"])
        return corr, mae

    era_stats = df_val.groupby("era").apply(_era_metrics).tolist()
    era_corrs, era_maes = zip(*era_stats) 

    corr_mean = float(np.mean(era_corrs))
    mae_mean = float(np.mean(era_maes))

    LOG.info(
        "train_done",
        model=cfg["model_name"],
        avg_corr=corr_mean,
        avg_mae=mae_mean,
        n_eras=len(era_corrs),
    )

    # save
    out = MODEL_DIR / f"{cfg['model_name']}.json"
    model.save_model(out)
    return out


def train_all() -> None:
    for cfg in (list(glob.glob("configs/*.json"))):
        fetch = lambda t: pd.read_parquet(
                data.download_data(cfg["data_version"], file_name=t))
        feats = data.get_feature_names(cfg["data_version"], cfg["feature_set"])
        df_tr = fetch("train")
        df_va = fetch("validation").dropna(subset=["target"])
        train_single(cfg, df_tr, df_va, feats)


if __name__ == "__main__":
    train_all()

