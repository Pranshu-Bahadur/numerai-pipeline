"""
Score the v5.0 *live* slice with the two XGB models and
write both CSV and Parquet (Numerai accepts either).
"""

from pathlib import Path
import pandas as pd, xgboost as xgb, numerapi
from src import data

ROOT       = Path(__file__).resolve().parents[1]
MODEL_DIR  = ROOT / "models"
OUT_DIR    = ROOT / "out"; OUT_DIR.mkdir(exist_ok=True, parents=True)

CFG = {
    "xgba": ("xgb_A.json", "v5.0", "small"),
    "xgbb": ("xgb_B.json", "v5.0", "small"),
}

def _load_live(version="v5.0"):
    napi = numerapi.NumerAPI()
    pq   = OUT_DIR / f"{version}_live.parquet"
    if not pq.exists():
        napi.download_dataset(f"{version}/live.parquet", str(pq))
    return pd.read_parquet(pq)

def predict_once(slot: str, model_file: str, feats: list[str], live: pd.DataFrame):
    model = xgb.XGBRegressor();  model.load_model(MODEL_DIR / model_file)
    preds = model.predict(live[feats])
    # Parquet (optional)
    pq_path  = OUT_DIR / f"predictions_{slot}.parquet"
    pd.DataFrame({"prediction": preds}).to_parquet(pq_path, index=live.index)

    return pq_path

def main():
    live = _load_live()
    for slot, (model_file, ver, feat_set) in CFG.items():
        feats = data.get_feature_names(ver, feat_set)
        predict_once(slot, model_file, feats, live)

if __name__ == "__main__":
    main()

