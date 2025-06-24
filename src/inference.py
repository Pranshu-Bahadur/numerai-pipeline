from pathlib import Path
import cloudpickle, pandas as pd, xgboost as xgb
from src import data

ROOT      = Path(__file__).resolve().parents[1]
PRED_DIR  = ROOT / "preds";   PRED_DIR.mkdir(exist_ok=True)

CFG = {
    "xgb_A": ("models/xgb_A.json", "v5.0", "medium"),
    "xgb_B": ("models/xgb_B.json", "v5.0", "medium"),
}


def build_predict_fn(model_path: Path, feature_cols: list[str]):
    """Return a Numerai-ready predict() callable."""
    model = xgb.XGBRegressor()
    model.load_model(model_path)              

    def predict(live_features: pd.DataFrame,
                live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
        preds = model.predict(live_features[feature_cols])
        return pd.DataFrame({"prediction": preds}, index=live_features.index)

    return predict


def main() -> None:
    for name, (model_rel, version, feat_set) in CFG.items():
        model_path = ROOT / model_rel
        feats      = data.get_feature_names(version, feat_set)

        fn = build_predict_fn(model_path, feats)
        out_pkl = PRED_DIR / f"model_{name}.pkl"
        out_pkl.write_bytes(cloudpickle.dumps(fn))
        print("created →", out_pkl)


if __name__ == "__main__":
    main()

