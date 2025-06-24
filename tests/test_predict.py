import pytest, numerapi, pandas as pd, xgboost as xgb
from pathlib import Path
from src import predict_live

LIVE_PQ = Path("data/full_live_v50.parquet")

def download_live():
    if not LIVE_PQ.exists():
        LIVE_PQ.parent.mkdir(exist_ok=True, parents=True)
        numerapi.NumerAPI().download_dataset("v5.0/live.parquet", str(LIVE_PQ))
    return pd.read_parquet(LIVE_PQ)

@pytest.mark.integration
def test_full_predict_live():
    # require that models have already been trained and saved
    model_a = Path("models/xgb_A.json")
    model_b = Path("models/xgb_B.json")
    if not (model_a.exists() and model_b.exists()):
        pytest.skip("Trained model files not found; run src.trainer first")

    live = download_live()

    # Use predict_live's public helper to generate CSVs in tmp directory
    out_dir = Path("out_full"); out_dir.mkdir(exist_ok=True)
    predict_live.OUT_DIR = out_dir  # redirect output

    predict_live.main()  # this will generate predictions_xgba.csv / xgbb.csv

    for slot in ("xgba", "xgbb"):
        csv_path = out_dir / f"predictions_{slot}.parquet"
        assert csv_path.exists()
        preds = pd.read_parquet(csv_path, header=None)
        assert len(preds) == len(live)

