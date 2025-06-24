# tests/test_inference.py
import numerapi, cloudpickle, pandas as pd, pathlib, os

LIVE_PQ = pathlib.Path("data/numerai_v5_live.parquet")
PICKLE  = pathlib.Path("preds/model_xgb_A.pkl")   # test one model is enough


def _download_live():
    """Grab v5.0 live parquet once; cache locally."""
    if LIVE_PQ.exists():
        return LIVE_PQ
    LIVE_PQ.parent.mkdir(exist_ok=True, parents=True)
    napi = numerapi.NumerAPI()
    napi.download_dataset("v5.0/live.parquet", str(LIVE_PQ))
    return LIVE_PQ


def test_inference_on_live():
    # 1) ensure live data is available
    live_df = pd.read_parquet(_download_live())

    # 2) ensure pickle exists (trainer+inference scripts must have run)
    assert PICKLE.exists(), "Run `python -m src.inference` first"
    predict_fn = cloudpickle.load(open(PICKLE, "rb"))

    # 3) call predict
    out = predict_fn(live_df, pd.DataFrame())

    # 4) basic sanity checks
    assert list(out.columns) == ["prediction"]
    assert len(out) == len(live_df)
    assert out["prediction"].dtype.kind in ("f", "i")   # numeric

