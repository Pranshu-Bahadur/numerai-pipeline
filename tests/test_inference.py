from src.inference import build_predict_fn
import pandas as pd, numpy as np, joblib, tempfile, cloudpickle, gzip

def test_predict_signature(tmp_path):
    # minimal fake model: outputs zeros
    fake_model = lambda X: np.zeros(len(X))
    model_pkl = tmp_path / "m.pkl"
    joblib.dump(fake_model, model_pkl)

    feats = [f"f{i}" for i in range(3)]
    fn = build_predict_fn(model_pkl, feats)

    live = pd.DataFrame(np.ones((5, 3)), columns=feats)
    out = fn(live, pd.DataFrame())
    assert list(out.columns) == ["prediction"] and len(out) == 5

