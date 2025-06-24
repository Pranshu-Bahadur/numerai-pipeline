# Numerai Pipeline

![CI](https://github.com/Pranshu-Bahadur/numerai-pipeline/actions/workflows/ci.yml/badge.svg)

Lean, reproducible workflow that  

1. trains **two** preset XGBoost models on the **train** slice of Numerai v5.0  
2. evaluates on the official **validation** slice (Spearman corr / MAE / Sharpe)  
3. wraps each model in a Numerai-compliant cloud-pickled `predict()` callable  

> **Note:** Numerai’s model-upload is UI-only, so the generated `model_xgb_*.pkl`
> files must be uploaded manually in the dashboard.

---

## 1 Repo layout

```

.
├─ src/
│   ├─ data.py            # parquet + feature-list helpers
│   ├─ trainer.py         # trains xgb\_A / xgb\_B → models/\*.json
│   ├─ inference.py       # builds & cloud-pickles predict()
│   └─ **init**.py
├─ configs/               # hyper-param JSONs
├─ models/   (generated)  # XGBoost JSON models
├─ preds/    (generated)  # model\_xgb\_A.pkl / model\_xgb\_B.pkl
├─ tests/                 # fast unit tests (mocked + live smoke)
└─ .github/workflows/
└─ ci.yml             # push / PR → run tests

````

---

## 2 Quick-start (local)

```bash
git clone https://github.com/<user>/numerai-pipeline.git
cd numerai-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) train the two models (~3 – 5 min CPU)
python -m src.trainer

# 2) build Numerai-compatible pickles
python -m src.inference       # → preds/model_xgb_A.pkl, model_xgb_B.pkl
````

---

## 3 Upload to Numerai

1. Locate `preds/model_xgb_A.pkl` and `model_xgb_B.pkl`.
2. Numerai **Dashboard → Models → Upload Model** (one per slot).
3. Map each file to its slot name (`xgb_A`, `xgb_B`, …).
4. Numerai will call your `predict()` automatically on live rounds.

*(No API keys needed for this manual step.)*

---

## 4 Config knobs

| Field (in `configs/xgb_*.json`) | Purpose                                   |
| ------------------------------- | ----------------------------------------- |
| `data_version`                  | `"v5.0"` (or `"v4"`, etc.)                |
| `feature_set`                   | `"small"`, `"medium"`, `"large"`          |
| `params`                        | Passed straight to `xgboost.XGBRegressor` |

Add new configs and list them in `trainer.py::train_all()` to train more models.

---

## 5 Testing

```bash
pytest -q      # mocked unit tests + one live-parquet smoke test
```

*CI uses monkey-patched data, so the push/PR run finishes in ≈ 7 s.*

---

## 6 Extending

* LightGBM or CatBoost → add configs + a small branch in `trainer.py`.
* Hyper-parameter sweeps → plug Optuna into the trainer.
* CSV nightly submission → re-enable `src.predict.py` and NumerAPI upload.

---

© 2025 Pranshu Bahadur — MIT License. PRs and issues welcome!

