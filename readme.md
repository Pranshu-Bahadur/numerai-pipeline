# Numerai CI-Pipeline — XGBoost (v5.0 medium)

![CI](https://github.com/<your-user>/numerai-ci-pipeline/actions/workflows/ci.yml/badge.svg)

Lean, reproducible pipeline that

1. trains two preset XGBoost models on the **train** slice of Numerai v5.0  
2. evaluates on the official **validation** slice (corr / MAE / Sharpe)  
3. wraps each model in a cloud-pickled `predict()` callable expected by Numerai  

> **Important:** Numerai’s “model-upload” endpoint is *UI-only*.  

---

## 1 Repo layout

```

.
├─ src/
│   ├─ data.py               # download parquet + feature sets
│   ├─ trainer.py            # trains xgb\_A / xgb\_B
│   ├─ inference.py  # wraps & cloud-pickles predict()
│   └─ **init**.py
├─ configs/                  # hyper-param JSONs
├─ models/   (generated)     # trained XGB pickles
├─ preds/    (generated)     # model\_xgb\_A.pkl / model\_xgb\_B.pkl
├─ tests/                    # mocked fast unit tests
└─ .github/workflows/
	├─ ci.yml                # push/PR – fast tests only

````

---

## 2 Quick-start locally

```bash
git clone https://github.com/<user>/numerai-ci-pipeline.git
cd numerai-ci-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train two models (~3-5 min CPU)
python -m src.trainer

# build Numerai-compatible pickles
python -m src.inference.py     # outputs preds/model_xgb_*.pkl
````

---

## 3 Uploading the models to Numerai

1. Unzip `numerai-models.zip` – you’ll get
   `model_xgb_A.pkl`, `model_xgb_B.pkl`.
2. Numerai dashboard → **Models** → **Upload Model** for each slot.
3. Assign filenames to their corresponding slots (e.g., `xgb_A`, `xgb_B`).
4. Numerai will now call your `predict()` function automatically on future rounds.

*(No secret keys are required for this manual upload.)*

---

## 4 Config knobs

| Field in `configs/xgb_*.json` | Meaning                                    |
| ----------------------------- | ------------------------------------------ |
| `data_version`                | `"v5.0"` (change to `"v4"` etc. if needed) |
| `feature_set`                 | `"small"`, `"medium"`, `"large"`           |
| `params`                      | Passed directly to `xgboost.XGBRegressor`  |

Add more config files and list them in `src/trainer.py::train_all()` to train extra models.

---

## 5 Testing

```bash
pytest -q          # 5 tests, all mocked, ~7 s
```

*No large downloads in CI—the trainer is monkey-patched to use a tiny DataFrame.*

---

## 6 Extending

* Add LightGBM / CatBoost: new configs + a small branch in `trainer.py`.
* Hyper-parameter sweeps: plug Optuna into the trainer.
* CSV nightly submission: re-enable `src.predict.py` + NumerAPI upload if needed.

---

© 2025 Pranshu Bahadur – MIT License
PRs and issues welcome.

