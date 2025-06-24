# Numerai Pipeline

![CI](https://github.com/Pranshu-Bahadur/numerai-pipeline/actions/workflows/ci.yml/badge.svg)
![Daily Submit](https://github.com/Pranshu-Bahadur/numerai-pipeline/actions/workflows/submit.yml/badge.svg)

Lean, reproducible workflow that

1. trains **two** preset XGBoost models on the **train** slice of Numerai v5.0  
2. evaluates on the **validation** slice (Spearman corr / MAE / Sharpe)  
3. wraps each model in a Numerai-compliant, cloud-pickled `predict()` callable  
4. (optional) **daily workflow** scores the live slice and uploads CSV predictions automatically

> **Note:** Numerai’s model-upload is UI-only, so `model_xgb_*.pkl`
> must be uploaded manually once.  
> After that, the scheduled workflow pushes fresh CSV predictions each day.

---

## 1 · Repo layout

```

.
├─ src/
│   ├─ data.py             # parquet + feature-list helpers
│   ├─ trainer.py          # trains xgb\_A / xgb\_B → models/*.json
│   ├─ inference.py        # builds & cloud-pickles predict()
│   ├─ predict\_live.py     # scores live slice → CSV / Parquet
│   ├─ submit.py       # uploads daily CSVs via NumerAPI
│   └─ **init**.py
├─ configs/                # hyper-param JSONs
├─ models/   (generated)   # XGBoost JSON models
├─ preds/    (generated)   # model\_xgb\_A.pkl / model\_xgb\_B.pkl
├─ out/      (generated)   # daily predictions\_*.csv / .parquet
├─ tests/                  # fast unit tests (mocked + live smoke)
└─ .github/workflows/
├─ ci.yml               # push / PR → run tests
└─ submit.yml           # 03:15 UTC daily → predict + upload

````

---

## 2 · Quick-start (local)

```bash
git clone https://github.com/<user>/numerai-pipeline.git
cd numerai-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) train the two models (~3–5 min CPU)
python -m src.trainer

# 2) build Numerai-compatible pickles
python -m src.inference        # → preds/model_xgb_A.pkl, model_xgb_B.pkl
````

---

## 3 · Upload models (one-time)

1. Upload `model_xgb_A.pkl` and `model_xgb_B.pkl` in **Dashboard → Models**.
2. Assign them to slot names (`xgba`, `xgbb`, …).
3. Once uploaded, Numerai can call your `predict()`.

---

## 4 · Automated daily submission

### 4.1 Secrets to add in GitHub → Settings → Actions → Secrets

| Secret           | Value                                    |
| ---------------- | ---------------------------------------- |
| `NUMERAI_PUBLIC` | your public API key                      |
| `NUMERAI_SECRET` | your secret API key                      |

### 4.2 Workflow

* `.github/workflows/submit.yml` runs at **03 : 15 UTC** daily (and via the “Run workflow” button).
* Steps: install deps → `python -m src.predict_live` → `python -m src.submit_csv`.
* On success you’ll see two “uploaded … → slot” lines in the logs and fresh submission timestamps in the Numerai dashboard.

---

## 5 · Config knobs

| Field          | Purpose                                   |
| -------------- | ----------------------------------------- |
| `data_version` | `"v5.0"` (or `"v4"`, etc.)                |
| `feature_set`  | `"small"`, `"medium"`, `"large"`          |
| `params`       | Passed straight to `xgboost.XGBRegressor` |

---

## 6 · Testing

```bash
pytest -q     # mocked unit tests + live smoke, ~7 s
```

---

## 7 · Extending

* Add LightGBM / CatBoost → new configs + branch in `trainer.py`.
* Hyper-param sweeps → plug Optuna into the trainer.
* Extra robustness → retries + Slack alert in `submit.yml`.

---

© 2025 Pranshu Bahadur — MIT License. PRs and issues welcome!

```
```
``

