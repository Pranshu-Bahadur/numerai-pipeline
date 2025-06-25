# Numerai Pipeline

![CI](https://github.com/Pranshu-Bahadur/numerai-pipeline/actions/workflows/ci.yml/badge.svg)
![Daily Submit](https://github.com/Pranshu-Bahadur/numerai-pipeline/actions/workflows/submit.yml/badge.svg)

Lean, reproducible workflow that

1. trains **two** preset XGBoost models on the **train** slice of Numerai v5.0  
2. evaluates on the **validation** slice (Spearman corr / MAE / Sharpe)  
3. wraps each model in a Numerai-compliant, cloud-pickled `predict()` callable  
4. **daily workflow** scores the live slice and uploads predictions automatically

> **Note:** Numerai’s *model-upload* is UI-only.  
> Upload `model_xgb_*.pkl` once; the scheduled workflow handles daily predictions thereafter.

---

## 1 · Repo layout

```

.
├─ src/
│   ├─ data.py             # parquet + feature-list helpers
│   ├─ trainer.py          # trains xgb\_A / xgb\_B  → models/\*.json
│   ├─ inference.py        # builds & cloud-pickles predict()
│   ├─ predict\_live.py     # scores live slice → Parquet files
│   ├─ submit.py           # uploads predictions via NumerAPI
│   └─ **init**.py
├─ configs/                # hyper-param JSONs
├─ models/   (generated)   # XGBoost JSON models
├─ preds/    (generated)   # model\_xgb\_A.pkl / model\_xgb\_B.pkl
├─ out/      (generated)   # predictions\_xgba.parquet / predictions\_xgbb.parquet
├─ tests/                  # fast unit tests (mocked + smoke)
└─ .github/workflows/
├─ ci.yml               # push / PR → run tests
└─ submit.yml           # 13 : 00 UTC daily → watch-predict-submit

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

1. **Dashboard → Models → Upload Model**
2. Select `model_xgb_A.pkl` and `model_xgb_B.pkl` → assign to slots (`xgba`, `xgbb`, …).
3. Numerai now calls your `predict()` function on each live round.

---

## 4 · Automated daily submission

### 4.1 Secrets (GitHub → Settings → Actions → *New secret*)

| Secret name      | Value (example)             |
| ---------------- | --------------------------- |
| `NUMERAI_PUBLIC` | `NXXXXXXXXXXXXXXXXXXXXXXXX` |
| `NUMERAI_SECRET` | `SXXXXXXXXXXXXXXXXXXXXXXXX` |

### 4.2 Workflow behaviour

* `.github/workflows/submit.yml` starts once at **13 : 00 UTC**.

* A Python watcher polls every 10 min (max 5 h) using `napi.check_round_open()`.

* As soon as the live file is ready it runs

  ```bash
  python -m src.predict_live   # writes out/predictions_xgba.parquet, ...
  python -m src.submit         # uploads both files
  ```

* Logs show `uploaded … → slot` on success, plus a Slack/Discord alert if configured.

---

## 5 · Config knobs

| Field (in `configs/xgb_*.json`) | Purpose                                   |
| ------------------------------- | ----------------------------------------- |
| `data_version`                  | `"v5.0"` (or `"v4"`, etc.)                |
| `feature_set`                   | `"small"`, `"medium"`, `"large"`          |
| `params`                        | passed straight to `xgboost.XGBRegressor` |

---

## 6 · Testing

```bash
pytest -q                  # mocked unit tests + live-data smoke (< 10 s)
```

---

## 7 · Extending

* **LightGBM / CatBoost** → add configs + branch in `trainer.py`
* **Hyper-parameter sweeps** → plug Optuna into the trainer
* **Extra robustness** → add retries + Slack alert step in `submit.yml`

---

© 2025 Pranshu Bahadur — MIT License. PRs and issues welcome!

```
```

