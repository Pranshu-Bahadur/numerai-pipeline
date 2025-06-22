"""
Upload cloud-pickled models to Numerai.
Expects repo secrets:
  NUMERAI_PUBLIC, NUMERAI_SECRET
  MODEL_NAME_A,  MODEL_NAME_B
"""

import os, numerapi
from pathlib import Path

PRED_DIR = Path("preds")
FILES = [
    ("model_xgb_A.pkl", "xgba"),
    ("model_xgb_B.pkl", "xgbb"),
]

def submit_all() -> None:
    napi = numerapi.NumerAPI(os.getenv("NUMERAI_PUBLIC"), os.getenv("NUMERAI_SECRET"))
    for fname, slot in FILES:
        path = PRED_DIR / fname
        if not path.exists():
            raise FileNotFoundError(path)
        model_id = napi.get_models()[slot]
        napi.upload_predictions(str(path), model_id=model_id)
        print("uploaded", path, "→", slot)

if __name__ == "__main__":
    submit_all()

