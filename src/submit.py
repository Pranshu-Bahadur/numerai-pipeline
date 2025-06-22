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
    ("model_xgb_A.pkl", os.getenv("MODEL_NAME_A")),
    ("model_xgb_B.pkl", os.getenv("MODEL_NAME_B")),
]

def submit_all() -> None:
    napi = numerapi.NumerAPI(
        public_id=os.getenv("NUMERAI_PUBLIC"),
        secret_key=os.getenv("NUMERAI_SECRET"),
    )
    for fname, slot in FILES:
        path = PRED_DIR / fname
        if not path.exists():
            raise FileNotFoundError(path)
        napi.upload_predictions(str(path), model_name=slot)
        print("uploaded", path, "→", slot)

if __name__ == "__main__":
    submit_all()

