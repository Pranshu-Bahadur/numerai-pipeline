"""
Upload the daily CSVs to Numerai.
"""

import os, numerapi
from pathlib import Path

OUT_DIR = Path("out")
FILES = [
    ("predictions_xgba.parquet", "xgba"),   # secrets
    ("predictions_xgbb.parquet", "xgbb"),
]

def main():
    napi = numerapi.NumerAPI(
        public_id=os.getenv("NUMERAI_PUBLIC"),
        secret_key=os.getenv("NUMERAI_SECRET"),
    )
    for fname, slot in FILES:
        path = OUT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(path)
        model_id=napi.get_models()[slot]
        napi.upload_predictions(str(path), model_id=model_id)
        print("uploaded", path, "→", slot)

if __name__ == "__main__":
    main()

