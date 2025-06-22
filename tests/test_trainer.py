from src import trainer
from pathlib import Path

def _run(cfg):
    model_path = trainer.train_single(cfg)
    assert model_path.exists() and model_path.stat().st_size > 0

def test_xgb_A(tmp_path):
    _run("configs/xgb_A.json")

def test_xgb_B(tmp_path):
    _run("configs/xgb_B.json")

