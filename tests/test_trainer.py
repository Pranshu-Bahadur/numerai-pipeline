import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import xgboost as xgb

import src.trainer as training_script

@pytest.fixture
def synthetic_data():
    """
    Provides a synthetic regression dataset as a dictionary containing
    training and validation pandas DataFrames.
    """
    X, y = make_regression(
        n_samples=1000, 
        n_features=20, 
        noise=15.0, 
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    df_train, df_validation = train_test_split(df, test_size=0.3, random_state=42)
    
    return {
        "train": df_train,
        "validation": df_validation,
        "features": feature_names,
    }

@pytest.fixture
def training_config(tmp_path):
    """
    Creates a dummy training configuration JSON file in a temporary directory
    and returns a dictionary with its path and model name.
    """
    model_name = "test_xgb_model_pytest"
    config = {
        "model_name": model_name,
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "random_state": 42
        }
    }
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
        
    return {
        "path": config_path,
        "name": model_name
    }


def test_load_cfg(training_config):
    """
    Tests the configuration loading function `load_cfg` to ensure it correctly
    reads and parses the JSON file.
    """
    cfg_path = training_config["path"]
    
    cfg = training_script.load_cfg(cfg_path)
    
    assert isinstance(cfg, dict)
    assert "model_name" in cfg
    assert "params" in cfg
    assert cfg["model_name"] == training_config["name"]
    assert "n_estimators" in cfg["params"]

def test_train_single_performance(monkeypatch, tmp_path, synthetic_data, training_config):
    """
    Tests the `train_single` function. This test ensures that the function:
    1. Trains a model without errors.
    2. Saves the trained model to the correct path.
    3. The resulting model achieves a good performance score on the validation data.
    """

    monkeypatch.setattr(training_script, 'MODEL_DIR', tmp_path)

    df_tr = synthetic_data["train"]
    df_va = synthetic_data["validation"]
    feats = synthetic_data["features"]
    cfg_path = training_config["path"]
    model_name = training_config["name"]

    output_model_path = training_script.train_single(cfg_path, df_tr, df_va, feats=feats)

    expected_model_path = tmp_path / f"{model_name}.json"
    assert output_model_path == expected_model_path
    assert output_model_path.is_file(), "Model file was not created at the expected path."

    model = xgb.XGBRegressor()
    model.load_model(output_model_path)

    X_va = df_va[feats]
    y_va = df_va["target"]
    predictions = model.predict(X_va)

    corr, _ = spearmanr(y_va, predictions)
    mae = mean_absolute_error(y_va, predictions)

    print(f"\n[Test] Validation Spearman Correlation: {corr:.4f}")
    print(f"[Test] Validation MAE: {mae:.4f}")

    assert corr > 0.95, f"Model correlation ({corr:.4f}) is below the expected threshold of 0.85."
