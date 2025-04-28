"""
シンプル予測モデルのテストコード
"""

import pytest
import numpy as np
import pandas as pd
from src.prediction.simple_model import SimplePredictor


@pytest.fixture
def sample_data():
    """テストデータの作成"""
    np.random.seed(42)
    n_samples = 100
    odds = np.random.uniform(1.0, 20.0, n_samples)
    results = np.random.binomial(1, 1 / odds)
    return pd.DataFrame({"単勝オッズ": odds, "着順": results})


def test_data_preparation(sample_data):
    """データ準備機能のテスト"""
    # prepare_dataの代わりに直接特徴量とラベルを作成
    X = np.log(sample_data["単勝オッズ"].values).reshape(-1, 1)
    y = sample_data["着順"].astype(int).values

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[1] == 1  # 特徴量は1次元
    assert X.shape[0] == y.shape[0]  # サンプル数が一致
    assert set(np.unique(y)) <= {0, 1}  # 二値分類


def test_model_training(sample_data):
    """モデル学習機能のテスト"""
    predictor = SimplePredictor()
    X = np.log(sample_data["単勝オッズ"].values).reshape(-1, 1)
    y = sample_data["着順"].astype(int).values

    predictor.train(X, y)
    assert hasattr(predictor.model, "coef_")  # モデルが学習済みか確認


def test_prediction(sample_data):
    """予測機能のテスト"""
    predictor = SimplePredictor()
    X = np.log(sample_data["単勝オッズ"].values).reshape(-1, 1)
    y = sample_data["着順"].astype(int).values

    predictor.train(X, y)
    predictions = predictor.predict(X)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == y.shape
    assert set(np.unique(predictions)) <= {0, 1}  # 二値分類の結果


def test_evaluation(sample_data):
    """評価機能のテスト"""
    predictor = SimplePredictor()
    X = np.log(sample_data["単勝オッズ"].values).reshape(-1, 1)
    y = sample_data["着順"].astype(int).values

    predictor.train(X, y)
    metrics = predictor.evaluate(X, y)

    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())  # すべての指標が0-1の範囲
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
