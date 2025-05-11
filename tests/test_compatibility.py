import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def test_numpy_compatibility():
    """NumPyの基本機能テスト"""
    print("NumPyバージョン:", np.__version__)
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3
    print("✓ NumPyの互換性テスト成功")


def test_pandas_compatibility():
    """Pandasの基本機能テスト"""
    print("Pandasバージョン:", pd.__version__)
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    assert len(df) == 3
    print("✓ Pandasの互換性テスト成功")


def test_catboost_compatibility():
    # ダミーデータ
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = CatBoostClassifier(iterations=10, random_seed=42, verbose=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 100
    print("CatBoost動作確認OK")


if __name__ == "__main__":
    print("ライブラリ互換性テストを開始します...")
    test_numpy_compatibility()
    test_pandas_compatibility()
    test_catboost_compatibility()
    print("✓ すべてのテストが成功しました")
