import numpy as np
import pandas as pd
import torch
import traceback
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def test_numpy_compatibility():
    """NumPyの基本機能テスト"""
    try:
        print("NumPyバージョン:", np.__version__)

        # 基本的な配列操作
        arr = np.array([1, 2, 3, 4, 5])
        print("配列作成:", arr)
        mean = arr.mean()
        print("平均値:", mean)
        assert mean == 3

        # 行列演算
        matrix = np.array([[1, 2], [3, 4]])
        print("行列作成:", matrix)
        det = np.linalg.det(matrix)
        print("行列式:", det)
        assert np.abs(det + 2) < 1e-10  # 浮動小数点の比較を許容誤差付きで行う

        print("✓ NumPyの互換性テスト成功")
    except Exception as e:
        print(f"✗ NumPyテストエラー: {str(e)}")
        print("詳細なエラー情報:")
        print(traceback.format_exc())
        raise


def test_pandas_compatibility():
    """Pandasの基本機能テスト"""
    try:
        print("Pandasバージョン:", pd.__version__)

        # データフレーム操作
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        print("データフレーム作成:", df)
        assert len(df) == 3
        assert list(df.columns) == ["A", "B"]

        # 基本的な集計
        mean = df["A"].mean()
        print("A列の平均値:", mean)
        assert df["A"].mean() == 2

        print("✓ Pandasの互換性テスト成功")
    except Exception as e:
        print(f"✗ Pandasテストエラー: {str(e)}")
        print("詳細なエラー情報:")
        print(traceback.format_exc())
        raise


def test_sklearn_compatibility():
    """scikit-learnの基本機能テスト"""
    try:
        print("scikit-learnバージョン:", sklearn.__version__)

        # データセット生成
        X, y = make_classification(n_samples=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("データセット作成完了")

        # モデルのトレーニングと予測
        model = LogisticRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print("モデルスコア:", score)
        assert 0 <= score <= 1

        print("✓ scikit-learnの互換性テスト成功")
    except Exception as e:
        print(f"✗ scikit-learnテストエラー: {str(e)}")
        print("詳細なエラー情報:")
        print(traceback.format_exc())
        raise


def test_pytorch_compatibility():
    """PyTorchの基本機能テスト"""
    try:
        print("PyTorchバージョン:", torch.__version__)

        # テンソル操作
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
        print("テンソルx:", x)
        print("テンソルy:", y)
        z = torch.matmul(x, y)
        print("行列積z:", z)
        assert z.shape == (2, 2)

        # GPUが利用可能かチェック（オプション）
        gpu_available = torch.cuda.is_available()
        print(f"GPU利用可能: {gpu_available}")

        print("✓ PyTorchの互換性テスト成功")
    except Exception as e:
        print(f"✗ PyTorchテストエラー: {str(e)}")
        print("詳細なエラー情報:")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    print("ライブラリ互換性テストを開始します...")
    print("-" * 50)

    try:
        test_numpy_compatibility()
        test_pandas_compatibility()
        test_sklearn_compatibility()
        test_pytorch_compatibility()
        print("-" * 50)
        print("✓ すべてのテストが成功しました")
    except Exception as e:
        print("-" * 50)
        print(f"✗ テスト失敗: {str(e)}")
