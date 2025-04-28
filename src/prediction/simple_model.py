"""
オッズのみを特徴量として使用するシンプルな予測モデル
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


class SimplePredictor:
    """オッズベースの単純な予測モデル"""

    def __init__(self, threshold: float = 0.5):
        """
        モデルの初期化

        Args:
            threshold (float): 予測の閾値（デフォルト: 0.5）
        """
        self.model = LogisticRegression(random_state=42)
        self.threshold = threshold

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        モデルの学習を行う

        Args:
            X (np.ndarray): オッズデータ (n_samples, 1)
            y (np.ndarray): 結果データ (1: 3着以内, 0: その他)
        """
        # クラスの重みを計算
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))

        # 重み付きモデルの学習
        self.model = LogisticRegression(random_state=42, class_weight=class_weight_dict)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測を行う

        Args:
            X (np.ndarray): オッズデータ (n_samples, 1)

        Returns:
            np.ndarray: 予測結果 (1: 3着以内予測, 0: その他)
        """
        probas = self.predict_proba(X)
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        3着以内に入る確率を予測する

        Args:
            X (np.ndarray): オッズデータ (n_samples, 1)

        Returns:
            np.ndarray: 3着以内に入る確率 (0-1の範囲)
        """
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        モデルの評価を行う

        Args:
            X (np.ndarray): オッズデータ (n_samples, 1)
            y (np.ndarray): 実際の結果 (1: 3着以内, 0: その他)

        Returns:
            Dict[str, float]: 評価指標の辞書
        """
        y_pred = self.predict(X)

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
        }

    def find_optimal_threshold(
        self, X_val: np.ndarray, y_val: np.ndarray, target_precision: float = 0.5
    ) -> Tuple[float, Dict[str, float]]:
        """
        目標精度を達成する最適な閾値を探索する

        Args:
            X_val (np.ndarray): 検証用オッズデータ
            y_val (np.ndarray): 検証用結果データ
            target_precision (float): 目標精度

        Returns:
            Tuple[float, Dict[str, float]]: 最適な閾値と、その閾値での評価指標
        """
        probas = self.predict_proba(X_val)

        # 閾値の候補を生成（0.5から0.95まで）
        thresholds = np.arange(0.5, 0.96, 0.05)
        best_threshold = 0.5
        best_metrics = None

        for threshold in thresholds:
            y_pred = (probas >= threshold).astype(int)
            precision = precision_score(y_val, y_pred)

            if precision >= target_precision:
                best_threshold = threshold
                best_metrics = {
                    "accuracy": accuracy_score(y_val, y_pred),
                    "precision": precision,
                    "recall": recall_score(y_val, y_pred),
                    "f1": f1_score(y_val, y_pred),
                }
                break

        return best_threshold, best_metrics
