"""
実データを使用したモデル評価
3着以内に入るかどうかを予測
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.prediction.simple_model import SimplePredictor
from src.data_collection.load_race_data import combine_race_data
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    データの前処理を行う

    Args:
        df (pd.DataFrame): 元のデータフレーム

    Returns:
        pd.DataFrame: 前処理済みのデータフレーム
    """
    # 無効なデータを除外
    df = df[df["オッズ"] != "---"]

    # 特殊な着順（中止、除外など）を除外
    df = df[df["着順"].str.isdigit()]

    # データ型を変換
    df["オッズ"] = df["オッズ"].astype(float)
    df["着順"] = df["着順"].astype(int)

    return df


def evaluate_with_threshold(
    predictor: SimplePredictor, X: np.ndarray, y: np.ndarray, threshold: float
) -> dict:
    """
    指定された閾値でモデルを評価する

    Args:
        predictor (SimplePredictor): 予測モデル
        X (np.ndarray): 特徴量
        y (np.ndarray): 正解ラベル
        threshold (float): 予測閾値

    Returns:
        dict: 評価指標
    """
    predictor.threshold = threshold
    y_pred = predictor.predict(X)
    n_samples = len(y)
    n_positive = y.sum()
    n_predicted = y_pred.sum()

    metrics = {
        "accuracy": (y == y_pred).mean(),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "positive_rate": n_positive / n_samples,  # 実際の3着以内の割合
        "prediction_rate": n_predicted / n_samples,  # モデルが3着以内と予測した割合
    }

    return metrics


def evaluate_odds_range(
    predictor: SimplePredictor, X: np.ndarray, y: np.ndarray, start: float, end: float
) -> dict:
    """
    指定されたオッズ範囲でモデルを評価する

    Args:
        predictor (SimplePredictor): 予測モデル
        X (np.ndarray): 特徴量
        y (np.ndarray): 正解ラベル
        start (float): オッズの下限
        end (float): オッズの上限

    Returns:
        dict: 評価指標
    """
    mask = (X[:, 0] >= start) & (X[:, 0] < end)
    if not mask.any():
        return None

    X_range = X[mask]
    y_range = y[mask]
    y_pred = predictor.predict(X_range)
    n_samples = len(y_range)
    n_positive = y_range.sum()
    n_predicted = y_pred.sum()

    return {
        "accuracy": (y_range == y_pred).mean(),
        "precision": precision_score(y_range, y_pred) if n_predicted > 0 else 0,
        "recall": recall_score(y_range, y_pred),
        "f1": f1_score(y_range, y_pred) if n_predicted > 0 else 0,
        "n_samples": n_samples,
        "positive_rate": n_positive / n_samples,
        "prediction_rate": n_predicted / n_samples,
    }


def plot_calibration_curve(predictor, X, y, save_path="calibration_curve.png"):
    """
    モデルのキャリブレーションカーブを描画・保存
    """
    # 日本語フォント設定（AppleGothicがmacOS標準）
    import matplotlib

    try:
        matplotlib.rcParams["font.family"] = "AppleGothic"
    except:
        matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
    matplotlib.rcParams["axes.unicode_minus"] = False
    # 確率出力
    if hasattr(predictor, "predict_proba"):
        prob_proba = predictor.predict_proba(X)
        if prob_proba.ndim == 1:
            prob_pos = prob_proba
        else:
            prob_pos = prob_proba[:, 1]
    else:
        # シンプルなモデルの場合はpredictで代用
        prob_pos = predictor.predict(X)

    # 区間ごとに分けて平均予測確率と実際の3着以内率を計算
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y, prob_pos, n_bins=10
    )
    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="モデル")
    plt.plot([0, 1], [0, 1], "k--", label="理想")
    plt.xlabel("モデルの平均予測確率")
    plt.ylabel("実際の3着以内率")
    plt.title("キャリブレーションカーブ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"キャリブレーションカーブを {save_path} に保存しました。")


def evaluate_model_with_real_data():
    """実データを使用してモデルを評価"""
    # データの読み込み
    years = range(2018, 2023)
    df = combine_race_data(years)

    # データの前処理
    df = preprocess_data(df)

    # データの準備
    X = np.log(df["オッズ"].values).reshape(-1, 1)
    y = (df["着順"] <= 3).astype(int).values  # 3着以内を1、それ以外を0

    # データの分割（訓練:検証:テスト = 60:20:20）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # モデルの学習
    predictor = SimplePredictor()
    predictor.train(X_train, y_train)

    # モデルパラメータのログ出力（ロジスティック回帰の場合のみ）
    if hasattr(predictor, "model") and hasattr(predictor.model, "coef_"):
        logger.info(
            f"モデルのパラメータ: coef_={predictor.model.coef_}, intercept_={predictor.model.intercept_}"
        )

    # キャリブレーションカーブの作成
    plot_calibration_curve(predictor, X_test, y_test)

    # 様々な閾値でテスト
    thresholds = np.arange(0.50, 0.601, 0.01)
    logger.info("\n=== 閾値別の評価結果 ===")

    for threshold in thresholds:
        metrics = evaluate_with_threshold(predictor, X_test, y_test, threshold)
        logger.info(f"\n閾値 {threshold:.2f} での評価結果:")
        logger.info(f"正解率 (Accuracy): {metrics['accuracy']:.3f}")
        logger.info(f"適合率 (Precision): {metrics['precision']:.3f}")
        logger.info(f"再現率 (Recall): {metrics['recall']:.3f}")
        logger.info(f"F1スコア: {metrics['f1']:.3f}")
        logger.info(f"3着以内率: {metrics['positive_rate']:.3f}")
        logger.info(f"予測率: {metrics['prediction_rate']:.3f}")

        # オッズ別の評価
        # log(オッズ)スケールで区切りを設定
        odds_ranges = [
            (0.00, 0.70),  # オッズ1.0～2.0
            (0.70, 1.60),  # オッズ2.0～5.0
            (1.60, 2.30),  # オッズ5.0～10.0
            (2.30, 3.00),  # オッズ10.0～20.0
            (3.00, np.inf),  # オッズ20.0以上
        ]
        logger.info("\nオッズ別の評価結果:")

        for start, end in odds_ranges:
            metrics = evaluate_odds_range(predictor, X_test, y_test, start, end)
            if metrics:
                logger.info(
                    f"\nオッズ {start}-{end}:"
                    f"\n  サンプル数: {metrics['n_samples']}"
                    f"\n  正解率: {metrics['accuracy']:.3f}"
                    f"\n  適合率: {metrics['precision']:.3f}"
                    f"\n  再現率: {metrics['recall']:.3f}"
                    f"\n  F1スコア: {metrics['f1']:.3f}"
                    f"\n  3着以内率: {metrics['positive_rate']:.3f}"
                    f"\n  予測率: {metrics['prediction_rate']:.3f}"
                )


def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)
    evaluate_model_with_real_data()


if __name__ == "__main__":
    main()
