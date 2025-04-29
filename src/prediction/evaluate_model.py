"""
実データを使用したモデル評価
3着以内に入るかどうかを予測
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from src.data_collection.load_race_data import combine_race_data
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    データの前処理を行う

    Args:
        df (pd.DataFrame): 元のデータフレーム

    Returns:
        pd.DataFrame: 前処理済みのデータフレーム
    """
    # カラム名を表示
    logger.info(f"データフレームのカラム: {df.columns.tolist()}")

    # 無効なデータを除外
    df = df[df["オッズ"] != "---"]

    # 特殊な着順（中止、除外など）を除外
    df = df[df["着順"].str.isdigit()]

    # データ型を変換
    df["オッズ"] = df["オッズ"].astype(float)
    df["着順"] = df["着順"].astype(int)

    # 出走日を日付型に変換
    df["日付"] = (
        df["日付"]
        .str.extract(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
        .apply(lambda x: f"{x[0]}-{int(x[1]):02d}-{int(x[2]):02d}", axis=1)
    )
    df["日付"] = pd.to_datetime(df["日付"])

    # 全データを馬ごと、日付順にソート
    df = df.sort_values(["馬", "日付", "race_id"], ascending=True)

    # 前回出走日の計算
    df["last_race_date"] = df.groupby("馬")["日付"].shift(1)

    # レース間隔の計算
    df["days_since_last_race"] = (df["日付"] - df["last_race_date"]).dt.days

    # デビュー戦フラグの作成
    df["is_debut"] = df["last_race_date"].isna().astype(int)

    # レース間隔の欠損値はNaNのままにする（LightGBMが自動的に処理）
    # df['days_since_last_race'] = df['days_since_last_race'].fillna(-1)

    return df


def plot_calibration_curve(model, X, y, save_path="calibration_curve.png"):
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
    prob_pos = model.predict(X)

    # 区間ごとに分けて平均予測確率と実際の3着以内率を計算
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y, prob_pos, n_bins=10
    )

    # キャリブレーションカーブの数値をログに出力
    logger.info("\n=== キャリブレーションカーブの数値 ===")
    for i, (pred, actual) in enumerate(
        zip(mean_predicted_value, fraction_of_positives)
    ):
        logger.info(
            f"区間 {i+1}: 平均予測確率={pred:.3f}, 実際の3着以内率={actual:.3f}"
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


def analyze_by_weight_range(X_test, y_test, y_pred, weight_ranges):
    """斤量範囲ごとの分析"""
    logger.info("\n=== 斤量別の評価結果 ===")
    for start, end in weight_ranges:
        # 元の斤量に戻す
        weight_start = start / 2 + 48
        weight_end = end / 2 + 48
        mask = (X_test[:, 1] >= start) & (X_test[:, 1] < end)
        if not mask.any():
            continue

        y_range = y_test[mask]
        y_pred_range = y_pred[mask]

        metrics = {
            "accuracy": (y_range == y_pred_range).mean(),
            "precision": (
                precision_score(y_range, y_pred_range) if y_pred_range.sum() > 0 else 0
            ),
            "recall": recall_score(y_range, y_pred_range),
            "f1": f1_score(y_range, y_pred_range) if y_pred_range.sum() > 0 else 0,
            "n_samples": len(y_range),
            "positive_rate": y_range.mean(),
            "prediction_rate": y_pred_range.mean(),
        }

        logger.info(
            f"\n斤量 {weight_start:.1f}-{weight_end:.1f}kg:"
            f"\n  サンプル数: {metrics['n_samples']}"
            f"\n  正解率: {metrics['accuracy']:.3f}"
            f"\n  適合率: {metrics['precision']:.3f}"
            f"\n  再現率: {metrics['recall']:.3f}"
            f"\n  F1スコア: {metrics['f1']:.3f}"
            f"\n  3着以内率: {metrics['positive_rate']:.3f}"
            f"\n  予測率: {metrics['prediction_rate']:.3f}"
        )


def analyze_odds_weight_cross(X_test, y_test, y_pred, odds_ranges, weight_ranges):
    """オッズと斤量のクロス分析"""
    logger.info("\n=== オッズ×斤量のクロス分析 ===")
    for o_start, o_end in odds_ranges:
        for w_start, w_end in weight_ranges:
            # 元の値に戻す
            odds_start = np.exp(o_start)
            odds_end = np.exp(o_end)
            weight_start = w_start / 2 + 48
            weight_end = w_end / 2 + 48

            # オッズと斤量の両方の条件を満たすマスク
            mask = (
                (X_test[:, 0] >= o_start)
                & (X_test[:, 0] < o_end)
                & (X_test[:, 1] >= w_start)
                & (X_test[:, 1] < w_end)
            )
            if not mask.any():
                continue

            y_range = y_test[mask]
            y_pred_range = y_pred[mask]

            if len(y_range) < 50:  # サンプル数が少なすぎる場合はスキップ
                continue

            metrics = {
                "accuracy": (y_range == y_pred_range).mean(),
                "precision": (
                    precision_score(y_range, y_pred_range)
                    if y_pred_range.sum() > 0
                    else 0
                ),
                "recall": recall_score(y_range, y_pred_range),
                "f1": f1_score(y_range, y_pred_range) if y_pred_range.sum() > 0 else 0,
                "n_samples": len(y_range),
                "positive_rate": y_range.mean(),
                "prediction_rate": y_pred_range.mean(),
            }

            logger.info(
                f"\nオッズ {odds_start:.1f}-{odds_end:.1f} × 斤量 {weight_start:.1f}-{weight_end:.1f}kg:"
                f"\n  サンプル数: {metrics['n_samples']}"
                f"\n  正解率: {metrics['accuracy']:.3f}"
                f"\n  適合率: {metrics['precision']:.3f}"
                f"\n  再現率: {metrics['recall']:.3f}"
                f"\n  F1スコア: {metrics['f1']:.3f}"
                f"\n  3着以内率: {metrics['positive_rate']:.3f}"
                f"\n  予測率: {metrics['prediction_rate']:.3f}"
            )


def analyze_prediction_probability(model, X_test, y_test, prob_ranges):
    """予測確率帯ごとの分析"""
    logger.info("\n=== 予測確率帯ごとの評価結果 ===")
    probs = model.predict(X_test)

    for start, end in prob_ranges:
        mask = (probs >= start) & (probs < end)
        if not mask.any():
            continue

        y_range = y_test[mask]
        y_pred_range = (probs[mask] >= 0.5).astype(int)  # 0.5を基準とした予測

        metrics = {
            "accuracy": (y_range == y_pred_range).mean(),
            "precision": (
                precision_score(y_range, y_pred_range) if y_pred_range.sum() > 0 else 0
            ),
            "recall": recall_score(y_range, y_pred_range),
            "f1": f1_score(y_range, y_pred_range) if y_pred_range.sum() > 0 else 0,
            "n_samples": len(y_range),
            "positive_rate": y_range.mean(),
            "prediction_rate": y_pred_range.mean(),
        }

        logger.info(
            f"\n予測確率 {start:.2f}-{end:.2f}:"
            f"\n  サンプル数: {metrics['n_samples']}"
            f"\n  正解率: {metrics['accuracy']:.3f}"
            f"\n  適合率: {metrics['precision']:.3f}"
            f"\n  再現率: {metrics['recall']:.3f}"
            f"\n  F1スコア: {metrics['f1']:.3f}"
            f"\n  3着以内率: {metrics['positive_rate']:.3f}"
            f"\n  予測率: {metrics['prediction_rate']:.3f}"
        )


def analyze_interval_top3_rate(df: pd.DataFrame):
    """レース間隔と3着以内率の関係を詳細に分析"""
    logger.info("\n=== レース間隔と3着以内率の詳細分析 ===")

    # デビュー戦を除外
    non_debut_df = df[df["days_since_last_race"].notna()]

    # 間隔を5日刻みでグループ化
    interval_bins = np.arange(0, 365, 5)
    interval_labels = [f"{i}-{i+5}" for i in interval_bins[:-1]]

    # 各間隔帯の3着以内率を計算
    non_debut_df["interval_bin"] = pd.cut(
        non_debut_df["days_since_last_race"],
        bins=interval_bins,
        labels=interval_labels,
        right=False,
    )

    interval_stats = (
        non_debut_df.groupby("interval_bin")
        .agg({"着順": lambda x: (x <= 3).mean(), "馬": "count"})
        .rename(columns={"着順": "top3_rate", "馬": "count"})
    )

    # 結果を表示
    for interval, stats in interval_stats.iterrows():
        if stats["count"] >= 100:  # サンプル数が100以上の区間のみ表示
            logger.info(
                f"間隔 {interval}日:"
                f"\n  サンプル数: {stats['count']}"
                f"\n  3着以内率: {stats['top3_rate']:.3f}"
            )

    # 長期休養馬の分析
    long_rest_bins = [0, 180, 365, 730, 10000]
    long_rest_labels = ["0-180日", "180-365日", "365-730日", "730日以上"]

    long_rest_df = df[df["days_since_last_race"].notna()].copy()
    long_rest_df["rest_category"] = pd.cut(
        long_rest_df["days_since_last_race"],
        bins=long_rest_bins,
        labels=long_rest_labels,
        right=False,
    )

    long_rest_stats = (
        long_rest_df.groupby("rest_category")
        .agg({"着順": lambda x: (x <= 3).mean(), "馬": "count"})
        .rename(columns={"着順": "top3_rate", "馬": "count"})
    )

    logger.info("\n=== 長期休養馬の分析 ===")
    for category, stats in long_rest_stats.iterrows():
        logger.info(
            f"休養期間 {category}:"
            f"\n  サンプル数: {stats['count']}"
            f"\n  3着以内率: {stats['top3_rate']:.3f}"
        )


def analyze_by_interval_category(X_test, y_test, y_pred):
    """レース間隔カテゴリ別の分析"""
    logger.info("\n=== レース間隔カテゴリ別の分析 ===")

    for category in X_test["interval_category"].unique():
        mask = X_test["interval_category"] == category
        if not mask.any():
            continue

        y_range = y_test[mask.index[mask]]
        y_pred_range = y_pred[mask.index[mask]]

        metrics = {
            "accuracy": accuracy_score(y_range, y_pred_range),
            "precision": (
                precision_score(y_range, y_pred_range) if y_pred_range.sum() > 0 else 0
            ),
            "recall": recall_score(y_range, y_pred_range),
            "f1": f1_score(y_range, y_pred_range) if y_pred_range.sum() > 0 else 0,
            "n_samples": len(y_range),
            "positive_rate": y_range.mean(),
            "prediction_rate": y_pred_range.mean(),
        }

        logger.info(
            f"\n間隔カテゴリ {category}:"
            f"\n  サンプル数: {metrics['n_samples']}"
            f"\n  正解率: {metrics['accuracy']:.3f}"
            f"\n  適合率: {metrics['precision']:.3f}"
            f"\n  再現率: {metrics['recall']:.3f}"
            f"\n  F1スコア: {metrics['f1']:.3f}"
            f"\n  3着以内率: {metrics['positive_rate']:.3f}"
            f"\n  予測率: {metrics['prediction_rate']:.3f}"
        )


def prepare_features(data_df):
    """特徴量の準備"""
    # レース間隔のカテゴリ分け（ドメイン知識ベース）
    data_df["interval_category"] = pd.cut(
        data_df["days_since_last_race"],
        bins=[-1, 0, 8, 15, 35, 91, 182, 365, float("inf")],
        labels=[
            "デビュー",
            "連闘",
            "中1週",
            "中2-4週",
            "1-3ヶ月",
            "3-6ヶ月",
            "6ヶ月-1年",
            "1年以上",
        ],
        right=False,
    )

    # カテゴリの統計情報を表示
    logger.info("\n=== ドメイン知識ベースのカテゴリ分けの統計 ===")
    for category in data_df["interval_category"].unique():
        if pd.isna(category):
            continue
        mask = data_df["interval_category"] == category
        category_df = data_df[mask]
        days_stats = category_df["days_since_last_race"].agg(["min", "max", "mean"])
        top3_rate = (category_df["着順"] <= 3).mean()
        sample_size = len(category_df)

        logger.info(
            f"\nカテゴリ {category}:"
            f"\n  サンプル数: {sample_size}"
            f"\n  間隔範囲: {days_stats['min']:.1f}-{days_stats['max']:.1f}日"
            f"\n  平均間隔: {days_stats['mean']:.1f}日"
            f"\n  3着以内率: {top3_rate:.3f}"
        )

    return pd.DataFrame(
        {
            "log_odds": np.log(data_df["オッズ"].values),
            "weight_scaled": (data_df["斤量"].values - 48) * 2,
            "interval_category": data_df["interval_category"].values,
            "is_debut": data_df["is_debut"].values,
        },
        index=data_df.index,
    )


def generate_feature_documentation(model, feature_names: list) -> str:
    """
    特徴量の情報をマークダウン形式で生成する

    Args:
        model: 学習済みのモデル
        feature_names: 特徴量名のリスト

    Returns:
        str: マークダウン形式のドキュメント
    """
    feature_importance = model.feature_importance()

    doc = f"""
# モデル評価レポート

## 最終更新日時
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 使用特徴量一覧

### 基本特徴量
- **オッズ関連**
  - `log_odds`: オッズの対数変換値
    - 変換方法: `np.log(data_df["オッズ"].values)`
    - 重要度: {feature_importance[0]}

- **斤量関連**
  - `weight_scaled`: スケーリングされた斤量
    - 変換方法: `(data_df["斤量"].values - 48) * 2`
    - 重要度: {feature_importance[1]}

### レース間隔関連特徴量
- **interval_category**: レース間隔のカテゴリ分類
  - カテゴリ:
    1. `デビュー`: 初出走
    2. `連闘`: 0-8日
    3. `中1週`: 8-15日
    4. `中2-4週`: 15-35日
    5. `1-3ヶ月`: 35-91日
    6. `3-6ヶ月`: 91-182日
    7. `6ヶ月-1年`: 182-365日
    8. `1年以上`: 365日超
  - 重要度: {feature_importance[2]}

- **is_debut**: デビュー戦フラグ
  - 値: `0` または `1`
  - 計算方法: `last_race_date.isna().astype(int)`
  - 重要度: {feature_importance[3]}

### 時系列関連
- **days_since_last_race**: 前走からの経過日数
  - 計算方法: `(df["日付"] - df["last_race_date"]).dt.days`
  - 注意: デビュー戦の場合はNaN

### 目的変数
- **3着以内フラグ**
  - 計算方法: `(着順 <= 3).astype(int)`
  - 値: `0`（3着より下）または `1`（3着以内）

## 特徴量の重要度ランキング
"""
    # 特徴量の重要度をソートして追加
    importance_pairs = list(zip(feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    for name, importance in importance_pairs:
        doc += f"- {name}: {importance}\n"

    return doc


def save_evaluation_history(
    metrics: dict, feature_importance: list, feature_names: list
) -> None:
    """
    評価履歴を保存する

    Args:
        metrics: 評価指標の辞書
        feature_importance: 特徴量の重要度のリスト
        feature_names: 特徴量名のリスト
    """
    # 履歴ディレクトリの作成
    history_dir = Path("model_history")
    history_dir.mkdir(exist_ok=True)

    # 現在の日時を取得
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 評価結果を辞書にまとめる
    evaluation_data = {
        "timestamp": timestamp,
        "metrics": metrics,
        "feature_importance": dict(zip(feature_names, feature_importance.tolist())),
    }

    # JSONファイルとして保存
    history_file = history_dir / f"evaluation_{timestamp}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)

    logger.info(f"評価履歴を保存しました: {history_file}")


def generate_history_summary() -> str:
    """
    評価履歴のサマリーを生成する

    Returns:
        str: マークダウン形式のサマリー
    """
    history_dir = Path("model_history")
    if not history_dir.exists():
        return "評価履歴がありません"

    # 履歴ファイルの一覧を取得
    history_files = sorted(history_dir.glob("evaluation_*.json"))
    if not history_files:
        return "評価履歴がありません"

    # サマリーの生成
    summary = "# モデル評価履歴\n\n"

    # 各履歴ファイルからデータを読み込んでサマリーに追加
    for file in history_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary += f"## {data['timestamp']}\n\n"
        summary += "### 評価指標\n"
        for metric, value in data["metrics"].items():
            summary += f"- {metric}: {value:.3f}\n"

        summary += "\n### 特徴量の重要度\n"
        for feature, importance in sorted(
            data["feature_importance"].items(), key=lambda x: x[1], reverse=True
        ):
            summary += f"- {feature}: {importance}\n"

        summary += "\n---\n\n"

    return summary


def update_model_documentation(model, feature_names: list, metrics: dict) -> None:
    """
    モデルのドキュメントを更新する

    Args:
        model: 学習済みのモデル
        feature_names: 特徴量名のリスト
        metrics: 評価指標の辞書
    """
    doc = generate_feature_documentation(model, feature_names)

    # 評価指標の情報を追加
    doc += "\n## モデル評価指標\n"

    # 全体評価指標
    doc += "\n### 全体評価指標\n"
    for metric_name, value in metrics.items():
        if not metric_name.endswith("_at_3"):
            doc += f"- {metric_name}: {value:.3f}\n"

    # レース単位評価指標
    doc += "\n### レース単位評価指標\n"
    doc += (
        "- Precision@3: 各レースで上位3頭予測に含まれた実際の3着以内馬の割合（平均）\n"
    )
    doc += f"  - 値: {metrics.get('precision_at_3', 0):.3f}\n"
    doc += "- Recall@3: 各レースで実際の3着以内馬のうち、上位3頭予測に含まれた割合（平均）\n"
    doc += f"  - 値: {metrics.get('recall_at_3', 0):.3f}\n"
    doc += "- Hit Rate@3: 上位3頭予測に1頭でも3着以内馬が含まれていたレースの割合\n"
    doc += f"  - 値: {metrics.get('hit_rate_at_3', 0):.3f}\n"
    doc += "- NDCG@3: 予測順位の質を評価する指標（1.0が最高値）\n"
    doc += f"  - 値: {metrics.get('ndcg_at_3', 0):.3f}\n"
    doc += "  - 説明: 上位3頭の予測順位が実際の3着以内馬の順位にどれだけ近いかを評価\n"
    doc += "  - 計算方法: DCG（予測順位の利得）をIDCG（理想的な順位の利得）で正規化\n"

    # 履歴サマリーを追加
    doc += "\n## 評価履歴\n"
    doc += generate_history_summary()

    # ドキュメントの保存先を設定
    docs_path = Path("model_features.md")

    # ドキュメントを更新
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(doc)

    logger.info(f"モデルのドキュメントを更新しました: {docs_path}")


def calculate_ndcg(y_true: pd.Series, y_pred_proba: pd.Series, k: int = 3) -> float:
    """
    NDCG@kを計算する

    Args:
        y_true: 実際の3着以内フラグ
        y_pred_proba: モデルの予測確率
        k: 上位何頭を評価するか

    Returns:
        float: NDCG@kの値
    """
    # 予測確率で降順ソート
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    sorted_true = y_true.iloc[sorted_indices]

    # 理想的な順位（実際の3着以内馬が上位に来る順位）
    ideal_indices = np.argsort(y_true)[::-1]
    ideal_true = y_true.iloc[ideal_indices]

    # DCGの計算
    dcg = 0
    for i in range(min(k, len(y_true))):
        dcg += (2 ** sorted_true.iloc[i] - 1) / np.log2(i + 2)

    # IDCGの計算
    idcg = 0
    for i in range(min(k, len(y_true))):
        idcg += (2 ** ideal_true.iloc[i] - 1) / np.log2(i + 2)

    # NDCGの計算（IDCGが0の場合は1を返す）
    return dcg / idcg if idcg > 0 else 1.0


def calculate_race_metrics(
    y_true: pd.Series, y_pred_proba: pd.Series, race_ids: pd.Series, k: int = 3
) -> dict:
    """
    レース単位の評価指標を計算する

    Args:
        y_true: 実際の3着以内フラグ
        y_pred_proba: モデルの予測確率
        race_ids: レースID
        k: 上位何頭を評価するか

    Returns:
        dict: 評価指標の辞書
    """
    # レースごとにグループ化
    race_groups = pd.DataFrame(
        {"y_true": y_true, "y_pred_proba": y_pred_proba, "race_id": race_ids}
    ).groupby("race_id")

    # 各レースでの指標を計算
    precision_at_k = []
    recall_at_k = []
    hit_rate_at_k = []
    ndcg_at_k = []

    for race_id, group in race_groups:
        # 予測確率で降順ソート
        sorted_group = group.sort_values("y_pred_proba", ascending=False)

        # 上位k頭を取得
        top_k = sorted_group.head(k)

        # Precision@k: 上位k頭の中の実際の3着以内馬の割合
        precision = top_k["y_true"].sum() / k
        precision_at_k.append(precision)

        # Recall@k: 実際の3着以内馬のうち、上位k頭に含まれる割合
        actual_top3 = group["y_true"].sum()
        if actual_top3 > 0:
            recall = top_k["y_true"].sum() / actual_top3
            recall_at_k.append(recall)

        # Hit Rate@k: 上位k頭に1頭でも3着以内馬が含まれていたか
        hit_rate = 1 if top_k["y_true"].sum() > 0 else 0
        hit_rate_at_k.append(hit_rate)

        # NDCG@kの計算
        ndcg = calculate_ndcg(group["y_true"], group["y_pred_proba"], k)
        ndcg_at_k.append(ndcg)

    # 全レースでの平均を計算
    metrics = {
        "precision_at_3": np.mean(precision_at_k),
        "recall_at_3": np.mean(recall_at_k),
        "hit_rate_at_3": np.mean(hit_rate_at_k),
        "ndcg_at_3": np.mean(ndcg_at_k),
    }

    return metrics


def evaluate_model_with_real_data():
    """実データを使用してモデルを評価"""
    # データの読み込みと前処理
    data_df = preprocess_data(combine_race_data(range(2018, 2023)))

    # 訓練データとテストデータの分割
    train_df = data_df[data_df["日付"].dt.year <= 2021].copy()
    test_df = data_df[data_df["日付"].dt.year == 2022].copy()
    logger.info(f"訓練データ: {len(train_df)}行, テストデータ: {len(test_df)}行")

    # 特徴量の準備
    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    # 目的変数の準備
    y_train = pd.Series((train_df["着順"] <= 3).astype(int), index=train_df.index)
    y_test = pd.Series((test_df["着順"] <= 3).astype(int), index=test_df.index)

    # LightGBMのパラメータ設定
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    # カテゴリ特徴量の指定
    categorical_features = ["interval_category"]

    # データセットの作成
    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_features
    )
    test_data = lgb.Dataset(
        X_test,
        label=y_test,
        reference=train_data,
        categorical_feature=categorical_features,
    )

    # モデルの学習
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10)],
    )

    # 特徴量の重要度を表示
    feature_importance = model.feature_importance()
    feature_names = ["log(オッズ)", "斤量", "interval_category", "デビュー戦フラグ"]
    logger.info("\n=== 特徴量の重要度 ===")
    for name, importance in zip(feature_names, feature_importance):
        logger.info(f"{name}: {importance}")

    # キャリブレーションカーブの作成
    plot_calibration_curve(model, X_test, y_test)

    # 閾値探索
    logger.info("\n=== 閾値探索結果 ===")
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}

    for threshold in thresholds:
        y_pred = pd.Series(
            (model.predict(X_test) >= threshold).astype(int), index=X_test.index
        )
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        prediction_rate = y_pred.mean()

        logger.info(
            f"\n閾値 {threshold:.2f}:"
            f"\n  F1スコア: {f1:.3f}"
            f"\n  正解率: {accuracy:.3f}"
            f"\n  適合率: {precision:.3f}"
            f"\n  再現率: {recall:.3f}"
            f"\n  予測率: {prediction_rate:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "prediction_rate": prediction_rate,
            }

    logger.info(f"\n最適な閾値: {best_threshold:.2f} (F1スコア: {best_f1:.3f})")
    logger.info(f"最適な閾値での性能:")
    logger.info(f"  正解率: {best_metrics['accuracy']:.3f}")
    logger.info(f"  適合率: {best_metrics['precision']:.3f}")
    logger.info(f"  再現率: {best_metrics['recall']:.3f}")
    logger.info(f"  予測率: {best_metrics['prediction_rate']:.3f}")

    # 最適な閾値をモデルに保存
    model.best_threshold = best_threshold

    # 最適な閾値での詳細分析
    y_pred = pd.Series(
        (model.predict(X_test) >= best_threshold).astype(int), index=X_test.index
    )

    # レース間隔カテゴリ別の分析
    analyze_by_interval_category(X_test, y_test, y_pred)

    # デビュー戦と非デビュー戦の比較分析
    logger.info("\n=== デビュー戦 vs 非デビュー戦の比較 ===")
    for is_debut in [0, 1]:
        mask = X_test["is_debut"] == is_debut
        if not mask.any():
            continue

        y_range = y_test[mask.index[mask]]
        y_pred_range = y_pred[mask.index[mask]]

        metrics = {
            "accuracy": accuracy_score(y_range, y_pred_range),
            "precision": (
                precision_score(y_range, y_pred_range) if y_pred_range.sum() > 0 else 0
            ),
            "recall": recall_score(y_range, y_pred_range),
            "f1": f1_score(y_range, y_pred_range) if y_pred_range.sum() > 0 else 0,
            "n_samples": len(y_range),
            "positive_rate": y_range.mean(),
            "prediction_rate": y_pred_range.mean(),
        }

        logger.info(
            f"\n{'デビュー戦' if is_debut else '非デビュー戦'}:"
            f"\n  サンプル数: {metrics['n_samples']}"
            f"\n  正解率: {metrics['accuracy']:.3f}"
            f"\n  適合率: {metrics['precision']:.3f}"
            f"\n  再現率: {metrics['recall']:.3f}"
            f"\n  F1スコア: {metrics['f1']:.3f}"
            f"\n  3着以内率: {metrics['positive_rate']:.3f}"
            f"\n  予測率: {metrics['prediction_rate']:.3f}"
        )

    # 検証用：2022年内で複数回出走している馬のレース間隔を確認
    logger.info("\n=== 検証: 2022年の複数回出走馬のレース間隔 ===")
    multiple_runners = test_df.groupby("馬").filter(lambda x: len(x) > 1)
    if not multiple_runners.empty:
        sample_horses = multiple_runners["馬"].unique()[
            :3
        ]  # 最大3頭をサンプルとして表示
        for horse in sample_horses:
            horse_races = test_df[test_df["馬"] == horse].sort_values("日付")
            logger.info(f"\n馬: {horse}")
            for _, race in horse_races.iterrows():
                logger.info(
                    f"  日付: {race['日付'].strftime('%Y-%m-%d')}, "
                    f"前走: {race['last_race_date'].strftime('%Y-%m-%d') if pd.notna(race['last_race_date']) else 'なし'}, "
                    f"間隔: {int(race['days_since_last_race']) if race['days_since_last_race'] >= 0 else 'デビュー'}"
                )

    # モデル評価後、以下のコードを追加
    metrics = {
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1_score": best_f1,
        "best_threshold": best_threshold,
    }

    # レース単位の評価指標を計算
    y_pred_proba = model.predict(X_test)
    race_metrics = calculate_race_metrics(y_test, y_pred_proba, test_df["race_id"])
    metrics.update(race_metrics)

    feature_names = ["log(オッズ)", "斤量", "interval_category", "デビュー戦フラグ"]

    # 評価履歴の保存
    save_evaluation_history(metrics, model.feature_importance(), feature_names)

    # ドキュメントの更新
    update_model_documentation(model, feature_names, metrics)

    return model


def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)
    evaluate_model_with_real_data()


if __name__ == "__main__":
    main()
