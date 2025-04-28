"""
ファイル名: model.py
作成日: 2024/03/01
更新日: 2024/03/01
作成者: KeibaAI Team
説明: 競馬予測のための機械学習モデルを実装するモジュール
     RandomForestを使用してレース結果の予測を行います
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import sys

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger


class RacePredictionModel:
    """
    クラス名: RacePredictionModel
    説明: 競馬予測のための機械学習モデルを管理するクラス
         RandomForestを使用してレース結果の予測を行います

    主な機能:
    - prepare_data: データの前処理と特徴量エンジニアリング
    - train: モデルの学習と評価
    - predict: 新しいデータに対する予測

    属性:
    - model: RandomForestClassifier - 予測に使用する機械学習モデル
    - model_dir: Path - モデルの保存先ディレクトリ
    """

    def __init__(self):
        # ===========================================
        # モデルの初期化
        # ===========================================
        logger.info("RacePredictionModelの初期化を開始します")
        # 予測モデルのインスタンス
        # - n_estimators: 決定木の数
        # - random_state: 乱数シード
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # モデルファイルの保存先ディレクトリ
        self.model_dir = Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"モデル保存先: {self.model_dir}")

    def prepare_data(self, data_path: str) -> tuple:
        """
        メソッド名: prepare_data
        説明: データの前処理と特徴量エンジニアリングを行います

        引数:
        - data_path: str - 入力データのパス

        戻り値:
        - tuple: (X, y) - 特徴量と目的変数のタプル
            - X: pd.DataFrame - 特徴量データ
            - y: pd.Series - 目的変数データ

        例外:
        - FileNotFoundError - 指定されたパスにファイルが存在しない場合
        - ValueError - データの形式が不正な場合
        """
        logger.info(f"データの前処理を開始します: {data_path}")

        try:
            # ===========================================
            # データの読み込み
            # ===========================================
            logger.debug("データファイルを読み込みます")
            # レースデータを格納するDataFrame
            # 列: ['race_id', 'date', 'place', 'distance', ...]
            df = pd.read_csv(data_path)
            logger.info(f"データの読み込みが完了しました: {len(df)} 行")

            # TODO: 特徴量エンジニアリングの実装
            logger.warning("特徴量エンジニアリングが未実装です")
            # 以下の特徴量を追加する必要があります：
            # - 馬の過去の成績（直近5レースの平均着順など）
            # - 騎手の勝率（過去1年間の勝率など）
            # - コースの特徴（距離、天候、馬場状態など）
            # - 馬の特徴（年齢、性別、体重など）

            # 特徴量と目的変数の分離
            X = df.drop(["target"], axis=1)  # 実際のデータ構造に合わせて変更
            y = df["target"]

            logger.info("データの前処理が完了しました")
            return X, y

        except FileNotFoundError as e:
            logger.error(f"データファイルが見つかりません: {e}")
            raise
        except ValueError as e:
            logger.error(f"データの形式が不正です: {e}")
            raise
        except Exception as e:
            logger.error(f"予期せぬエラーが発生しました: {e}")
            raise

    def train(self, data_path: str) -> None:
        """
        メソッド名: train
        説明: モデルの学習と評価を行います

        引数:
        - data_path: str - 学習データのパス

        戻り値:
        - None

        例外:
        - FileNotFoundError - 指定されたパスにファイルが存在しない場合
        - ValueError - データの形式が不正な場合
        """
        logger.info(f"モデルの学習を開始します: {data_path}")

        try:
            # ===========================================
            # データの準備
            # ===========================================
            # 特徴量と目的変数の取得
            X, y = self.prepare_data(data_path)

            # 学習データとテストデータの分割
            # - test_size: テストデータの割合
            # - random_state: 乱数シード
            logger.info("データの分割を開始します")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logger.info(
                f"学習データ: {len(X_train)} 行, テストデータ: {len(X_test)} 行"
            )

            # ===========================================
            # モデルの学習
            # ===========================================
            logger.info("モデルの学習を開始します")
            self.model.fit(X_train, y_train)
            logger.info("モデルの学習が完了しました")

            # ===========================================
            # モデルの評価
            # ===========================================
            logger.info("モデルの評価を開始します")
            # テストデータでの予測
            y_pred = self.model.predict(X_test)
            # 予測精度の計算
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"モデルの精度: {accuracy:.2f}")

            # ===========================================
            # モデルの保存
            # ===========================================
            model_path = self.model_dir / "race_prediction_model.joblib"
            logger.info(f"モデルを保存します: {model_path}")
            joblib.dump(self.model, model_path)
            logger.info("モデルの保存が完了しました")

        except Exception as e:
            logger.error(f"モデルの学習中にエラーが発生しました: {e}")
            raise

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        メソッド名: predict
        説明: 新しいデータに対する予測を行います

        引数:
        - features: pd.DataFrame - 予測に使用する特徴量データ
            - 列: ['race_id', 'date', 'place', 'distance', ...]

        戻り値:
        - np.ndarray - 予測結果の配列
            - 各要素は予測された着順

        例外:
        - Exception - モデルが学習されていない場合
        """
        logger.info("予測を開始します")

        try:
            if not hasattr(self.model, "predict"):
                # ===========================================
                # モデルのロード
                # ===========================================
                model_path = self.model_dir / "race_prediction_model.joblib"
                if model_path.exists():
                    logger.info(f"モデルをロードします: {model_path}")
                    self.model = joblib.load(model_path)
                else:
                    raise Exception("モデルが学習されていません")

            # ===========================================
            # 予測の実行
            # ===========================================
            logger.debug("予測を実行します")
            predictions = self.model.predict(features)
            logger.info("予測が完了しました")

            return predictions

        except Exception as e:
            logger.error(f"予測中にエラーが発生しました: {e}")
            raise


if __name__ == "__main__":
    # ===========================================
    # テスト実行
    # ===========================================
    try:
        logger.info("テスト実行を開始します")
        model = RacePredictionModel()
        # 実際のデータパスを指定して学習
        # model.train("data/processed/training_data.csv")
        logger.info("テスト実行が完了しました")
    except Exception as e:
        logger.critical(f"テスト実行中にエラーが発生しました: {e}")
        sys.exit(1)
