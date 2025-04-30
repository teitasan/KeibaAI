"""
特徴量生成を担当するクラス群

このモジュールでは、生データから特徴量を生成するクラスを提供します。
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    特徴量生成の基本クラス
    
    生のレースデータから特徴量を生成し、訓練用・テスト用に分割します。
    このクラスは共通の特徴量生成パイプラインを提供します。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        特徴量生成器の初期化
        
        Args:
            config (Dict[str, Any], optional): 設定情報辞書
                - start_year (int): データの開始年
                - end_year (int): データの終了年
                - test_year (int): テストデータの年
                - features (List[str]): 使用する特徴量のリスト
                - target (str): 目的変数
        """
        # デフォルト設定
        self.default_config = {
            'start_year': 2018,
            'end_year': 2023,
            'test_year': 2022,
            'features': ['odds', 'weight', 'interval', 'track_condition', 'track_aptitude'],
            'target': 'top3'
        }
        
        # 設定の適用（ユーザー指定の設定がある場合は上書き）
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # データフレームの初期化
        self.raw_df = None
        self.processed_df = None
        self.feature_df = None
        self.train_df = None
        self.test_df = None
        
        # 特徴量の重要度
        self.feature_importance = {}
        
    def load_data(self, data_loader=None):
        """
        データの読み込み
        
        Args:
            data_loader: データ読み込み用の関数またはオブジェクト
                         指定がない場合はデフォルトのcombine_race_dataを使用
        
        Returns:
            FeatureGenerator: self（メソッドチェーン用）
        """
        # データローダーがない場合はデフォルトのものを使用
        if data_loader is None:
            from src.data_collection.load_race_data import combine_race_data
            data_loader = combine_race_data
        
        # データの読み込み
        years_range = range(self.config['start_year'], self.config['end_year'])
        self.raw_df = data_loader(years_range)
        logger.info(f"データを読み込みました: {len(self.raw_df)}行")
        
        return self
    
    def preprocess(self):
        """
        基本的な前処理を実行
        
        以下の処理を行います：
        - データ型の変換
        - 日付の処理
        - 無効データの除外
        - ソート
        
        Returns:
            FeatureGenerator: self（メソッドチェーン用）
        """
        if self.raw_df is None:
            raise ValueError("データがロードされていません。load_data()を先に実行してください。")
        
        df = self.raw_df.copy()
        
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
        
        self.processed_df = df
        logger.info(f"前処理が完了しました: {len(df)}行")
        
        return self
    
    def generate_time_features(self):
        """
        時系列関連の特徴量を生成
        
        以下の特徴量を生成します：
        - 前走日
        - レース間隔
        - デビュー戦フラグ
        - レース間隔カテゴリ
        
        Returns:
            FeatureGenerator: self（メソッドチェーン用）
        """
        if self.processed_df is None:
            raise ValueError("前処理が実行されていません。preprocess()を先に実行してください。")
        
        df = self.processed_df.copy()
        
        # 前回出走日の計算
        df["last_race_date"] = df.groupby("馬")["日付"].shift(1)
        
        # レース間隔の計算
        df["days_since_last_race"] = (df["日付"] - df["last_race_date"]).dt.days
        
        # デビュー戦フラグの作成
        df["is_debut"] = df["last_race_date"].isna().astype(int)
        
        # レース間隔のカテゴリ分け（ドメイン知識ベース）
        df["interval_category"] = pd.cut(
            df["days_since_last_race"],
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
        
        self.processed_df = df
        logger.info("時系列特徴量の生成が完了しました")
        
        return self
    
    def generate_track_features(self):
        """
        馬場関連の特徴量を生成
        
        以下の特徴量を生成します：
        - 馬場状態フラグ
        - 道悪適性スコア
        - 道悪経験フラグ
        
        Returns:
            FeatureGenerator: self（メソッドチェーン用）
        """
        if self.processed_df is None:
            raise ValueError("前処理が実行されていません。preprocess()を先に実行してください。")
        
        df = self.processed_df.copy()
        
        # 「重」または「不良」を道悪として判定
        df["is_bad_track"] = df["馬場"].isin(["重", "不良"]).astype(int)
        
        # 時系列順に各馬の道悪と全体の平均着順を計算（データリーケージ防止のため過去データのみ使用）
        df["all_tracks_avg_rank"] = float('nan')  # 全馬場での平均着順
        df["bad_tracks_avg_rank"] = float('nan')  # 道悪での平均着順
        df["bad_tracks_count"] = 0  # 道悪での出走回数
        df["diff_rank_bad_vs_overall"] = float('nan')  # 道悪と全体の平均着順の差
        df["has_bad_track_exp"] = 0  # 道悪経験フラグ
        
        # 馬ごとに処理
        horse_groups = df.groupby("馬")
        for horse, horse_df in horse_groups:
            # 日付順にソート
            horse_df = horse_df.sort_values("日付")
            
            # 各レースについて、そのレース時点での過去の成績から特徴量を計算
            for i, (idx, row) in enumerate(horse_df.iterrows()):
                if i == 0:  # 初出走の場合はスキップ
                    continue
                
                # 現在のレースより前のレースデータを取得
                past_races = horse_df.iloc[:i]
                
                # 全馬場での平均着順
                all_tracks_avg = past_races["着順"].mean()
                df.at[idx, "all_tracks_avg_rank"] = all_tracks_avg
                
                # 道悪での出走回数と平均着順
                bad_track_races = past_races[past_races["is_bad_track"] == 1]
                bad_track_count = len(bad_track_races)
                df.at[idx, "bad_tracks_count"] = bad_track_count
                
                # 道悪経験判定（2回以上を経験ありとする）
                df.at[idx, "has_bad_track_exp"] = 1 if bad_track_count >= 2 else 0
                
                # 道悪での平均着順（出走実績がある場合のみ）
                if bad_track_count > 0:
                    bad_tracks_avg = bad_track_races["着順"].mean()
                    df.at[idx, "bad_tracks_avg_rank"] = bad_tracks_avg
                    
                    # 道悪と全体の平均着順の差（負の値=道悪得意、正の値=道悪苦手）
                    df.at[idx, "diff_rank_bad_vs_overall"] = bad_tracks_avg - all_tracks_avg
        
        # NaN値の処理（平均値で埋める）
        df["all_tracks_avg_rank"] = df["all_tracks_avg_rank"].fillna(df["all_tracks_avg_rank"].mean())
        df["bad_tracks_avg_rank"] = df["bad_tracks_avg_rank"].fillna(df["all_tracks_avg_rank"])
        df["diff_rank_bad_vs_overall"] = df["diff_rank_bad_vs_overall"].fillna(0)  # 差がない場合は0
        
        self.processed_df = df
        logger.info("馬場関連特徴量の生成が完了しました")
        
        return self
    
    def prepare_final_features(self):
        """
        最終的な特徴量を準備
        
        モデルに入力する特徴量の変換・選択を行います。
        
        Returns:
            FeatureGenerator: self（メソッドチェーン用）
        """
        if self.processed_df is None:
            raise ValueError("特徴量が生成されていません。generate_features()を先に実行してください。")
        
        data_df = self.processed_df.copy()
        
        # 特徴量の準備
        features = {
            "log_odds": np.log(data_df["オッズ"].values),
            "weight_scaled": (data_df["斤量"].values - 48) * 2,
            "interval_category": data_df["interval_category"].values,
            "is_debut": data_df["is_debut"].values,
            "diff_rank_bad_vs_overall": data_df["diff_rank_bad_vs_overall"].values,
            "has_bad_track_exp": data_df["has_bad_track_exp"].values,
            "track_condition": data_df["馬場"].astype('category').values,
        }
        
        # 設定に基づいて特徴量をフィルタリング
        if 'features' in self.config:
            feature_mapping = {
                'odds': ['log_odds'],
                'weight': ['weight_scaled'],
                'interval': ['interval_category', 'is_debut'],
                'track_condition': ['track_condition'],
                'track_aptitude': ['diff_rank_bad_vs_overall', 'has_bad_track_exp']
            }
            
            selected_features = {}
            for feature_group in self.config['features']:
                if feature_group in feature_mapping:
                    for feature in feature_mapping[feature_group]:
                        if feature in features:
                            selected_features[feature] = features[feature]
            
            features = selected_features if selected_features else features
        
        # 特徴量データフレームの作成
        self.feature_df = pd.DataFrame(features, index=data_df.index)
        
        # 目的変数の準備
        if self.config['target'] == 'top3':
            self.target = (data_df["着順"] <= 3).astype(int)
        else:
            self.target = (data_df["着順"] == 1).astype(int)  # 1着のみ
        
        logger.info(f"最終特徴量の準備が完了しました: {self.feature_df.shape[1]}個の特徴量")
        
        return self
    
    def split_train_test(self):
        """
        訓練データとテストデータに分割
        
        設定に基づいて、特定の年をテストデータとして分割します。
        
        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: 
                (X_train, y_train, X_test, y_test)
        """
        if self.feature_df is None:
            raise ValueError("最終特徴量が準備されていません。prepare_final_features()を先に実行してください。")
        
        # 日付でフィルタリング
        train_mask = self.processed_df["日付"].dt.year < self.config['test_year']
        test_mask = self.processed_df["日付"].dt.year == self.config['test_year']
        
        # 訓練データとテストデータの準備
        X_train = self.feature_df[train_mask].copy()
        y_train = self.target[train_mask].copy()
        X_test = self.feature_df[test_mask].copy()
        y_test = self.target[test_mask].copy()
        
        logger.info(f"データを分割しました: 訓練データ {len(X_train)}行, テストデータ {len(X_test)}行")
        
        # インスタンス変数に保存
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        return X_train, y_train, X_test, y_test
    
    def process_all(self):
        """
        すべての処理を順番に実行
        
        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: 
                (X_train, y_train, X_test, y_test)
        """
        return (self
                .load_data()
                .preprocess()
                .generate_time_features()
                .generate_track_features()
                .prepare_final_features()
                .split_train_test())
    
    def get_categorical_features(self):
        """
        カテゴリカル特徴量のリストを取得
        
        Returns:
            List[str]: カテゴリカル特徴量のリスト
        """
        categorical_features = []
        
        # カテゴリ型の特徴量を検出
        for col, dtype in self.feature_df.dtypes.items():
            if pd.api.types.is_categorical_dtype(dtype) or col == 'interval_category':
                categorical_features.append(col)
        
        return categorical_features
    
    def get_feature_names(self):
        """
        特徴量名のリストを取得
        
        Returns:
            List[str]: 特徴量名のリスト
        """
        if self.feature_df is not None:
            return self.feature_df.columns.tolist()
        return []
    
    def set_feature_importance(self, importance_dict):
        """
        特徴量の重要度を設定
        
        Args:
            importance_dict (Dict[str, float]): 特徴量名と重要度のマッピング
        """
        self.feature_importance = importance_dict
    
    def generate_documentation(self):
        """
        特徴量のドキュメントを生成
        
        Returns:
            str: マークダウン形式のドキュメント
        """
        if not self.feature_importance:
            return "特徴量の重要度情報がありません。"
        
        doc = f"""
# 特徴量ドキュメント

## 生成日時
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 使用特徴量一覧

### 基本特徴量
- **オッズ関連**
  - `log_odds`: オッズの対数変換値
    - 変換方法: `np.log(data_df["オッズ"].values)`
    - 重要度: {self.feature_importance.get('log_odds', 'N/A')}

- **斤量関連**
  - `weight_scaled`: スケーリングされた斤量
    - 変換方法: `(data_df["斤量"].values - 48) * 2`
    - 重要度: {self.feature_importance.get('weight_scaled', 'N/A')}

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
  - 重要度: {self.feature_importance.get('interval_category', 'N/A')}

- **is_debut**: デビュー戦フラグ
  - 値: `0` または `1`
  - 計算方法: `last_race_date.isna().astype(int)`
  - 重要度: {self.feature_importance.get('is_debut', 'N/A')}

### 馬場適性関連特徴量
- **diff_rank_bad_vs_overall**: 道悪（重・不良）と全体での平均着順の差
  - 値: 数値（負の値=道悪得意、正の値=道悪苦手）
  - 計算方法: `道悪での平均着順 - 全体での平均着順`
  - 重要度: {self.feature_importance.get('diff_rank_bad_vs_overall', 'N/A')}

- **has_bad_track_exp**: 道悪経験フラグ
  - 値: `0`（経験少）または `1`（2回以上の出走経験あり）
  - 計算方法: `道悪での出走回数 >= 2`
  - 重要度: {self.feature_importance.get('has_bad_track_exp', 'N/A')}

- **track_condition**: 現在のレースの馬場状態
  - 値: `良`、`稍`、`重`、`不`
  - データソース: 元データの`馬場`カラム
  - 重要度: {self.feature_importance.get('track_condition', 'N/A')}

## 特徴量の重要度ランキング
"""
        # 特徴量の重要度をソートして追加
        sorted_importance = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for name, importance in sorted_importance:
            doc += f"- {name}: {importance}\n"
 