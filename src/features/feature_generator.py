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

print("LOADED FeatureGenerator from src/features/feature_generator.py")

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
    
    # src/features/feature_generator.py 内の generate_time_features(self) メソッド

    def generate_time_features(self):
        if self.processed_df is None:
            raise ValueError("前処理が実行されていません。preprocess()を先に実行してください。")

        # --- DEBUGGING: メソッド開始時の確認 ---
        print("DEBUG: generate_time_features メソッド開始。")
        print(f"DEBUG: generate_time_features 開始時のカラム: {self.processed_df.columns.tolist()}")
        print(f"DEBUG: '馬名'カラムが存在するか: {'馬名' in self.processed_df.columns}")
        print(f"DEBUG: '着順'カラムが存在するか: {'着順' in self.processed_df.columns}")
        print(f"DEBUG: '着順'カラムの dtype: {self.processed_df['着順'].dtype}")
        print(f"DEBUG: '着順'カラムのユニーク値 (最初の10個): {self.processed_df['着順'].value_counts(dropna=False).head(10)}")

        # 馬・日付・race_idでソート（preprocessでソートされているはずですが、ここでも明示的に行うのは安全策）
        self.processed_df = self.processed_df.sort_values(by=['馬', '日付', 'race_id'], ascending=True).copy()
        print("DEBUG: generate_time_features 内でデータソート完了。")

        # --- 'last_finish_order' と 'has_prev_race_data' の生成 ---
        print("DEBUG: 前走着順/上がりタイム計算の直前。") 

        # shifted_finish_order_series を計算し、reindexでインデックスを保証
        shifted_finish_order_series = self.processed_df.groupby("馬")["着順"].shift(1).reindex(self.processed_df.index)
        
        # --- DEBUGGING: shifted_finish_order_series の中身を詳しく確認 ---
        print(f"DEBUG: shifted_finish_order_series のshape: {shifted_finish_order_series.shape}")
        print(f"DEBUG: shifted_finish_order_series のdtype: {shifted_finish_order_series.dtype}")
        print(f"DEBUG: shifted_finish_order_series のhead():\n{shifted_finish_order_series.head()}")
        print(f"DEBUG: shifted_finish_order_series のvalue_counts(dropna=False):\n{shifted_finish_order_series.value_counts(dropna=False)}")
        print(f"DEBUG: shifted_finish_order_series にNaNが含まれるか: {shifted_finish_order_series.isna().any()}")

        # self.processed_df への代入
        self.processed_df["has_prev_race_data"] = shifted_finish_order_series.notna().astype(int)
        self.processed_df["last_finish_order"] = shifted_finish_order_series

        # --- DEBUGGING: 追加後のカラム確認 ---
        print("DEBUG: last_finish_order と has_prev_race_data が追加されました。")
        print(f"DEBUG: generate_time_features 中間時点のカラム: {self.processed_df.columns.tolist()}")

        # --- 'last_agari_time' と 'no_last_agari_time' も同様に修正 ---
        if '上がり' in self.processed_df.columns:
            # '上がり'カラムが数値型であることを確認（必要に応じて変換）
            if not pd.api.types.is_numeric_dtype(self.processed_df['上がり']):
                print(f"DEBUG: '上がり'カラムのdtypeが数値ではありません: {self.processed_df['上がり'].dtype}. 変換を試みます。")
                self.processed_df.loc[:, '上がり'] = pd.to_numeric(self.processed_df['上がり'], errors='coerce')
                # NaNを埋めるのはここではしない。
            shifted_agari_time_series = self.processed_df.groupby("馬")["上がり"].shift(1).reindex(self.processed_df.index)
            self.processed_df["no_last_agari_time"] = shifted_agari_time_series.notna().astype(int)
            self.processed_df["last_agari_time"] = shifted_agari_time_series
            print("DEBUG: last_agari_time と no_last_agari_time が追加されました。")
        else:
            print("DEBUG: '上がり'カラムが見つかりません。last_agari_timeは生成されません。")


        # --- 既存の時系列特徴量生成コード ---
        # 'last_race_date' の計算
        temp_last_race_date = self.processed_df.groupby("馬")["日付"].shift(1).reindex(self.processed_df.index)
        self.processed_df["last_race_date"] = temp_last_race_date
        
        # 'days_since_last_race' の計算
        # ★ここがエラーの原因だったはず：df["日付"] を self.processed_df["日付"] に修正 ★
        self.processed_df["days_since_last_race"] = (self.processed_df["日付"] - self.processed_df["last_race_date"]).dt.days
        
        # レース間隔のカテゴリ分け
        self.processed_df["interval_category"] = pd.cut(
            self.processed_df["days_since_last_race"],
            bins=[-1, 0, 8, 15, 35, 91, 182, 365, float("inf")],
            labels=["デビュー", "連闘", "中1週", "中2-4週", "1-3ヶ月", "3-6ヶ月", "6ヶ月-1年", "1年以上",],
            right=False,
        ).astype('category') # pd.cutの結果はCategoricalDtypeになるので、明示的なastype('category')は不要かも。

        # デビュー戦フラグの作成
        self.processed_df["is_debut"] = (self.processed_df["interval_category"] == "デビュー").astype(int)
        
        # 前走クラス
        if 'クラス' in self.processed_df.columns:
            self.processed_df['last_class'] = self.processed_df.groupby('馬')['クラス'].shift(1)
        
        # 出走頭数
        # race_idの型・桁数を明示的に統一（int型13桁に揃える）
        if self.processed_df['race_id'].dtype != 'int64':
            # まずstr型→int型へ変換（13桁保証）
            self.processed_df['race_id'] = self.processed_df['race_id'].astype(str).str.zfill(13).astype('int64')
        # 出走頭数をrace_idごとにカウント
        self.processed_df['出走頭数'] = self.processed_df.groupby('race_id')['馬'].transform('size')
        
        print("DEBUG: 時系列特徴量の生成が完了しましたメソッド終了直前。")
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
        # is_bad_trackをself.processed_dfに直接追加
        self.processed_df["is_bad_track"] = self.processed_df["馬場"].isin(["重", "不良"]).astype(int)
        # 馬ごとに日付でソート
        self.processed_df = self.processed_df.sort_values(["馬", "日付"])
        # 全馬場の平均着順
        self.processed_df["all_tracks_avg_rank"] = (
            self.processed_df.groupby("馬")["着順"]
            .expanding()
            .mean()
            .groupby(level=0)
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        # 道悪馬場での累積出走回数
        self.processed_df["bad_tracks_count"] = (
            self.processed_df.groupby("馬")["is_bad_track"]
            .expanding()
            .sum()
            .groupby(level=0)
            .shift(1)
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        # 道悪経験フラグ (2回以上の道悪経験がある場合に1)
        self.processed_df["has_bad_track_exp"] = (self.processed_df["bad_tracks_count"] >= 2).astype(int)
        # 道悪での平均着順 (過去の道悪レースのみでexpanding計算し、全レース行に持たせる)
        self.processed_df["bad_tracks_avg_rank"] = (
            self.processed_df[self.processed_df["is_bad_track"] == 1]
            .groupby("馬")["着順"]
            .expanding()
            .mean()
            .groupby(level=0)
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        # 全レース行にmapで割り当てる（馬・日付で）
        self.processed_df["bad_tracks_avg_rank"] = self.processed_df.set_index(["馬", "日付"]).index.map(
            self.processed_df[self.processed_df["is_bad_track"] == 1].set_index(["馬", "日付"])["bad_tracks_avg_rank"]
        )
        # 道悪と全体の平均着順の差
        self.processed_df["diff_rank_bad_vs_overall"] = self.processed_df["bad_tracks_avg_rank"] - self.processed_df["all_tracks_avg_rank"]
        # NANの処理
        self.processed_df["all_tracks_avg_rank"] = self.processed_df["all_tracks_avg_rank"].fillna(self.processed_df["all_tracks_avg_rank"].mean())
        self.processed_df["bad_tracks_avg_rank"] = self.processed_df["bad_tracks_avg_rank"].fillna(self.processed_df["all_tracks_avg_rank"])
        self.processed_df["diff_rank_bad_vs_overall"] = self.processed_df["diff_rank_bad_vs_overall"].fillna(0)
        logger.info("馬場適性特徴量の生成が完了しました")
        print("[DEBUG] after generate_track_features:", self.processed_df.columns.tolist())
        return self
    
    def generate_performance_features(self):
        """
        パフォーマンス関連の特徴量を生成
        
        以下の特徴量を生成します：
        - 上がり3ハロン最速値
        
        Returns:
            FeatureGenerator: self（メソッドチェーン用）
        """
        if self.processed_df is None:
            raise ValueError("前処理が実行されていません。preprocess()を先に実行してください。")
        
        df = self.processed_df.copy()
        
        # 「上がり」を数値型に変換（無効値はNaNに）
        self.processed_df["上がり"] = pd.to_numeric(df["上がり"], errors="coerce")
        
        # 各馬の過去レースにおける上がり3ハロンの最速値（最小値）を計算
        self.processed_df["best_final_3f"] = (
            df.groupby("馬")["上がり"]
            .expanding()
            .min()
            .groupby(level=0)
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        # best_final_3fはNaNのままにする（データがない馬はNaN値を維持）
        
        # 統計情報を出力
        valid_best_final = self.processed_df["best_final_3f"].dropna()
        logger.info("\n=== 上がり3ハロン最速値の統計 ===")
        logger.info(f"データあり: {len(valid_best_final)}頭 ({len(valid_best_final) / len(df) * 100:.1f}%)")
        
        if len(valid_best_final) > 0:
            logger.info(f"最速値: {valid_best_final.min():.1f}秒")
            logger.info(f"平均値: {valid_best_final.mean():.1f}秒")
            logger.info(f"最遅値: {valid_best_final.max():.1f}秒")
        
        logger.info("パフォーマンス特徴量の生成が完了しました")
        print("[DEBUG] after generate_performance_features:", self.processed_df.columns.tolist())
        return self
    
    def prepare_final_features(self):
        """
        最終的な特徴量を準備
        
        Returns:
            FeatureGenerator: self（メソッドチェーン用）
        """
        if self.processed_df is None:
            raise ValueError("特徴量が生成されていません。generate_features()を先に実行してください。")
        data_df = self.processed_df.copy()
        # evaluate_model.pyのuse_colsと一致させる
        final_feature_names_for_model = [
            '斤量',
            'interval_category',
            'diff_rank_bad_vs_overall',
            'has_bad_track_exp',
            '馬場',
            'best_final_3f',
            '騎手',
            'クラス',
            '開催',
            '芝・ダート',
            '距離',
            '性別',
            '齢カテゴリ',
            'last_finish_order',
            'has_prev_race_data',
            'last_class',
            '出走頭数',
        ]
        # 存在しないカラムは自動的に除外
        available_features = [col for col in final_feature_names_for_model if col in data_df.columns]
        self.feature_df = data_df[available_features].copy()
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
                .generate_performance_features()
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
        
        return doc

    def finalize_features_for_model(self, use_cols=None):
        """
        モデル投入直前の特徴量選択・型変換・追加処理をまとめて行う
        use_cols: 最終的にモデルに渡す特徴量名リスト（Noneならデフォルト）
        戻り値: X（特徴量DataFrame）, y（目的変数Series）、race_id（Series, 存在すれば）
        """
        df = self.processed_df.copy()
        # 性カラムを「性別」として追加
        if '性' in df.columns:
            df['性別'] = df['性']
        # 齢カラムをカテゴリ分けして「齢カテゴリ」として追加
        if '齢' in df.columns:
            bins = [-float('inf'), 2, 3, 4, 5, 6, float('inf')]
            labels = ['2以下', '3', '4', '5', '6', '7以上']
            df['齢カテゴリ'] = pd.cut(df['齢'], bins=bins, labels=labels, right=True)
            df['齢カテゴリ'] = df['齢カテゴリ'].astype('category')
        # interval_category, 馬場, 騎手, has_bad_track_exp, クラス, 開催, 芝・ダート, 距離, 性別, 齢カテゴリ, last_class をカテゴリ型に
        for col in ['interval_category', '馬場', '騎手', 'has_bad_track_exp', 'クラス', '開催', '芝・ダート', '距離', '性別', '齢カテゴリ', 'last_class']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('NaN').astype('category')
        # 最終的な特徴量リスト
        if use_cols is None:
            use_cols = [
                '斤量',
                'interval_category',
                'diff_rank_bad_vs_overall',
                'has_bad_track_exp',
                '馬場',
                'best_final_3f',
                '騎手',
                'クラス',
                '開催',
                '芝・ダート',
                '距離',
                '性別',
                '齢カテゴリ',
                'last_finish_order',
                'has_prev_race_data',
                'last_class',
                '出走頭数',
            ]
        # race_idもあれば残す
        use_cols_with_race_id = use_cols + [col for col in ['race_id'] if col in df.columns]
        X = df[use_cols_with_race_id].copy()
        # 目的変数
        if self.config['target'] == 'top3':
            y = (df['着順'] <= 3).astype(int)
        else:
            y = (df['着順'] == 1).astype(int)
        race_id = df['race_id'] if 'race_id' in df.columns else None
        return X, y, race_id