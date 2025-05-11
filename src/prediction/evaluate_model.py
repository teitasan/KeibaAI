"""
CatBoost専用モデル評価スクリプト
3着以内に入るかどうかを予測
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from src.data_collection.load_race_data import combine_race_data
from src.features.feature_generator import FeatureGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    # 1. データ読み込み（例：2018-2022年）
    logger.info("レースデータを読み込んでいます...")
    df = combine_race_data(range(2018, 2023))
    if df.empty:
        print("データがありません")
        return
    logger.info(f"全データを結合しました。最終レコード数: {len(df)}")

    # 2. 特徴量生成
    logger.info("特徴量生成を開始します...")
    fg = FeatureGenerator(config={"start_year": 2018, "end_year": 2023})
    fg.raw_df = df
    fg.preprocess()
    logger.info(f"前処理が完了しました: {len(fg.processed_df)}行")
    fg.generate_time_features()
    logger.info("時系列特徴量の生成が完了しました")
    fg.generate_track_features()
    logger.info("馬場適性特徴量の生成が完了しました")
    fg.generate_performance_features()
    logger.info("パフォーマンス特徴量の生成が完了しました")
    # finalize_features_for_modelで全ての特徴量準備・型変換を完結
    X_all, y_all, race_ids_all = fg.finalize_features_for_model()

    # 学習・テスト分割（例：2021年までを学習、2022年をテスト）
    logger.info("学習・テストデータの分割を開始します...")
    train_idx = fg.processed_df["日付"].dt.year <= 2021
    test_idx = fg.processed_df["日付"].dt.year == 2022
    X_train, X_test = X_all[train_idx].copy(), X_all[test_idx].copy()
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    race_ids_train, race_ids_test = None, None
    if race_ids_all is not None:
        race_ids_train, race_ids_test = race_ids_all[train_idx], race_ids_all[test_idx]
    logger.info(f"学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")

    # モデル訓練前にrace_idや馬名を含むX_test_with_infoを保存
    X_test_with_info = X_test.copy()  # drop前の情報を保持
    print('X_test_with_info columns:', X_test_with_info.columns.tolist())
    # --- ここで「馬番」を追加 ---
    if '馬番' not in X_test_with_info.columns:
        X_test_with_info['馬番'] = fg.processed_df.loc[X_test_with_info.index, '馬番'].values
    X_train_model = X_train.drop(columns=["race_id", "馬"], errors="ignore")
    X_test_model = X_test.drop(columns=["race_id", "馬"], errors="ignore")

    # --- TimeSeriesSplit用に学習データを日付順でソート ---
    if '日付' in X_train.columns:
        sort_idx = X_train['日付'].argsort()
        X_train_model = X_train_model.iloc[sort_idx].reset_index(drop=True)
        y_train = y_train.iloc[sort_idx].reset_index(drop=True)

    # 5. CatBoostで学習・評価
    logger.info("CatBoostモデルの学習を開始します...")
    cat_features = [col for col in ['interval_category', '馬場', '騎手', 'has_bad_track_exp', 'クラス', '開催', '芝・ダート', '距離', '性別', '齢カテゴリ'] if col in X_train_model.columns]
    base_model = CatBoostClassifier(
        iterations=100,
        random_seed=42,
        verbose=0,
        eval_metric='F1'
    )
    # CalibratedClassifierCVでキャリブレーション（isotonic, sigmoid両方）
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_model_iso = CalibratedClassifierCV(
        estimator=base_model,
        method='isotonic',
        cv=3
    )
    calibrated_model_sig = CalibratedClassifierCV(
        estimator=base_model,
        method='sigmoid',
        cv=3
    )
    # 学習
    logger.info("CalibratedClassifierCV (isotonic) で学習中...")
    calibrated_model_iso.fit(X_train_model, y_train, cat_features=cat_features)
    logger.info("CalibratedClassifierCV (sigmoid) で学習中...")
    calibrated_model_sig.fit(X_train_model, y_train, cat_features=cat_features)
    # 予測確率
    y_proba_iso = calibrated_model_iso.predict_proba(X_test_model)[:, 1]
    y_proba_sig = calibrated_model_sig.predict_proba(X_test_model)[:, 1]

    # best_thresholdをF1最大化で決定（isotonic, sigmoid両方）
    from sklearn.metrics import precision_recall_curve, average_precision_score
    def get_best_threshold_and_pred(y_true, y_proba):
        f1_scores = []
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        for p, r, t in zip(precisions, recalls, thresholds):
            if p + r > 1e-8:
                f1 = 2 * (p * r) / (p + r)
            else:
                f1 = 0
            f1_scores.append(f1)
        best_idx = int(np.argmax(f1_scores))
        best_threshold = thresholds[best_idx]
        y_pred = (y_proba >= best_threshold).astype(int)
        return best_threshold, y_pred

    best_threshold_iso, y_pred_iso = get_best_threshold_and_pred(y_test, y_proba_iso)
    best_threshold_sig, y_pred_sig = get_best_threshold_and_pred(y_test, y_proba_sig)

    # 全体評価指標
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    def print_metrics(y_true, y_pred, best_threshold, label):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"=== 全体評価指標 ({label}) ===")
        print(f"accuracy: {accuracy:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")
        print(f"f1_score: {f1:.4f}")
        print(f"best_threshold: {best_threshold:.4f}")

    print_metrics(y_test, y_pred_iso, best_threshold_iso, 'isotonic')
    print_metrics(y_test, y_pred_sig, best_threshold_sig, 'sigmoid')

    # レース単位指標
    def print_race_metrics(X_test, y_test, y_proba, race_ids_test, label):
        if race_ids_test is not None:
            race_ids_eval = race_ids_test.values
        elif 'race_id' in X_test.index.names:
            race_ids_eval = X_test.index.get_level_values('race_id')
        elif 'race_id' in X_test.columns:
            race_ids_eval = X_test['race_id']
        else:
            race_ids_eval = pd.Series([0]*len(X_test), index=X_test.index)
        df_eval = X_test.copy()
        df_eval['y_true'] = y_test.values
        df_eval['y_proba'] = y_proba
        df_eval['race_id'] = race_ids_eval
        prec3s, rec3s, hit3s, ndcg3s = [], [], [], []
        for _, group in df_eval.groupby('race_id'):
            if len(group) < 3:
                continue
            top3 = group.sort_values('y_proba', ascending=False).head(3)
            actual_top3_horses = group[group['y_true'] == 1]
            n_actual_top3_in_race = actual_top3_horses['y_true'].sum()
            n_pred_correct_in_top3 = top3['y_true'].sum()
            prec3 = n_pred_correct_in_top3 / 3
            rec3 = n_pred_correct_in_top3 / (n_actual_top3_in_race if n_actual_top3_in_race > 0 else 1)
            hit3 = 1 if n_pred_correct_in_top3 > 0 else 0
            dcg = ((top3['y_true'] / np.log2(np.arange(2, len(top3) + 2))).sum())
            ideal_relevance_scores = np.sort(group['y_true'].values)[::-1][:len(top3)]
            idcg = ((ideal_relevance_scores / np.log2(np.arange(2, len(ideal_relevance_scores) + 2))).sum())
            ndcg = dcg / idcg if idcg > 0 else 0
            prec3s.append(prec3)
            rec3s.append(rec3)
            hit3s.append(hit3)
            ndcg3s.append(ndcg)
        print(f"=== レース単位評価指標 ({label}) ===")
        print(f"precision_at_3: {np.mean(prec3s):.4f}")
        print(f"recall_at_3: {np.mean(rec3s):.4f}")
        print(f"hit_rate_at_3: {np.mean(hit3s):.4f}")
        print(f"ndcg_at_3: {np.mean(ndcg3s):.4f}")

    print_race_metrics(X_test, y_test, y_proba_iso, race_ids_test, 'isotonic')
    print_race_metrics(X_test, y_test, y_proba_sig, race_ids_test, 'sigmoid')

    # キャリブレーションカーブの比較
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    prob_true_iso, prob_pred_iso = calibration_curve(y_test, y_proba_iso, n_bins=10)
    prob_true_sig, prob_pred_sig = calibration_curve(y_test, y_proba_sig, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred_iso, prob_true_iso, marker='o', label='isotonic')
    plt.plot(prob_pred_sig, prob_true_sig, marker='o', label='sigmoid')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='理想的なキャリブレーション')
    plt.xlabel('予測確率', fontsize=14)
    plt.ylabel('実際の確率', fontsize=14)
    plt.title('キャリブレーションカーブ比較', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('calibration_curve_compare.png')
    plt.close()
    print('キャリブレーションカーブ比較をcalibration_curve_compare.pngとして保存しました')

    # --- 追加: キャリブレーション前のCatBoostモデルで特徴量重要度を出力 ---
    logger.info("キャリブレーション前のCatBoostモデルを訓練データ全体でfitします...")
    base_model.fit(X_train_model, y_train, cat_features=cat_features)
    importances = base_model.get_feature_importance()
    feature_names = base_model.feature_names_
    print("=== キャリブレーション前モデルの特徴量重要度 ===")
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    for index, row in feature_importance_df.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")

    # === ここからグラフ出力 ===
    import matplotlib
    import matplotlib.font_manager as fm
    import platform
    from pathlib import Path as _Path
    font_candidates = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴ ProN W6.ttc",
        "/Library/Fonts/IPAexGothic.ttf",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
    ]
    font_path = None
    for path in font_candidates:
        if _Path(path).exists():
            font_path = path
            break
    if font_path:
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        print(f'日本語フォントを適用: {font_prop.get_name()}')
    else:
        print('日本語フォントが見つかりませんでした。')

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='skyblue')
    plt.xlabel('重要度', fontsize=14)
    plt.ylabel('特徴量', fontsize=14)
    plt.title('特徴量の重要度', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print('特徴量重要度グラフをfeature_importance.pngとして保存しました')

    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, y_proba_iso, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='isotonic')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='理想的なキャリブレーション')
    plt.xlabel('予測確率', fontsize=14)
    plt.ylabel('実際の確率', fontsize=14)
    plt.title('キャリブレーションカーブ', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('calibration_curve_isotonic.png')
    plt.close()
    print('isotonicキャリブレーションカーブをcalibration_curve_isotonic.pngとして保存しました')

    prob_true, prob_pred = calibration_curve(y_test, y_proba_sig, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='sigmoid')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='理想的なキャリブレーション')
    plt.xlabel('予測確率', fontsize=14)
    plt.ylabel('実際の確率', fontsize=14)
    plt.title('キャリブレーションカーブ', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('calibration_curve_sigmoid.png')
    plt.close()
    print('sigmoidキャリブレーションカーブをcalibration_curve_sigmoid.pngとして保存しました')

    # --- 手動で最良パラメータを指定して再学習・評価のみ実行する分岐 ---
    RUN_ONLY_BEST = True  # Trueならチューニングをスキップし、下記パラメータでのみ再学習・評価
    manual_best_params = {'learning_rate': 0.1817321323824161, 'depth': 10, 'l2_leaf_reg': 8, 'iterations': 688}

    if RUN_ONLY_BEST:
        print('手動指定の最良パラメータでモデルを再学習し、テストデータで評価します...')
        best_params = manual_best_params.copy()
        best_params.update({'random_seed': 42, 'eval_metric': 'F1', 'verbose': 0})
        best_model = CatBoostClassifier(**best_params)
        best_model.fit(X_train_model, y_train, cat_features=cat_features)
        y_pred_best = best_model.predict(X_test_model)
        y_proba_best = best_model.predict_proba(X_test_model)[:, 1]
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        print('=== 手動最良パラメータモデルのテスト評価 ===')
        print(f'accuracy: {accuracy_score(y_test, y_pred_best):.4f}')
        print(f'precision: {precision_score(y_test, y_pred_best):.4f}')
        print(f'recall: {recall_score(y_test, y_pred_best):.4f}')
        print(f'f1_score: {f1_score(y_test, y_pred_best):.4f}')
        print_race_metrics(X_test, y_test, y_proba_best, race_ids_test, 'Manual_best')

        # --- F1最大化の閾値自動探索 ---
        def get_best_threshold_and_pred(y_true, y_proba):
            f1_scores = []
            from sklearn.metrics import precision_recall_curve
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
            for p, r, t in zip(precisions, recalls, thresholds):
                if p + r > 1e-8:
                    f1 = 2 * (p * r) / (p + r)
                else:
                    f1 = 0
                f1_scores.append(f1)
            best_idx = int(np.argmax(f1_scores))
            best_threshold = thresholds[best_idx]
            y_pred = (y_proba >= best_threshold).astype(int)
            return best_threshold, y_pred

        best_threshold, y_pred_best_f1 = get_best_threshold_and_pred(y_test, y_proba_best)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        print('--- F1最大化閾値での評価 ---')
        print(f'best_threshold: {best_threshold:.4f}')
        print(f'accuracy: {accuracy_score(y_test, y_pred_best_f1):.4f}')
        print(f'precision: {precision_score(y_test, y_pred_best_f1):.4f}')
        print(f'recall: {recall_score(y_test, y_pred_best_f1):.4f}')
        print(f'f1_score: {f1_score(y_test, y_pred_best_f1):.4f}')
        print_race_metrics(X_test, y_test, y_proba_best, race_ids_test, f'Manual_best@F1th={best_threshold:.3f}')

        # --- X_test_with_infoに馬名をrace_idと馬番でjoin ---
        # if '馬' not in X_test_with_info.columns:
        #     key_cols = ['race_id', '馬番']
        #     horse_map = fg.processed_df[key_cols + ['馬']].drop_duplicates().set_index(key_cols)['馬']
        #     X_test_with_info['馬'] = X_test_with_info.set_index(key_cols).index.map(horse_map)

        # --- 予測確率が高い順に上位10頭を表示（X_test_with_infoを使う） ---
        # print('\n--- 予測確率が高い上位10頭（参考値） ---')
        # top10 = X_test_with_info.copy()
        # top10['y_true'] = y_test.values
        # top10['y_proba'] = y_proba_best
        # if '馬' not in top10.columns:
        #     top10['馬'] = '(不明)'
        # display_cols = ['race_id', '馬', 'y_proba', 'y_true']
        # top10 = top10.sort_values('y_proba', ascending=False).head(10)
        # print(top10[display_cols].to_string(index=False))

        return

if __name__ == "__main__":
    main()