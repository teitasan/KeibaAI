# モデル評価レポート

## 最終更新日時
2025-04-30 22:36:16

## 学習方法
- **サンプル重み**: オッズに基づく重み付け
  - 変換方法: `np.log1p(data_df["オッズ"].values)`
  - 説明: オッズの高い馬（低確率）が正解した場合の価値を重視

## 使用特徴量一覧

### 基本特徴量
- **斤量**: レース時の斤量（数値）
- **interval_category**: レース間隔のカテゴリ分類
- **diff_rank_bad_vs_overall**: 道悪（重・不良）と全体での平均着順の差
- **has_bad_track_exp**: 道悪経験フラグ（カテゴリ変数として0/1を利用）
- **馬場**: 現在のレースの馬場状態（良、稍、重、不良）
- **best_final_3f**: 上がり3ハロン最速値
- **騎手**: 騎手名（カテゴリ変数としてそのまま利用）
- **クラス**: レースのクラス（カテゴリ変数としてそのまま利用）
- **開催**: 開催回次（カテゴリ変数としてそのまま利用）
- **芝・ダート**: 芝かダートかの区分
- **距離**: レース距離
- **性別**: 馬の性別（元データの「性」カラムをリネーム、カテゴリ変数としてそのまま利用。例: 牡, 牝, セ）
- **齢カテゴリ**: 馬の年齢をカテゴリ分け（2以下,3,4,5,6,7以上）したもの。元データの「齢」カラムを利用。
- **last_finish_order**: 前走の着順
- **has_prev_race_data**: 前走データがあれば1、なければ0となる汎用フラグ

### 使用しているカテゴリカル特徴量
- interval_category
- 馬場
- 騎手
- has_bad_track_exp
- クラス
- 開催
- 芝・ダート
- 距離
- 性別
- 齢カテゴリ

## 特徴量の重要度ランキング
- log(オッズ): 1131
- 上がり3ハロン最速値: 799
- 斤量: 336
- interval_category: 243
- 道悪適性差: 208
- 馬場状態: 84
- 道悪経験フラグ: 44

## モデル評価指標

### 全体評価指標
- accuracy: 0.749
- precision: 0.454
- recall: 0.718
- f1_score: 0.556
- best_threshold: 0.250

### レース単位評価指標
- Precision@3: 各レースで上位3頭予測に含まれた実際の3着以内馬の割合（平均）
  - 値: 0.529
- Recall@3: 各レースで実際の3着以内馬のうち、上位3頭予測に含まれた割合（平均）
  - 値: 0.529
- Hit Rate@3: 上位3頭予測に1頭でも3着以内馬が含まれていたレースの割合
  - 値: 0.936
- NDCG@3: 予測順位の質を評価する指標（1.0が最高値）
  - 値: 0.558
  - 説明: 上位3頭の予測順位が実際の3着以内馬の順位にどれだけ近いかを評価
  - 計算方法: DCG（予測順位の利得）をIDCG（理想的な順位の利得）で正規化

## 評価履歴
# モデル評価履歴

## 20250429_184831

### 評価指標
- accuracy: 0.747
- precision: 0.455
- recall: 0.716
- f1_score: 0.556
- best_threshold: 0.250

### 特徴量の重要度
- log(オッズ): 1635
- 斤量: 670
- interval_category: 313
- デビュー戦フラグ: 52

---

## 20250429_190040

### 評価指標
- accuracy: 0.747
- precision: 0.455
- recall: 0.716
- f1_score: 0.556
- best_threshold: 0.250
- precision_at_3: 0.531
- recall_at_3: 0.531
- hit_rate_at_3: 0.937

### 特徴量の重要度
- log(オッズ): 1635
- 斤量: 670
- interval_category: 313
- デビュー戦フラグ: 52

---

## 20250429_190640

### 評価指標
- accuracy: 0.747
- precision: 0.455
- recall: 0.716
- f1_score: 0.556
- best_threshold: 0.250
- precision_at_3: 0.531
- recall_at_3: 0.531
- hit_rate_at_3: 0.937
- ndcg_at_3: 0.560

### 特徴量の重要度
- log(オッズ): 1635
- 斤量: 670
- interval_category: 313
- デビュー戦フラグ: 52

---

## 20250430_183012

### 評価指標
- accuracy: 0.748
- precision: 0.455
- recall: 0.719
- f1_score: 0.557
- best_threshold: 0.250
- precision_at_3: 0.531
- recall_at_3: 0.531
- hit_rate_at_3: 0.936
- ndcg_at_3: 0.560

### 特徴量の重要度
- log(オッズ): 1309
- 道悪適性差: 936
- 斤量: 422
- interval_category: 233
- 道悪経験フラグ: 78
- デビュー戦フラグ: 22

---

## 20250430_185225

### 評価指標
- accuracy: 0.746
- precision: 0.454
- recall: 0.722
- f1_score: 0.557
- best_threshold: 0.250
- precision_at_3: 0.530
- recall_at_3: 0.530
- hit_rate_at_3: 0.936
- ndcg_at_3: 0.559

### 特徴量の重要度
- log(オッズ): 1174
- 道悪適性差: 726
- 斤量: 396
- interval_category: 210
- 馬場状態: 75
- 道悪経験フラグ: 58
- デビュー戦フラグ: 31

---

## 20250430_193659

### 評価指標
- accuracy: 0.746
- precision: 0.454
- recall: 0.722
- f1_score: 0.557
- best_threshold: 0.250
- precision_at_3: 0.530
- recall_at_3: 0.530
- hit_rate_at_3: 0.936
- ndcg_at_3: 0.559

### 特徴量の重要度
- log(オッズ): 1174
- 道悪適性差: 726
- 斤量: 396
- interval_category: 210
- 馬場状態: 75
- 道悪経験フラグ: 58
- デビュー戦フラグ: 31

---

## 20250430_200703

### 評価指標
- accuracy: 0.745
- precision: 0.452
- recall: 0.723
- f1_score: 0.556
- best_threshold: 0.250
- precision_at_3: 0.530
- recall_at_3: 0.530
- hit_rate_at_3: 0.935
- ndcg_at_3: 0.559

### 特徴量の重要度
- log(オッズ): 1381
- 道悪適性差: 546
- 斤量: 529
- interval_category: 263
- 馬場状態: 112
- 道悪経験フラグ: 101
- デビュー戦フラグ: 68

---

## 20250430_202602

### 評価指標
- accuracy: 0.745
- precision: 0.452
- recall: 0.723
- f1_score: 0.556
- best_threshold: 0.250
- precision_at_3: 0.530
- recall_at_3: 0.530
- hit_rate_at_3: 0.935
- ndcg_at_3: 0.559

### 特徴量の重要度
- log(オッズ): 1381
- 道悪適性差: 546
- 斤量: 529
- interval_category: 263
- 馬場状態: 112
- 道悪経験フラグ: 101
- デビュー戦フラグ: 68

---

## 20250430_204536

### 評価指標
- accuracy: 0.746
- precision: 0.453
- recall: 0.720
- f1_score: 0.556
- best_threshold: 0.250
- precision_at_3: 0.531
- recall_at_3: 0.530
- hit_rate_at_3: 0.935
- ndcg_at_3: 0.559

### 特徴量の重要度
- log(オッズ): 1139
- 上がり3ハロン最速値: 771
- 斤量: 358
- 道悪適性差: 266
- interval_category: 179
- 馬場状態: 76
- 道悪経験フラグ: 52
- デビュー戦フラグ: 9

---

## 20250430_210853

### 評価指標
- accuracy: 0.749
- precision: 0.454
- recall: 0.719
- f1_score: 0.557
- best_threshold: 0.250
- precision_at_3: 0.530
- recall_at_3: 0.529
- hit_rate_at_3: 0.934
- ndcg_at_3: 0.558

### 特徴量の重要度
- log(オッズ): 1225
- 上がり3ハロン最速値: 816
- 斤量: 341
- 道悪適性差: 265
- interval_category: 213
- 馬場状態: 85
- 道悪経験フラグ: 49
- デビュー戦フラグ: 6

---

## 20250430_220512

### 評価指標
- accuracy: 0.422
- precision: 0.245
- recall: 0.783
- f1_score: 0.373
- best_threshold: 0.200
- precision_at_3: 0.302
- recall_at_3: 0.302
- hit_rate_at_3: 0.696
- ndcg_at_3: 0.306

### 特徴量の重要度
- 上がり3ハロン最速値: 1058
- 道悪適性差: 583
- 斤量: 578
- interval_category: 372
- 馬場状態: 145
- 道悪経験フラグ: 111
- デビュー戦フラグ: 3

---

## 20250430_222634

### 評価指標
- accuracy: 0.749
- precision: 0.454
- recall: 0.719
- f1_score: 0.557
- best_threshold: 0.250
- precision_at_3: 0.530
- recall_at_3: 0.529
- hit_rate_at_3: 0.934
- ndcg_at_3: 0.558

### 特徴量の重要度
- log(オッズ): 1225
- 上がり3ハロン最速値: 816
- 斤量: 341
- 道悪適性差: 265
- interval_category: 213
- 馬場状態: 85
- 道悪経験フラグ: 49
- デビュー戦フラグ: 6

---

## 20250430_223616

### 評価指標
- accuracy: 0.749
- precision: 0.454
- recall: 0.718
- f1_score: 0.556
- best_threshold: 0.250
- precision_at_3: 0.529
- recall_at_3: 0.529
- hit_rate_at_3: 0.936
- ndcg_at_3: 0.558

### 特徴量の重要度
- log(オッズ): 1131
- 上がり3ハロン最速値: 799
- 斤量: 336
- interval_category: 243
- 道悪適性差: 208
- 馬場状態: 84
- 道悪経験フラグ: 44

---

