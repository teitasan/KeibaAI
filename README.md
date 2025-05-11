# KeibaAI_cat (CatBoost専用)

このプロジェクトは競馬予測AIのためのCatBoost専用リポジトリです。

## 特徴
- 機械学習モデルはCatBoostのみを使用
- 依存関係もCatBoostに最適化

## セットアップ
```bash
pip install -r requirements.txt
```

## 使い方
- モデルの学習・推論は`src/prediction/model.py`や`src/prediction/catboost_model.py`を参照してください。
- 評価スクリプトは`src/prediction/evaluate_model.py`です。

## テスト
```bash
pytest
```

## 注意
- LightGBMやscikit-learn等、CatBoost以外のMLライブラリはサポートしていません。
