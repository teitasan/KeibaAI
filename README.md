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

## ハイパーパラメータ自動チューニング（Optuna）

- `src/prediction/evaluate_model.py` には、Optunaを用いたCatBoostのハイパーパラメータ自動チューニング機能が実装されています。
- `RUN_OPTUNA_TUNING = True` に設定してスクリプトを実行すると、Optunaによる最適化が始まります。
- **注意：自動チューニングは試行回数やデータ量によっては数十分～数時間かかる場合があります。時間に余裕があるときに実行するか、他の作業と並行することを推奨します。**
- チューニング結果（最良パラメータ）はコンソールに出力されます。

## テスト
```bash
pytest
```

## 注意
- LightGBMやscikit-learn等、CatBoost以外のMLライブラリはサポートしていません。
