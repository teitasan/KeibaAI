# KeibaAI - 競馬予想ソフト

競馬のレース結果を分析し、予測を行うAIシステムです。

## 機能

- 過去のレースデータの収集
- 馬・騎手の情報収集
- 機械学習による予測
- 予測結果の可視化

## セットアップ手順

### 1. 仮想環境の作成と有効化

```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化（macOS/Linux）
source venv/bin/activate

# 仮想環境の有効化（Windows）
.\venv\Scripts\activate
```

### 2. 依存関係のインストール

```bash
# コア依存関係のインストール
pip install -r requirements.txt

# 開発用依存関係のインストール
pip install -r requirements-dev.txt
```

### 3. 環境変数の設定

```bash
# .envファイルの作成
cp .env.example .env

# .envファイルを編集して必要な環境変数を設定
```

## 開発ガイド

### 依存関係の更新

1. 仮想環境の有効化
```bash
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate   # Windows
```

2. 依存関係の更新
```bash
# 現在の依存関係を更新
pip install --upgrade -r requirements.txt

# 開発用依存関係を更新
pip install --upgrade -r requirements-dev.txt

# 依存関係のバージョンを固定
pip freeze > requirements.txt
```

### コードスタイル

- コードスタイルは`black`を使用して自動フォーマット
- 型チェックは`mypy`を使用
- リンターは`flake8`を使用

```bash
# コードの自動フォーマット
black .

# 型チェック
mypy .

# リンターの実行
flake8
```

## プロジェクト構造

```
├── KeibaAI/
│   ├── CODING_STANDARDS.md
│   ├── README.md
│   ├── data/
│   │   ├── processed/
│   │   ├── raw/
│   ├── docs/
│   ├── pyproject.toml
│   ├── requirements-dev.txt
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── setup_dev_tools.sh
│   ├── src/
│   │   ├── app/
│   │   │   ├── main.py
│   │   ├── data_collection/
│   │   │   ├── collect_race_data.py
│   │   ├── prediction/
│   │   │   ├── model.py
│   │   ├── utils/
│   │   │   ├── logger.py
│   │   │   ├── test_file.py
│   │   │   ├── update_readme.py
│   ├── tests/
│   │   ├── test_utils.py
```

## 使用方法

1. データ収集
```bash
python src/data_collection/collect_race_data.py
```

2. 予測の実行
```bash
python src/prediction/predict.py
```

3. Webインターフェースの起動
```bash
streamlit run src/app/main.py
``` # KeibaAI
