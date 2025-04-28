#!/bin/bash

echo "開発ツールのセットアップを開始します..."

# 仮想環境の有効化
source venv/bin/activate

# 開発用依存関係のインストール
echo "開発用依存関係をインストール中..."
pip install -r requirements-dev.txt

# 追加の開発ツールのインストール
echo "追加の開発ツールをインストール中..."
pip install pip-audit  # セキュリティチェック
pip install pytest-cov  # テストカバレッジ
pip install sphinx  # ドキュメント生成
pip install pre-commit  # Gitフック管理

# Gitフックのセットアップ
echo "Gitフックをセットアップ中..."
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/post-commit

# ドキュメントディレクトリの作成
echo "ドキュメントディレクトリを作成中..."
mkdir -p docs

echo "開発ツールのセットアップが完了しました。" 