"""
ファイル名: logger.py
作成日: 2024/03/01
更新日: 2024/03/01
作成者: KeibaAI Team
説明: アプリケーション全体で使用するロギング機能を提供するモジュール
     カラー出力とログファイルへの保存をサポートします
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import colorama
from colorama import Fore, Style


class ColoredFormatter(logging.Formatter):
    """
    クラス名: ColoredFormatter
    説明: ログメッセージに色を付けるためのフォーマッター

    主な機能:
    - ログレベルに応じた色付け
    - カスタムフォーマットの提供
    """

    def format(self, record):
        # ログレベルに応じた色を設定
        if record.levelno == logging.DEBUG:
            color = Fore.BLUE
        elif record.levelno == logging.INFO:
            color = Fore.GREEN
        elif record.levelno == logging.WARNING:
            color = Fore.YELLOW
        elif record.levelno == logging.ERROR:
            color = Fore.RED
        elif record.levelno == logging.CRITICAL:
            color = Fore.RED + Style.BRIGHT
        else:
            color = Fore.WHITE

        # メッセージに色を付ける
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def setup_logger(name="KeibaAI"):
    """
    メソッド名: setup_logger
    説明: ロガーの設定を行います

    引数:
    - name: str - ロガーの名前

    戻り値:
    - logging.Logger - 設定済みのロガー
    """
    # ===========================================
    # ログディレクトリの作成
    # ===========================================
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # ===========================================
    # ログファイルの設定
    # ===========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    # ===========================================
    # ロガーの作成
    # ===========================================
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # ===========================================
    # ハンドラーの設定
    # ===========================================
    # コンソール出力用のハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # ファイル出力用のハンドラー
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # ===========================================
    # ハンドラーの追加
    # ===========================================
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # ===========================================
    # カラーマの初期化
    # ===========================================
    colorama.init()

    return logger


# グローバルロガーの作成
logger = setup_logger()
