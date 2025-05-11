"""
レースデータの読み込みと結合を行うモジュール
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_race_data(year: int) -> pd.DataFrame:
    """
    指定された年のレースデータを読み込む

    Args:
        year (int): 読み込む年

    Returns:
        pd.DataFrame: レースデータ（障害レースは除外済み）
    """
    data_dir = Path("data/raw")
    file_path = data_dir / f"{year}.csv"

    if not file_path.exists():
        logger.warning(f"データファイルが存在しません: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path, encoding="shift-jis")
        # 障害レースを除外（「芝・ダート」列が「障」のデータ）
        initial_len = len(df)
        df = df[df['芝・ダート'] != '障']
        filtered_len = len(df)
        if initial_len > filtered_len:
            logger.info(f"{year}年の障害レースを{initial_len - filtered_len}件除外しました。")
        logger.info(f"{year}年のレースデータを読み込みました。レコード数: {filtered_len}")
        return df
    except Exception as e:
        logger.error(f"データ読み込み中にエラーが発生しました: {e}")
        return pd.DataFrame()


def combine_race_data(years: range) -> pd.DataFrame:
    """
    複数年のレースデータを結合する

    Args:
        years (range): 結合する年のrange

    Returns:
        pd.DataFrame: 結合されたレースデータ
    """
    dfs = []
    for year in years:
        df = load_race_data(year)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        logger.warning("結合可能なデータがありません")
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"データを結合しました。最終レコード数: {len(combined_df)}")

    return combined_df
