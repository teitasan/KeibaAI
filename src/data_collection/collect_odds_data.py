"""
単勝オッズデータを収集するモジュール
"""

import pandas as pd
from typing import List, Dict
import requests
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OddsCollector:
    """単勝オッズデータを収集するクラス"""

    def __init__(self, base_url: str = "https://race.netkeiba.com/odds/"):
        """
        初期化

        Args:
            base_url (str): データ収集先のベースURL
        """
        self.base_url = base_url
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_odds(self, race_id: str) -> Dict:
        """
        指定されたレースの単勝オッズを収集

        Args:
            race_id (str): レースID

        Returns:
            Dict: 馬番をキーとする単勝オッズの辞書
        """
        # TODO: 実際のスクレイピング処理を実装
        # この例では仮のデータを返します
        return {
            1: 2.5,
            2: 4.8,
            3: 8.2,
            # ... 他の馬番も同様に
        }

    def save_odds(self, race_id: str, odds_data: Dict):
        """
        収集したオッズデータを保存

        Args:
            race_id (str): レースID
            odds_data (Dict): 収集したオッズデータ
        """
        df = pd.DataFrame(odds_data.items(), columns=["馬番", "単勝オッズ"])
        output_path = self.data_dir / f"odds_{race_id}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"オッズデータを保存しました: {output_path}")


def main():
    """メイン処理"""
    collector = OddsCollector()

    # テスト用のレースID
    test_race_id = "202401010101"

    try:
        odds_data = collector.collect_odds(test_race_id)
        collector.save_odds(test_race_id, odds_data)
    except Exception as e:
        logger.error(f"データ収集中にエラーが発生しました: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
