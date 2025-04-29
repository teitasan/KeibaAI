"""
ファイル名: collect_race_data.py
作成日: 2024/03/01
更新日: 2024/03/01
作成者: KeibaAI Team
説明: 競馬データを収集するためのスクリプト
     netkeiba.comからレースデータを取得し、CSVファイルとして保存します
"""

import logging
import time
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class RaceDataCollector:
    """レースデータ収集クラス"""

    def __init__(self):
        """初期化"""
        self.base_url = "https://db.netkeiba.com/race/"
        self.data_dir = Path("data/race_results")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_race_data(self, year: int) -> None:
        """
        指定された年のレースデータを収集する

        Args:
            year (int): 収集対象の年
        """
        logger.info(f"{year}年のレースデータ収集を開始します")

        race_results = []
        # 競馬場コード（1: 札幌 〜 10: 小倉）
        for track_id in range(1, 11):
            # 開催回（1〜6）
            for kai in range(1, 7):
                # 開催日（1〜12）
                for day in range(1, 13):
                    race_id_prefix = f"{year:04d}{track_id:02d}{kai:02d}{day:02d}"
                    # レース番号（1〜12）
                    for race_number in range(1, 13):
                        race_id = f"{race_id_prefix}{race_number:02d}"
                        try:
                            race_data = self._scrape_race_data(race_id)
                            if race_data:
                                race_results.extend(race_data)
                                time.sleep(1)  # サーバー負荷軽減のため
                        except Exception as e:
                            logger.error(
                                f"レースID {race_id} の取得中にエラーが発生: {e}"
                            )

        if race_results:
            df = pd.DataFrame(race_results)
            # 出走日を日付型に変換
            df["date"] = pd.to_datetime(df["date"])
            # 馬IDごとに出走履歴を時系列でソート
            df = df.sort_values(["horse_id", "date"])
            # 前回出走日と間隔を計算
            df["last_race_date"] = df.groupby("horse_id")["date"].shift(1)
            df["days_since_last_race"] = (df["date"] - df["last_race_date"]).dt.days
            # デビュー戦フラグを設定
            df["is_debut"] = df["last_race_date"].isna().astype(int)

            output_file = self.data_dir / f"race_results_{year}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"{year}年のデータを保存しました: {len(df)}件")

    def _scrape_race_data(self, race_id: str) -> list:
        """
        個別のレースデータをスクレイピングする

        Args:
            race_id (str): レースID

        Returns:
            list: レース結果のリスト
        """
        url = f"{self.base_url}{race_id}"
        response = requests.get(url)

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        result_table = soup.find("table", class_="race_table_01")

        if not result_table:
            return []

        results = []
        rows = result_table.find_all("tr")[1:]  # ヘッダーを除外

        # レース日を取得
        race_date = soup.find("div", class_="race_head_info").find("p").text.strip()
        race_date = datetime.strptime(race_date, "%Y年%m月%d日")

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 12:
                result = {
                    "着順": self._parse_rank(cols[0].text.strip()),
                    "枠番": int(cols[1].text.strip()),
                    "馬番": int(cols[2].text.strip()),
                    "オッズ": float(cols[12].text.strip()),
                    "レースID": race_id,
                    "date": race_date,
                    "horse_id": self._extract_horse_id(cols[3].find("a")["href"]),
                }
                results.append(result)

        return results

    def _parse_rank(self, rank_str: str) -> int:
        """着順を数値に変換する"""
        try:
            return int(rank_str)
        except ValueError:
            return 99  # 失格、取消等の場合

    def _extract_horse_id(self, horse_url: str) -> str:
        """馬のURLから馬IDを抽出する"""
        match = re.search(r"horse/(\d+)", horse_url)
        return match.group(1) if match else None


def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)
    collector = RaceDataCollector()

    # 2018年から2022年までのデータを収集
    for year in range(2018, 2023):
        collector.collect_race_data(year)


if __name__ == "__main__":
    main()
