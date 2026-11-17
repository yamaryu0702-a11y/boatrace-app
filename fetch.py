"""公式サイトから番組表(B)と競走成績(K)をダウンロードして data/raw に展開する。

使い方:
  python fetch.py 2026-10-01 2026-10-31     # 期間指定
  python fetch.py                           # 直近30日
"""
import sys
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
import urllib.error

from lzh import extract

BASE = "https://www1.mbrace.or.jp/od2"
OUT = Path("data/raw")
WAIT = 1.0  # サーバー負荷軽減のための待ち時間(秒)


def fetch_one(kind, day):
    """kind='B' or 'K'。取得できたら True、休止日などで無ければ False。"""
    OUT.mkdir(parents=True, exist_ok=True)
    yymmdd = day.strftime("%y%m%d")
    dest = OUT / f"{kind}{yymmdd}.TXT"
    if dest.exists():
        return True
    url = f"{BASE}/{kind}/{day:%Y%m}/{kind.lower()}{yymmdd}.lzh"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    finally:
        time.sleep(WAIT)
    files = extract(data)
    dest.write_bytes(files[0][1])
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) >= 3:
        start = dt.date.fromisoformat(sys.argv[1])
        end = dt.date.fromisoformat(sys.argv[2])
    else:
        end = dt.date.today()
        start = end - dt.timedelta(days=30)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]

    def one_day(day):
        ok = [fetch_one(k, day) for k in ("B", "K")]
        return f"{day} {'B' if ok[0] else '-'} {'K' if ok[1] else '-'}"

    # サーバー応答が遅いので3並列(各接続は1秒の間隔を空ける)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for line in ex.map(one_day, days):
            print(line, flush=True)


if __name__ == "__main__":
    main()
