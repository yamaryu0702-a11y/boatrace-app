"""tide736.net(潮見表・タイドグラフ)の月間データを取得して data/tide736/ に保存する。

20分ごとの潮位(cm)と潮名が入っている。観測地点は東京都の 羽田(3) と 芝浦(2)。
1か月=1リクエスト。間隔を空けて取得する。
使い方: python fetch_tide736.py 2023-11 2026-11
"""
import json
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("data/tide736")
PORTS = {3: "haneda", 2: "shibaura"}


def months(a, b):
    y, m = map(int, a.split("-"))
    y2, m2 = map(int, b.split("-"))
    while (y, m) <= (y2, m2):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = sys.argv[1], sys.argv[2]
    for y, m in months(a, b):
        for hc, name in PORTS.items():
            p = OUT / f"{name}_{y}{m:02d}.json"
            if p.exists():
                continue
            url = f"https://tide736.net/api/get_tide.php?pc=13&hc={hc}&yr={y}&mn={m}&dy=1&rg=month"
            raw = subprocess.run(["curl.exe", "-s", "-m", "60", "-L", url], capture_output=True).stdout
            j = json.loads(raw.decode("utf-8"))
            if j.get("status") != 1:
                print("失敗", name, y, m, j.get("message"))
                continue
            p.write_text(json.dumps(j["tide"]["chart"], ensure_ascii=False), encoding="utf-8")
            print("ok", name, y, m, flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
