"""平和島公式サイト(スマホ版)の「オリジナルデータ」(展示航走の1周・周り足・直線)を取得して保存する。

使い方:
  python original.py                 # 今日の全12R(日付は実行日)
  python original.py --race 5        # 5Rだけ
  python original.py 2026-11-19      # 日付を指定(ページは「その日」の分しか見られないので、必ず当日中に実行)

出力: data/original/original_YYYYMMDD.csv (1行=1艇)
  date, race, boat, regno, name, lap_time, lap_speed, turn_time, turn_speed, straight_time, straight_speed
注意: ページに日付は無く、過去の日は取れない。毎日、レース終了後に実行してためていく。
"""
import csv
import datetime as dt
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

URL = "https://www.heiwajima.gr.jp/asp/kyogi/04/sp/yoso05{rr:02d}.htm"
UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"
WAIT = 1.0
OUT = Path("data/original")
COLS = ["date", "race", "boat", "regno", "name", "lap_time", "lap_speed",
        "turn_time", "turn_speed", "straight_time", "straight_speed"]


def fetch_html(rr):
    req = urllib.request.Request(URL.format(rr=rr), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ""
        raise
    finally:
        time.sleep(WAIT)


def num(s):
    s = re.sub(r"[^0-9.]", "", s or "")
    try:
        return float(s)
    except ValueError:
        return ""


def parse(html):
    """-> list of dict (1艇=1行)。数字が未掲載(展示前)・欠場の艇は含めない。"""
    if "オリジナルデータ" not in html:
        return []
    rows = []
    for tb in re.findall(r"<tbody>(.*?)</tbody>", html, re.S):
        w = re.search(r'class="waku0(\d)"', tb)
        if not w:
            continue
        trs = re.findall(r"<tr[^>]*>(.*?)</tr>", tb, re.S)
        if len(trs) < 2:
            continue
        tds1 = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", trs[0], re.S)]
        tds2 = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", trs[1], re.S)]
        # tds1 = [枠, 選手, 1周, 周り足, 直線] / tds2 = [1周, 周り足, 直線](時速)
        if len(tds1) < 5 or len(tds2) < 3:
            continue
        vals = [num(x) for x in tds1[2:5] + tds2[:3]]
        if any(v == "" for v in vals):
            continue
        regno = re.search(r"toban=(\d{4})", tb)
        name = re.search(r'racer_name"><a[^>]*>(.*?)</a>', tb, re.S)
        rows.append({
            "boat": int(w.group(1)),
            "regno": regno.group(1) if regno else "",
            "name": re.sub(r"\s+", "", name.group(1)) if name else "",
            "lap_time": vals[0], "turn_time": vals[1], "straight_time": vals[2],
            "lap_speed": vals[3], "turn_speed": vals[4], "straight_speed": vals[5],
        })
    return rows


def main():
    args = sys.argv[1:]
    only = None
    if "--race" in args:
        i = args.index("--race")
        only = int(args[i + 1])
        del args[i:i + 2]
    date = args[0] if args else dt.date.today().isoformat()
    ymd = date.replace("-", "")
    path = OUT / f"original_{ymd}.csv"
    # 既存分は残し、取れたレースだけ入れ替える(途中で何度実行しても、取れたものは消えない)
    old = []
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as fh:
            old = list(csv.DictReader(fh))
    got = {}
    for rr in ([only] if only else range(1, 13)):
        boats = parse(fetch_html(rr))
        print(f"{rr}R", len(boats), "艇", flush=True)
        if boats:
            got[rr] = [{"date": date, "race": rr, **b} for b in boats]
    if not got:
        print("データなし(開催なし、または展示前)")
        return
    rows = [r for r in old if int(r["race"]) not in got]
    for rr in got:
        rows += got[rr]
    rows.sort(key=lambda r: (int(r["race"]), int(r["boat"])))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(path, len(rows), "行")


if __name__ == "__main__":
    main()
