"""締切の直前に、3連単オッズ(120点)を公式サイトから取って保存する(平和島・多摩川)。

使い方:
  python odds_save.py                    # 今日。締切まで WINDOW 分以内のレースだけ取る(GitHub Actionsから10分おきに実行)
  python odds_save.py --all              # 今日の全レースを、いま取る(動作確認用)
  python odds_save.py --out dat/data/odds

出力: <out>/odds_YYYYMMDD.csv  1行=1レースの最新スナップショット(締切前のものだけ。締切に近いほうで上書き)
  date, venue, race, deadline, fetched_at, mins_to_close, 1-2-3, 1-2-4, ... (120列)
"""
import csv
import datetime as dt
import re
import sys
import urllib.request
from itertools import permutations
from pathlib import Path

import fetch as fetchmod
from parse import parse_b

VENUES = ("04", "05")
WINDOW = 12          # 締切まで何分以内のレースを取るか(10分おきに動かして、全レースを1回は拾う)
URL = "https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd}&hd={hd}"
COMBOS = ["-".join(map(str, p)) for p in permutations(range(1, 7), 3)]
JST = dt.timezone(dt.timedelta(hours=9))


def fetch_odds(date8, jcd, rno):
    req = urllib.request.Request(URL.format(rno=rno, jcd=jcd, hd=date8), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        h = r.read().decode("utf-8", "replace")
    m = re.search(r'<tbody class="is-p3-0">(.*?)</tbody>', h, re.S)
    out = {}
    if not m:
        return out
    left, second = [0] * 6, [0] * 6
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S):
        cells = re.findall(r"<td([^>]*)>(.*?)</td>", tr, re.S)
        k = 0
        for g in range(6):
            if left[g] == 0:
                if k >= len(cells):
                    break
                a, v = cells[k]
                k += 1
                rs = re.search(r'rowspan="(\d+)"', a)
                left[g] = int(rs.group(1)) if rs else 1
                second[g] = int(re.sub(r"\D", "", v) or 0)
            if k + 1 >= len(cells):
                break
            b = int(re.sub(r"\D", "", cells[k][1]) or 0)
            vo = cells[k + 1][1]
            k += 2
            left[g] -= 1
            try:
                o = float(vo.strip().replace(",", ""))
            except ValueError:
                continue
            if len({g + 1, second[g], b}) == 3:
                out[f"{g + 1}-{second[g]}-{b}"] = o
    return out


def main():
    args = sys.argv[1:]
    allr = "--all" in args
    outdir = Path(args[args.index("--out") + 1]) if "--out" in args else Path("data/odds")
    now = dt.datetime.now(JST)
    if "--at" in args:      # 動作確認用: --at 2026-11-22T10:20 (その時刻として動かす)
        now = dt.datetime.fromisoformat(args[args.index("--at") + 1]).replace(tzinfo=JST)
    day = now.date()
    if not fetchmod.fetch_one("B", day):
        print("番組表なし(開催なし)")
        return
    _, races = parse_b(Path(f"data/raw/B{day:%y%m%d}.TXT"), day.isoformat())
    path = outdir / f"odds_{day:%Y%m%d}.csv"
    cols = ["date", "venue", "race", "deadline", "fetched_at", "mins_to_close"] + COMBOS
    rows = {}
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                rows[(r["venue"], int(r["race"]))] = r
    n_new = 0
    for (v, rno), info in sorted(races.items()):
        if v not in VENUES or not info.get("deadline"):
            continue
        hh, mm = map(int, info["deadline"].split(":"))
        dl = dt.datetime.combine(day, dt.time(hh, mm), tzinfo=JST)
        mins = (dl - now).total_seconds() / 60
        if mins <= 0 or (not allr and mins > WINDOW):
            continue
        old = rows.get((v, rno))
        if old and float(old["mins_to_close"]) <= mins and not allr:
            continue          # 既に、より締切に近い時点のものがある
        try:
            od = fetch_odds(f"{day:%Y%m%d}", v, rno)
        except Exception as e:
            print(v, rno, "取得失敗", e)
            continue
        if len(od) < 24:          # 欠場があると120点より少ない(5艇=60点、4艇=24点)
            print(v, rno, "オッズ不足", len(od))
            continue
        rows[(v, rno)] = {"date": day.isoformat(), "venue": v, "race": rno, "deadline": info["deadline"],
                          "fetched_at": now.strftime("%H:%M:%S"), "mins_to_close": f"{mins:.1f}", **{c: od.get(c, "") for c in COMBOS}}
        n_new += 1
        print(v, f"{rno}R", f"締切{info['deadline']}の{mins:.0f}分前", len(od), "点", flush=True)
    if not n_new:
        print("対象なし")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: (r["venue"], int(r["race"]))))
    print(path, len(rows), "レース")


if __name__ == "__main__":
    main()
