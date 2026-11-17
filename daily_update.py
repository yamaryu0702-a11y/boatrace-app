"""成績の自動更新(クラウド用)。生データ(data/raw)がなくても動く。

history.csv の最後の日の翌日から昨日までの 番組表(B)・成績(K) を公式サイトから取り、
history / entries / races / kimarite に追記して、form.csv と course_form.csv を作り直す。
使い方: python daily_update.py            (GitHub Actions が毎朝実行)
"""
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

import fetch as fetchmod
from parse import TARGET_VENUES, VENUES, parse_b, parse_k

JST = dt.timezone(dt.timedelta(hours=9))


def main():
    H = pd.read_csv("data/history.csv.gz", dtype=str, keep_default_na=False)
    last = dt.date.fromisoformat(H.date.max())
    today = dt.datetime.now(JST).date()
    day, got = last + dt.timedelta(days=1), []
    hist, ent, rac, kim = [], [], [], []
    while day < today:
        okb, okk = fetchmod.fetch_one("B", day), fetchmod.fetch_one("K", day)
        print(f"取得 {day} B={'○' if okb else '-'} K={'○' if okk else '-'}", flush=True)
        if okb and okk:
            ymd, date = day.strftime("%y%m%d"), day.isoformat()
            be, br = parse_b(Path(f"data/raw/B{ymd}.TXT"), date)
            ke, kr = parse_k(Path(f"data/raw/K{ymd}.TXT"), date)
            for (v, r, b), d in ke.items():
                hist.append({"date": date, "venue": v, "race": r, "boat": b, "regno": d.get("regno", ""),
                             "finish": d.get("finish", ""), "course": d.get("course", ""), "start_time": d.get("start_time", "")})
            for (v, r), d in kr.items():
                kim.append({"date": date, "venue": v, "race": r, "kimarite": d.get("kimarite", "")})
            for (v, r), info in br.items():
                if v in TARGET_VENUES:
                    rac.append({"date": date, "venue": v, "venue_name": VENUES.get(v, ""), "race": r, **info, **kr.get((v, r), {})})
            for (v, r, b), info in be.items():
                if v in TARGET_VENUES:
                    ent.append({"date": date, "venue": v, "race": r, "boat": b, **info, **ke.get((v, r, b), {})})
            got.append(date)
        day += dt.timedelta(days=1)
    if not got:
        print("新しい成績はありません(更新なし)")
        return
    for name, rows, dtypes in (("history", hist, {}), ("entries", ent, {"venue": str, "regno": str, "finish": str}),
                               ("races", rac, {"venue": str}), ("kimarite", kim, {"venue": str})):
        if not rows:
            continue
        path = f"data/{name}.csv.gz"
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        new = pd.DataFrame(rows).astype(str)
        new = new.replace({"nan": "", "None": ""})
        out = pd.concat([old, new], ignore_index=True)
        out = out.drop_duplicates(subset=[c for c in ("date", "venue", "race", "boat") if c in out.columns], keep="last")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{path}: +{len(new)}行")
    env = dict(os.environ, FROM_HISTORY="1")
    for s in ("build_form.py", "build_course.py"):
        print("実行", s, flush=True)
        subprocess.run([sys.executable, "-W", "ignore", s], check=True, capture_output=True, env=env)
    print("更新した日:", ", ".join(got))


if __name__ == "__main__":
    main()
