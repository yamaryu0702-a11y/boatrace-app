"""公式サイトの直前情報ページから展示タイム等を取得する。

使い方:
  python beforeinfo.py 2026-11-14 04 05      # その日の平和島・多摩川の全12R
  python beforeinfo.py 2026-11-14 04 --race 3  # 3Rだけ

出力: data/beforeinfo_YYYYMMDD.csv (1行=1艇)
  date, venue, race, boat, regno, weight(kg), exhibit_time, tilt,
  temp, weather, wind_speed, wind_dir_code, water_temp, wave
展示前は exhibit_time が空になる。ページが無い(開催なし)場合は何も出力しない。
"""
import csv
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={hd}"
UA = "Mozilla/5.0"
WAIT = 1.0


def fetch_html(jcd, ymd, rno):
    req = urllib.request.Request(URL.format(rno=rno, jcd=jcd, hd=ymd), headers={"User-Agent": UA})
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
    s = re.sub(r"[^0-9.\-]", "", s or "")
    try:
        return float(s)
    except ValueError:
        return ""


def parse(html):
    """-> (boats[list of dict], weather dict)。開催なし/データなしなら ([], {})"""
    if "展示" not in html or "weather1" not in html:
        return [], {}
    boats = []
    for tb in re.findall(r'<tbody class="is-fs12[^"]*">(.*?)</tbody>', html, re.S):
        m = re.search(r'is-boatColor(\d)[^>]*>\s*(\d)\s*<', tb)
        if not m:
            continue
        regno = re.search(r"toban=(\d{4})", tb)
        # 艇ごとの先頭行のセル: 枠, 写真, 選手名, 体重, 展示タイム, チルト, ...
        first_row = re.search(r"<tr>(.*?)</tr>", tb, re.S).group(1)
        cells = [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", "").strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", first_row, re.S)]
        # cells[3]=体重 cells[4]=展示タイム cells[5]=チルト
        boats.append({
            "boat": int(m.group(2)),
            "regno": regno.group(1) if regno else "",
            "weight": num(cells[3]) if len(cells) > 3 else "",
            "exhibit_time": num(cells[4]) if len(cells) > 4 else "",
            "tilt": num(cells[5]) if len(cells) > 5 else "",
        })
    # スタート展示: 上から順に 1コース〜6コース。各行に艇番とST(Fは出遅れでなくフライングの印)
    se = html.find("スタート展示")
    if se >= 0:
        sec = html[se:]
        by_boat = {b["boat"]: b for b in boats}
        for course, row in enumerate(re.findall(r'<div class="table1_boatImage1">(.*?)</div>', sec, re.S), 1):
            nb = re.search(r"table1_boatImage1Number is-type\d\">\s*(\d)\s*<", row)
            st = re.search(r'boatImage1Time">([^<]*)<', row)
            if nb and int(nb.group(1)) in by_boat:
                raw = st.group(1).strip() if st else ""
                b = by_boat[int(nb.group(1))]
                b["se_course"] = course
                b["se_st"] = num(raw) if raw else ""
                b["se_flying"] = int("F" in raw)
    w = {}
    wb = re.search(r'<div class="weather1_body">(.*)', html, re.S)
    if wb:
        body = wb.group(1)
        units = re.split(r'<div class="weather1_bodyUnit ', body)[1:]
        for u in units:
            kind = u.split('"', 1)[0]
            data = re.search(r'LabelData">(.*?)<', u)
            title = re.search(r'LabelTitle">(.*?)<', u)
            if "is-direction" == kind.strip():
                w["temp"] = num(data.group(1)) if data else ""
            elif "is-weather" in kind:
                w["weather"] = title.group(1) if title else ""
            elif kind.startswith("is-windDirection"):
                d = re.search(r"is-wind(\d+)", u)
                w["wind_dir_code"] = int(d.group(1)) if d else ""
            elif kind.startswith("is-wind"):
                w["wind_speed"] = num(data.group(1)) if data else ""
            elif "is-waterTemperature" in kind:
                w["water_temp"] = num(data.group(1)) if data else ""
            elif "is-wave" in kind:
                w["wave"] = num(data.group(1)) if data else ""
    return boats, w


def main():
    args = sys.argv[1:]
    only = None
    if "--race" in args:
        i = args.index("--race")
        only = int(args[i + 1])
        del args[i:i + 2]
    date, venues = args[0], args[1:]
    ymd = date.replace("-", "")
    rows = []
    for v in venues:
        for rno in ([only] if only else range(1, 13)):
            boats, w = parse(fetch_html(v, ymd, rno))
            for b in boats:
                rows.append({"date": date, "venue": v, "race": rno, **b, **w})
            print(v, f"{rno}R", len(boats), "艇", flush=True)
    if not rows:
        print("データなし(開催なし、または公開期間外)")
        return
    out = Path("data") / f"beforeinfo_{ymd}.csv"
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(out, len(rows), "行")


if __name__ == "__main__":
    main()
