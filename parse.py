"""data/raw のB/Kファイルを CSV にまとめる。

出力:
  data/entries.csv.gz ... 1行=1艇(番組表の情報 + 結果)
  data/races.csv.gz   ... 1行=1レース(場・天候・風波・3連単払戻)
"""
import csv
import re
import unicodedata
from pathlib import Path

RAW = Path("data/raw")
TARGET_VENUES = {"04", "05"}  # 平和島・多摩川。空集合にすると全場
OUT = Path("data")


def norm(s):
    return unicodedata.normalize("NFKC", s)


def read_lines(path):
    return [norm(l.rstrip("\r\n")) for l in path.read_bytes().decode("cp932", "replace").split("\n")]


def f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return ""


def parse_b(path, date):
    """番組表 -> {(venue, race, boat): dict}, {(venue, race): dict}"""
    entries, races = {}, {}
    venue = race = None
    for line in read_lines(path):
        m = re.match(r"^(\d\d)BBGN", line)
        if m:
            venue = m.group(1)
            continue
        m = re.match(r"^\s*(\d+)R\s+(\S+).*?H(\d+)m", line)
        if m and venue:
            race = int(m.group(1))
            dl = re.search(r"締切予定\s*(\d+):(\d+)", line)
            races[(venue, race)] = {"title": m.group(2), "distance": m.group(3),
                                    "deadline": f"{int(dl.group(1)):02d}:{dl.group(2)}" if dl else ""}
            continue
        if venue and race and re.match(r"^[1-6] \d{4}", line):
            rest = line[18:].split()
            nums = [f(x) for x in rest[:8]]
            nums += [""] * (8 - len(nums))
            entries[(venue, race, int(line[0]))] = {
                "regno": line[2:6],
                "name": line[6:10].replace(" ", ""),
                "age": line[10:12],
                "branch": line[12:14],
                "weight": line[14:16],
                "class": line[16:18],
                "nat_win": nums[0], "nat_2r": nums[1],
                "loc_win": nums[2], "loc_2r": nums[3],
                "motor_no": int(nums[4]) if nums[4] != "" else "", "motor_2r": nums[5],
                "boat_no": int(nums[6]) if nums[6] != "" else "", "boat_2r": nums[7],
            }
    return entries, races


def parse_k(path, date):
    """競走成績 -> 結果と払戻"""
    results, races = {}, {}
    venue = None
    race = None
    in_pay = False
    for line in read_lines(path):
        m = re.match(r"^(\d\d)KBGN", line)
        if m:
            venue, race, in_pay = m.group(1), None, False
            continue
        if "[払戻金]" in line:
            in_pay = True
            continue
        if venue is None:
            continue
        # 払戻金テーブル: 3連単 / 3連複 / 2連単 / 2連複 (100円あたり)
        if in_pay:
            m = re.match(r"^\s+(\d+)R\s+(\d-\d-\d)\s+(\d+)(?:\s+(\d-\d-\d)\s+(\d+))?(?:\s+(\d-\d)\s+(\d+))?(?:\s+(\d-\d)\s+(\d+))?", line)
            if m:
                d = {"trifecta": m.group(2), "trifecta_pay": m.group(3)}
                if m.group(4):
                    d.update({"box3": m.group(4), "box3_pay": m.group(5)})
                if m.group(6):
                    d.update({"exacta": m.group(6), "exacta_pay": m.group(7)})
                if m.group(8):
                    d.update({"quinella": m.group(8), "quinella_pay": m.group(9)})
                races.setdefault((venue, int(m.group(1))), {}).update(d)
                continue
        # レース見出し
        m = re.match(r"^\s*(\d+)R\s+(\S+).*?H(\d+)m\s+(\S+)\s+風\s+(\S*)\s+(\d+)m\s+波\s+(\d+)cm", line)
        if m:
            in_pay = False
            race = int(m.group(1))
            races.setdefault((venue, race), {}).update({
                "weather": m.group(4), "wind_dir": m.group(5),
                "wind_speed": m.group(6), "wave": m.group(7)})
            continue
        # 列見出し行の末尾が決まり手(逃げ/差し/まくり/まくり差し/抜き/恵まれ)
        if race and "レースタイム" in line:
            mk = re.search(r"レースタイム\s+(\S+)", line)
            if mk:
                races.setdefault((venue, race), {})["kimarite"] = mk.group(1)
            continue
        # 単勝の払戻(レースごとのブロック内)
        if race:
            mw = re.match(r"^\s+単勝\s+(\d)\s+(\d+)", line)
            if mw:
                races.setdefault((venue, race), {}).update({"win_boat": mw.group(1), "win_pay": mw.group(2)})
                continue
        # 着順行
        if race:
            m = re.match(
                r"^\s*(\d\d|[SFLK]\d?)\s+([1-6])\s+(\d{4})\s+(.+?)\s+(\d{1,2})\s+(\d{1,3})\s+(\d\.\d\d)\s+([1-6])\s+(\S+)\s*(\S*)",
                line)
            if m:
                results[(venue, race, int(m.group(2)))] = {
                    "regno": m.group(3),
                    "finish": m.group(1),
                    "exhibit_time": m.group(7),
                    "course": m.group(8),
                    "start_time": m.group(9),
                    "race_time": m.group(10),
                }
                continue
            # 欠場(K0)・失格(L1)など形式が違う行は着順記号だけ記録
            m = re.match(r"^\s*(\d\d|[SFLK]\d?)\s+([1-6])\s+(\d{4})\s+", line)
            if m:
                results[(venue, race, int(m.group(2)))] = {"regno": m.group(3), "finish": m.group(1)}
    return results, races


VENUES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山",
    "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}


def main():
    entry_rows, race_rows = [], []
    for bpath in sorted(RAW.glob("B*.TXT")):
        ymd = bpath.stem[1:]
        date = f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}"
        be, br = parse_b(bpath, date)
        kpath = RAW / f"K{ymd}.TXT"
        ke, kr = parse_k(kpath, date) if kpath.exists() else ({}, {})
        for (v, r), info in br.items():
            if TARGET_VENUES and v not in TARGET_VENUES:
                continue
            row = {"date": date, "venue": v, "venue_name": VENUES.get(v, ""), "race": r}
            row.update(info)
            row.update(kr.get((v, r), {}))
            race_rows.append(row)
        for (v, r, b), info in be.items():
            if TARGET_VENUES and v not in TARGET_VENUES:
                continue
            row = {"date": date, "venue": v, "race": r, "boat": b}
            row.update(info)
            row.update(ke.get((v, r, b), {}))
            entry_rows.append(row)

    OUT.mkdir(exist_ok=True)
    for name, rows in (("entries.csv", entry_rows), ("races.csv", race_rows)):
        if not rows:
            continue
        cols = list(dict.fromkeys(k for row in rows for k in row))
        with open(OUT / name, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(name, len(rows), "行")


if __name__ == "__main__":
    main()
