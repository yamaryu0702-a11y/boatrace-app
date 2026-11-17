"""tide736.net(潮見表)のデータ版。tide.py と同じ関数(level, trend, daily_range, tide_name)を提供する。

地点は環境変数 TIDE_PORT で選ぶ(haneda=羽田 既定 / shibaura=芝浦)。
"""
import datetime as dt
import glob
import json
import os
from functools import lru_cache

PORT = os.environ.get("TIDE_PORT", "haneda")


@lru_cache(maxsize=4)
def load(port=None):
    port = port or PORT
    out = {}
    for path in sorted(glob.glob(f"data/tide736/{port}_*.json")):
        for day, d in json.load(open(path, encoding="utf-8")).items():
            out[dt.date.fromisoformat(day)] = {
                "cm": [p["cm"] for p in d["tide"]],  # 00:00〜24:00 の20分ごと(73点)
                "name": d["moon"]["title"],
            }
    return out


def level(date, hhmm, port=None):
    t = load(port)
    m = int(hhmm[:2]) * 60 + int(hhmm[3:5])
    if m >= 24 * 60:
        m = 24 * 60 - 1
    i, f = divmod(m, 20)
    a = t[date]["cm"][i]
    b = t[date]["cm"][i + 1]
    return a + (b - a) * f / 20


def trend(date, hhmm, port=None):
    h, m = int(hhmm[:2]), int(hhmm[3:5])
    base = dt.datetime.combine(date, dt.time(h, m))
    a, b = base - dt.timedelta(minutes=30), base + dt.timedelta(minutes=30)
    return level(b.date(), b.strftime("%H:%M"), port) - level(a.date(), a.strftime("%H:%M"), port)


def daily_range(date, port=None):
    v = load(port)[date]["cm"]
    return max(v) - min(v)


def tide_name(date, port=None):
    return load(port)[date]["name"]
