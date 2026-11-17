"""全国の成績(K)から選手ごとの出走履歴を作り、直近成績の特徴量を計算する。

出力:
  data/history.csv.gz ... 全場の出走履歴(1行=1艇1走)
  data/form.csv.gz    ... 平和島・多摩川の各出走に対する直近成績(そのレースより前の走だけを使用)
特徴量: 直近10走/20走の 勝率, 3着内率, 平均着順(事故は6着扱い), 直近20走の平均ST
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from parse import parse_k

RAW = Path("data/raw")
OUT = Path("data")
TARGET = {"04", "05"}

import os

# ---- 履歴作成 ----
if os.environ.get("FROM_HISTORY") == "1":
    # 生データ(data/raw)なしで、既存の data/history.csv.gz から特徴量だけを作り直す(クラウドの自動更新用)
    H = pd.read_csv(OUT / "history.csv.gz", dtype=str, keep_default_na=False)
    H["boat"] = H.boat.astype(int)
else:
    rows = []
    for kp in sorted(RAW.glob("K*.TXT")):
        ymd = kp.stem[1:]
        date = f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}"
        res, _ = parse_k(kp, date)
        for (v, r, b), d in res.items():
            rows.append((date, v, r, b, d.get("regno", ""), d.get("finish", ""), d.get("course", ""),
                         d.get("start_time", "")))
    H = pd.DataFrame(rows, columns=["date", "venue", "race", "boat", "regno", "finish", "course", "start_time"])
    H.to_csv(OUT / "history.csv.gz", index=False, encoding="utf-8-sig")
print("履歴", len(H), "行 /", H.date.nunique(), "日 /", H.regno.nunique(), "選手")

# ---- 特徴量 ----
H = H[H.regno != ""].copy()
H["race"] = H.race.astype(int)
H = H.sort_values(["regno", "date", "race"]).reset_index(drop=True)
fin = pd.to_numeric(H.finish, errors="coerce")  # '01'..'06' は数値、F/K0等は NaN
H["valid"] = fin.notna()
H["fin_n"] = fin.fillna(6).clip(upper=6)         # 事故・欠場は6着扱い
H.loc[H.finish == "", "valid"] = False
H["is_win"] = (fin == 1).astype(float)
H["is_top3"] = (fin <= 3).astype(float)
# 結果が空(中止等)の走は履歴から除く
H = H[H.finish != ""].reset_index(drop=True)
H["st"] = pd.to_numeric(H.start_time, errors="coerce")
H.loc[(H.st < -0.2) | (H.st > 0.5), "st"] = np.nan

g = H.groupby("regno", sort=False)
for n in (10, 20):
    for col, nm in (("is_win", "win"), ("is_top3", "top3"), ("fin_n", "avg")):
        # shift(1): そのレースより前の走だけを使う
        H[f"f{n}_{nm}"] = g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=5).mean())
H["f20_st"] = g["st"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
H["n_prior"] = g.cumcount()

F = H[H.venue.isin(TARGET)][["date", "venue", "race", "boat", "regno", "n_prior",
                              "f10_win", "f10_top3", "f10_avg", "f20_win", "f20_top3", "f20_avg", "f20_st"]]
F.to_csv(OUT / "form.csv.gz", index=False, encoding="utf-8-sig")
print("form.csv", len(F), "行 / 履歴5走以上あり:", int(F.f10_win.notna().sum()))
print(F.describe().loc[["mean", "std"]].round(3).to_string())
