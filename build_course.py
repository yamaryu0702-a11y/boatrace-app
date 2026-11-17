"""選手のコース別成績(そのレースより前の走だけ)を作る。

出力: data/course_form.csv.gz (平和島・多摩川の各出走 x 「枠なり(艇番=コース)」と仮定したときのコース別成績)
  c_*_all  : 全国での、その艇番のコースの成績(平滑化済み)
  c_*_here : その場での、その艇番のコースの成績(全国成績に向けて平滑化)
  *_n      : 元になった出走数
"""
import numpy as np
import pandas as pd

import os
ACTUAL = os.environ.get("ACTUAL") == "1"
OUTFILE = "data/course_form_actual.csv" if ACTUAL else "data/course_form.csv.gz"
TARGET = ["04", "05"]
K_ALL = 15   # 全国平均に向けた平滑化の強さ(擬似出走数)
K_HERE = 10  # 全国成績に向けた平滑化の強さ

H = pd.read_csv("data/history.csv.gz", dtype={"venue": str, "regno": str, "finish": str, "course": str})
H = H[(H.regno.notna()) & (H.finish.notna())].copy()
H["race"] = H.race.astype(int)
H["course_n"] = pd.to_numeric(H.course, errors="coerce")
fin = pd.to_numeric(H.finish, errors="coerce")
H["w"] = (fin == 1).astype(int)
H["t2"] = (fin <= 2).astype(int)
H["t3"] = (fin <= 3).astype(int)
H = H.sort_values(["regno", "date", "race"]).reset_index(drop=True)
print("履歴", len(H), "行、コース情報あり", int(H.course_n.notna().sum()))

# 全国のコース別の平均(平滑化の事前分布)
valid = H[H.course_n.between(1, 6)]
prior = valid.groupby("course_n")[["w", "t2", "t3"]].mean()
print("全国コース別 勝率/2連対/3連対:\n", (prior * 100).round(1).to_string())

STATS = ["n", "w", "t2", "t3"]
prior_arr = np.zeros((7, 3))  # 添字=コース(1..6)
for c in range(1, 7):
    prior_arr[c] = prior.loc[float(c), ["w", "t2", "t3"]].to_numpy()


def prior_counts(mask_extra=None):
    """各行について、その行より前の (選手, コースc) の累積 n,w,t2,t3 を返す: dict[c] -> DataFrame"""
    out = {}
    for c in range(1, 7):
        ind = (H.course_n == c)
        if mask_extra is not None:
            ind = ind & mask_extra
        ind = ind.astype(int)
        d = pd.DataFrame({"n": ind, "w": ind * H.w, "t2": ind * H.t2, "t3": ind * H.t3})
        d["regno"] = H.regno
        cs = d.groupby("regno")[STATS].cumsum()
        out[c] = (cs - d[STATS]).to_numpy()  # 自分の行を含めない
    return out


def pick(counts, boat):
    """行ごとに boat(=コース)に対応する累積値を取り出す -> (N,4)"""
    res = np.zeros((len(boat), 4))
    for c in range(1, 7):
        sel = boat == c
        res[sel] = counts[c][sel]
    return res


allc = prior_counts()
rows = []
for v in TARGET:
    sel = (H.venue == v).to_numpy()
    here = prior_counts(mask_extra=(H.venue == v))
    boat = H.boat.to_numpy()
    if ACTUAL:  # 上限見積もり用: 実際の進入コースで引く(事前には分からない情報)
        boat = H.course_n.fillna(H.boat).astype(int).to_numpy()
    A = pick(allc, boat)
    Hh = pick(here, boat)
    pri = prior_arr[boat]  # boat(1..6) で一括参照
    out = pd.DataFrame({"date": H.date, "venue": H.venue, "race": H.race, "boat": H.boat,
                        "regno": H.regno})
    out["c_n_all"] = A[:, 0]
    out["c_n_here"] = Hh[:, 0]
    for j, nm in enumerate(["win", "top2", "top3"]):
        sm_all = (A[:, 1 + j] + K_ALL * pri[:, j]) / (A[:, 0] + K_ALL)
        sm_here = (Hh[:, 1 + j] + K_HERE * sm_all) / (Hh[:, 0] + K_HERE)
        out[f"c_{nm}_all"] = sm_all
        out[f"c_{nm}_here"] = sm_here
    rows.append(out[sel])

C = pd.concat(rows)
C.to_csv(OUTFILE, index=False, encoding="utf-8-sig")
print(OUTFILE, len(C), "行")
print(C[["c_n_all", "c_n_here", "c_win_all", "c_win_here", "c_top3_all", "c_top3_here"]].describe().loc[["mean", "50%", "max"]].round(3).to_string())
