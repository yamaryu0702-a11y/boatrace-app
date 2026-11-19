"""共通の特徴量作成(指数v3と同じ)。build_dataset() で艇単位のデータを返す。"""
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import tide736 as t7

WARMUP = "2024-01-15"
K = ["date", "venue", "race", "boat"]
FEATS = ["nat_win_d", "loc_win_d", "motor_2r_d", "boat_2r_d", "cls_d", "weight_d", "exh_d",
         "f20_st_d", "f10_avg_d", "c_win_all_d", "rt_d"]   # rt_d: 自作レーティング(ratings.py)のレース内の差
LANES = [f"lane{i}" for i in range(1, 7)]
TIDE = ["easy_1", "hard_1"]


def tide_flags(date, deadline):
    """潮ルール(潮見表=羽田): (易, 難)"""
    try:
        d = dt.date.fromisoformat(date)
        lv, nm = t7.level(d, deadline, "haneda"), t7.tide_name(d, "haneda")
    except Exception:
        return 0, 0
    easy = ((nm == "大潮" and lv >= 175) or (nm == "中潮" and lv >= 132) or (nm == "小潮" and lv <= 110)
            or (nm == "長潮" and (lv >= 139 or lv <= 124)))
    hard = ((nm == "大潮" and lv <= 110) or (nm == "中潮" and lv <= 55) or (nm == "小潮" and lv >= 135)
            or (nm == "長潮" and 124 < lv < 139))
    return int(easy), int(hard)


def build_dataset():
    E = pd.read_csv("data/entries.csv.gz", dtype={"venue": str, "regno": str, "finish": str})
    F = pd.read_csv("data/form.csv.gz", dtype={"venue": str, "regno": str}).drop(columns=["regno"])
    C = pd.read_csv("data/course_form.csv.gz", dtype={"venue": str, "regno": str}).drop(columns=["regno"])
    R = pd.read_csv("data/races.csv.gz", dtype={"venue": str})[["date", "venue", "race", "deadline"]]
    KM = pd.read_csv("data/kimarite.csv.gz", dtype={"venue": str}).fillna("")
    E = E[E.finish.isin(["01", "02", "03", "04", "05", "06"]) & (E.date >= WARMUP)].copy()
    E = E.merge(F, on=K, how="left").merge(C, on=K, how="left")
    E = E.merge(R, on=["date", "venue", "race"], how="left").merge(KM, on=["date", "venue", "race"], how="left")
    E["win"] = (E.finish == "01").astype(int)
    E["top3"] = E.finish.isin(["01", "02", "03"]).astype(int)
    E["key"] = E.date + "_" + E.venue + "_" + E.race.astype(str)
    E = E[E.groupby("key").boat.transform("count") == 6]

    rk = E[E.venue == "04"].drop_duplicates("key")[["key", "date", "deadline"]]
    fl = [tide_flags(d, t) if isinstance(t, str) else (0, 0) for d, t in zip(rk.date, rk.deadline)]
    rk["easy"], rk["hard"] = zip(*fl)
    E = E.merge(rk[["key", "easy", "hard"]], on="key", how="left")
    E[["easy", "hard"]] = E[["easy", "hard"]].fillna(0)
    E["easy_1"] = E.easy * (E.boat == 1)
    E["hard_1"] = E.hard * (E.boat == 1)

    g = E.groupby("key")
    E["cls"] = E["class"].map({"A1": 3, "A2": 2, "B1": 1, "B2": 0})
    for c in ["nat_win", "loc_win", "motor_2r", "boat_2r", "cls", "weight", "c_win_all"]:
        E[c + "_d"] = E[c] - g[c].transform("mean")
    for c in ["f10_avg", "f20_st"]:
        E[c + "_d"] = -(E[c] - g[c].transform("mean"))
    E["exh_d"] = -(E.exhibit_time - g.exhibit_time.transform("mean"))
    # 自作レーティング(走る前の強さ)。ファイルが無いときは0(=この特徴は使わない)
    try:
        RT = pd.read_csv("data/rating_feat.csv.gz", dtype={"venue": str})[["date", "venue", "race", "boat", "rt"]]
        E = E.merge(RT, on=["date", "venue", "race", "boat"], how="left")
        g = E.groupby("key")
    except FileNotFoundError:
        E["rt"] = 0.0
    E["rt_d"] = E.rt.fillna(0) - E.groupby("key").rt.transform(lambda s: s.fillna(0).mean())
    E[FEATS] = E[FEATS].fillna(0)
    mu, sd = E[FEATS].mean(), E[FEATS].std()
    E[FEATS] = (E[FEATS] - mu) / sd
    for i in range(1, 7):
        E[f"lane{i}"] = (E.boat == i).astype(int)
    E = E.sort_values(["date", "venue", "race", "boat"]).reset_index(drop=True)
    E = add_motor_features(E)
    return E, mu, sd


# ---------------- モーターの直近成績(選手の実力を引いたもの) ----------------
MOTOR = ["mo60_d", "mx60_d"]
RACER = ["nat_win_d", "loc_win_d", "cls_d", "weight_d", "f20_st_d", "f10_avg_d", "c_win_all_d"]
# モーターは1年ごとに全台が一斉に入れ替わる(2連率が0に戻る日)。入れ替え前後は別の個体として扱う
REPLACE = {"04": ["2024-06-13", "2025-06-09", "2026-06-18"], "05": ["2024-04-14", "2025-04-09", "2026-04-18"]}


def add_motor_features(E, ksh=20, n=60):
    """mo60_d / mx60_d: そのモーターを『他の選手』が使った直近n走の、選手の実力から見た期待との差(3着内/1着)。
    同じ開催中は同じ選手が使い続けるため、本人の過去の走は除く。レース内の平均との差にして標準化。"""
    E = E.copy()
    E["motor_no"] = pd.to_numeric(E.motor_no, errors="coerce")
    E.loc[(E.motor_no < 1) | (E.motor_no > 99), "motor_no"] = np.nan
    period = [int(np.searchsorted(REPLACE.get(v, []), d, side="right")) for v, d in zip(E.venue, E.date)]
    E["mkey"] = np.where(E.motor_no.notna(),
                         E.venue + "_" + pd.Series(period, index=E.index).astype(str) + "_" + E.motor_no.fillna(0).astype(int).astype(str),
                         None)
    # 選手の実力だけからの期待(場ごと)。残差 = 実際 − 期待
    E["r_top3"] = np.nan
    E["r_win"] = np.nan
    for v in E.venue.unique():
        m = E.venue == v
        X = LANES + RACER
        for tgt, col in (("top3", "r_top3"), ("win", "r_win")):
            lr = LogisticRegression(max_iter=3000).fit(E.loc[m, X], E.loc[m, tgt])
            E.loc[m, col] = E.loc[m, tgt] - lr.predict_proba(E.loc[m, X])[:, 1]
    E = E.sort_values(["mkey", "date", "race", "boat"], na_position="last").reset_index(drop=True)
    valid_pos = np.flatnonzero(E.mkey.notna().to_numpy())
    sub = E.iloc[valid_pos]
    reg = E.regno.to_numpy()
    for col, name in (("r_top3", "mo60"), ("r_win", "mx60")):
        resid = E[col].to_numpy()
        out = np.zeros(len(E))
        for _, idx in sub.groupby("mkey").indices.items():
            idx = valid_pos[np.asarray(idx)]
            r, rg = resid[idx], reg[idx]
            for i in range(1, len(idx)):
                prev = r[:i][rg[:i] != rg[i]][-n:]
                if len(prev):
                    out[idx[i]] = prev.sum() / (len(prev) + ksh)
        E[name] = out
    E = E.sort_values(["date", "venue", "race", "boat"]).reset_index(drop=True)
    g = E.groupby("key")
    for k in ("mo60", "mx60"):
        E[k + "_d"] = E[k] - g[k].transform("mean")
    global MOTOR_SD
    MOTOR_SD = E[MOTOR].std()  # 当日予想で同じ尺度にそろえるために保持
    E[MOTOR] = E[MOTOR] / MOTOR_SD
    return E  # r_top3 / r_win / mkey は当日予想でモーター成績を計算するために残す


MOTOR_SD = None


def norm_top3(p, keys):
    q = p.copy()
    for _ in range(10):
        q = q * 3 / q.groupby(keys).transform("sum")
        q = q.clip(upper=0.995)
    return q


def fit_boat_models(tr, extra):
    X = LANES + FEATS + extra
    mw = LogisticRegression(max_iter=3000).fit(tr[X], tr.win)
    m3 = LogisticRegression(max_iter=3000).fit(tr[X], tr.top3)
    return mw, m3


def predict_boat(models, te, extra):
    mw, m3 = models
    X = LANES + FEATS + extra
    te = te.copy()
    pw = pd.Series(mw.predict_proba(te[X])[:, 1], index=te.index)
    te["win_prob"] = pw / pw.groupby(te.key).transform("sum")
    te["top3_prob"] = norm_top3(pd.Series(m3.predict_proba(te[X])[:, 1], index=te.index), te.key)
    return te


KIM_CODE = {"逃げ": 0, "差し": 1, "まくり": 2, "まくり差し": 3, "抜き": 4, "恵まれ": 4}
KIM_NAMES = ["逃げ", "差し", "まくり", "まくり差し", "その他"]


def national_races():
    """全国の(日付, 場, R, 1〜3着艇, 決まり手コード)。決まり手が不明なレースは除く"""
    H = pd.read_csv("data/history.csv.gz", dtype={"venue": str, "regno": str, "finish": str})
    KM = pd.read_csv("data/kimarite.csv.gz", dtype={"venue": str}).fillna("")
    H = H[H.finish.isin(["01", "02", "03"])]
    P = H.pivot_table(index=["date", "venue", "race"], columns="finish", values="boat", aggfunc=["first", "count"])
    ok = (P["count"] == 1).all(axis=1)
    R = P["first"][ok].dropna().astype(int)
    R.columns = ["b1", "b2", "b3"]
    R = R.reset_index().merge(KM, on=["date", "venue", "race"], how="left")
    R["k"] = R.kimarite.map(KIM_CODE)
    return R.dropna(subset=["k"]).astype({"k": int})
