"""当日予想スクリプト(平和島・多摩川)。

使い方:
  python predict.py                         # 今日の平和島・多摩川の全レース
  python predict.py 2026-11-20 --venues 04  # 日付と場を指定(04=平和島, 05=多摩川)
  python predict.py --race 12               # 12Rだけ
  python predict.py --no-before             # 直前情報(展示タイム)を取らない(展示前の予想)
  python predict.py --update                # 先に最新の成績を取り込み、履歴・特徴量を更新する
  python predict.py 2026-11-14 --check      # 過去日で、当日用の特徴量が学習時と一致するか検証

出力: レースごとの 決まり手の確率 / イン逃げ確率 / 艇ごとの1着率・3連対率 / 3連単の上位、
      data/predict_YYYYMMDD.csv, data/predict_YYYYMMDD_trifecta.csv
"""
import argparse
import datetime as dt
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import beforeinfo as bi
import features as ft
import fetch as fetchmod
import tide736 as t7
from parse import parse_b
from trifecta_lib import COMBO, tri_probs

NAMES = {"04": "平和島", "05": "多摩川"}
RF = ["nat_win_d", "exh_d", "f20_st_d", "c_win_all_d", "cls_d", "motor_2r_d", "mo60_d"]
KIM_NAMES = ft.KIM_NAMES


# ------------------------------------------------------------ データ更新
def update_data(target):
    """最新の成績(K)・番組表(B)を取り込み、履歴と特徴量を作り直す"""
    last = max(p.stem[1:] for p in Path("data/raw").glob("K*.TXT"))
    last = dt.date(2000 + int(last[:2]), int(last[2:4]), int(last[4:6]))
    day = last + dt.timedelta(days=1)
    got = 0
    while day < target:
        ok = [fetchmod.fetch_one(k, day) for k in ("B", "K")]
        print(f"  取得 {day} B={'○' if ok[0] else '-'} K={'○' if ok[1] else '-'}")
        got += all(ok)
        day += dt.timedelta(days=1)
    if got:
        for s in ("parse.py", "build_form.py", "build_course.py", "build_kimarite.py"):
            print("  実行", s, flush=True)
            subprocess.run([sys.executable, "-W", "ignore", s], check=True, capture_output=True)
    else:
        print("  新しい成績はありません(更新なし)")


# ------------------------------------------------------------ 当日の特徴量
def history_form(H, regnos, boats):
    """学習時(build_form/build_course)と同じ定義で、選手の直近成績とコース別勝率を作る"""
    # コース別の全国平均(平滑化の事前分布)は、全選手の履歴から計算する
    allc = pd.to_numeric(H.course, errors="coerce")
    allw = (pd.to_numeric(H.finish, errors="coerce") == 1).astype(int)
    prior_all = allw[allc.between(1, 6) & H.finish.notna()].groupby(allc[allc.between(1, 6) & H.finish.notna()]).mean()
    H = H[H.regno.isin(regnos) & H.finish.notna()].copy()
    H["race"] = H.race.astype(int)
    H = H.sort_values(["regno", "date", "race"])
    fin = pd.to_numeric(H.finish, errors="coerce")
    H["fin_n"] = fin.fillna(6).clip(upper=6)
    H["win"] = (fin == 1).astype(int)
    st = pd.to_numeric(H.start_time, errors="coerce")
    st[(st < -0.2) | (st > 0.5)] = np.nan
    H["st"] = st
    H["cn"] = pd.to_numeric(H.course, errors="coerce")
    prior = prior_all
    out = {}
    for reg, b in zip(regnos, boats):
        h = H[H.regno == reg]
        last10, last20 = h.fin_n.tail(10), h.st.tail(20)
        hc = h[h.cn == b]
        out[(reg, b)] = {
            "f10_avg": last10.mean() if len(last10) >= 5 else np.nan,
            "f20_st": last20.mean() if last20.notna().sum() >= 5 else np.nan,
            "c_win_all": (hc.win.sum() + 15 * prior.get(float(b), 0.0)) / (len(hc) + 15),
        }
    return out


def motor_scores(E, L, date, venue):
    """モーターの直近成績(他の選手が使った直近60走の、実力から見た期待との差)"""
    if (L.motor_2r.fillna(0) == 0).all():
        print(f"  ※{NAMES[venue]}: 全モーターの2連率が0(入れ替え直後)。モーター成績は中立にします")
        return np.zeros(len(L)), np.zeros(len(L))
    period = int(np.searchsorted(ft.REPLACE.get(venue, []), date, side="right"))
    past = E[(E.date < date) & E.mkey.notna()]
    groups = {k: g for k, g in past.groupby("mkey")}
    mo, mx = [], []
    for _, r in L.iterrows():
        g = groups.get(f"{venue}_{period}_{int(r.motor_no)}") if pd.notna(r.motor_no) else None
        if g is None:
            mo.append(0.0); mx.append(0.0)
            continue
        g = g[g.regno != r.regno].tail(60)
        mo.append(g.r_top3.sum() / (len(g) + 20))
        mx.append(g.r_win.sum() / (len(g) + 20))
    return np.array(mo), np.array(mx)


def tide_info(date, deadline):
    d = dt.date.fromisoformat(date)
    try:
        t7.level(d, deadline, "haneda")
    except Exception:
        ym = d.strftime("%Y-%m")
        subprocess.run([sys.executable, "fetch_tide736.py", ym, ym], capture_output=True)
        t7.load.cache_clear()
    try:
        lv, nm = t7.level(d, deadline, "haneda"), t7.tide_name(d, "haneda")
        tr = t7.trend(d, deadline, "haneda")
        easy, hard = ft.tide_flags(date, deadline)
        return nm, lv, tr, easy, hard
    except Exception:
        return None, np.nan, 0.0, 0, 0


def get_before(venue, ymd, races):
    def one(r):
        try:
            return r, bi.parse(bi.fetch_html(venue, ymd, r))[0]
        except Exception:
            return r, []
    with ThreadPoolExecutor(max_workers=3) as ex:
        return dict(ex.map(one, races))


def build_live(date, venue, bent, brace, E, H, mu, sd, before):
    rows = []
    for (v, r, b), d in bent.items():
        if v != venue:
            continue
        rows.append({"date": date, "venue": v, "race": r, "boat": b, "regno": d["regno"], "name": d["name"],
                     "class": d["class"], "weight": float(d["weight"]), "nat_win": d["nat_win"],
                     "loc_win": d["loc_win"], "motor_no": d["motor_no"], "motor_2r": d["motor_2r"],
                     "boat_2r": d["boat_2r"], "deadline": brace.get((v, r), {}).get("deadline", "")})
    L = pd.DataFrame(rows)
    if L.empty:
        return L
    ex = {}
    for r, boats in before.items():
        for b in boats:
            ex[(r, b["boat"])] = b.get("exhibit_time", np.nan)
    L["exhibit_time"] = [ex.get((r, b), np.nan) if ex.get((r, b), "") != "" else np.nan for r, b in zip(L.race, L.boat)]
    L["exhibit_time"] = pd.to_numeric(L.exhibit_time, errors="coerce")
    fm = history_form(H, L.regno.tolist(), L.boat.tolist())
    for c in ("f10_avg", "f20_st", "c_win_all"):
        L[c] = [fm[(a, b)][c] for a, b in zip(L.regno, L.boat)]
    L["mo60"], L["mx60"] = motor_scores(E, L, date, venue)
    L["key"] = date + "_" + venue + "_" + L.race.astype(str)
    L["cls"] = L["class"].map({"A1": 3, "A2": 2, "B1": 1, "B2": 0})
    g = L.groupby("key")
    for c in ["nat_win", "loc_win", "motor_2r", "boat_2r", "cls", "weight", "c_win_all", "mo60", "mx60"]:
        L[c + "_d"] = L[c] - g[c].transform("mean")
    for c in ["f10_avg", "f20_st"]:
        L[c + "_d"] = -(L[c] - g[c].transform("mean"))
    L["exh_d"] = -(L.exhibit_time - g.exhibit_time.transform("mean"))
    L[ft.FEATS] = (L[ft.FEATS].fillna(0) - mu) / sd
    L[ft.MOTOR] = L[ft.MOTOR].fillna(0) / ft.MOTOR_SD
    for i in range(1, 7):
        L[f"lane{i}"] = (L.boat == i).astype(int)
    tide = {}
    L["easy_1"], L["hard_1"] = 0, 0
    if venue == "04":
        for r, dl in L.drop_duplicates("race")[["race", "deadline"]].itertuples(index=False):
            tide[r] = tide_info(date, dl) if dl else (None, np.nan, 0.0, 0, 0)
        L["easy_1"] = [tide[r][3] * (b == 1) for r, b in zip(L.race, L.boat)]
        L["hard_1"] = [tide[r][4] * (b == 1) for r, b in zip(L.race, L.boat)]
    L.attrs["tide"] = tide
    return L


def predict_venue(L, venue, date):
    pk = pickle.load(open(f"models/v4_{venue}.pkl", "rb"))
    extra = pk["boat_extra"]
    ok = L.groupby("key").boat.transform("count") == 6
    if (~ok).any():
        print(f"  ※出走が6艇でないレースを除外: {sorted(L[~ok].race.unique())}")
    L = L[ok].copy()
    L = ft.predict_boat(pk["boat_models"], L, extra)
    out_tri, race_rows = [], []
    for race, d in L.groupby("race"):
        d = d.sort_values("boat")
        w, t3 = d.win_prob.to_numpy(), d.top3_prob.to_numpy()
        feat = {f"{c}_{b}": float(d.set_index("boat").loc[b, c]) for c in RF for b in range(1, 7)}
        if venue == "04":
            feat["easy"], feat["hard"] = float(d.easy_1.max()), float(d.hard_1.max())
        x = np.array([[feat[c] for c in pk["kimarite_features"]]])
        pkim = np.zeros(len(KIM_NAMES))
        pkim[pk["kimarite_model"].classes_] = pk["kimarite_model"].predict_proba(x)[0]
        p = tri_probs(pkim, w, t3, pk["tables"], pk["bw"], pk["bt"], *pk["gamma"])
        race_rows.append((race, d, pkim, p))
    return race_rows, pk


def mando_prob(venue, d, pkim, p):
    """万舟(3連単1万円以上)になる確率。レース前の予想確率から計算(models/mando_*.pkl)"""
    try:
        m = pickle.load(open(f"models/mando_{venue}.pkl", "rb"))
    except FileNotFoundError:
        return None
    from trifecta_lib import PERMS
    cls = d[d.boat == 1]["class"].iloc[0]
    x = {"p1": p[PERMS[:, 0] == 0].sum(), "conf": p.max(), "ent": float(-(p * np.log(np.clip(p, 1e-12, None))).sum()),
         "pk0": pkim[0], "pk1": pkim[1], "pk2": pkim[2], "pk3": pkim[3],
         "rg_early": float(d.race.iloc[0] <= 4), "rg_late": float(d.race.iloc[0] >= 9),
         "A1": float(cls == "A1"), "B": float(cls in ("B1", "B2")),
         "easy": float(d.easy_1.max()), "hard": float(d.hard_1.max())}
    X = pd.DataFrame([x])[m["features"]]
    return float(m["model"].predict_proba((X - m["mu"]) / m["sd"])[0, 1])


def show(venue, date, race_rows, L, show_top):
    nm = NAMES[venue]
    boat_out, tri_out = [], []
    for race, d, pkim, p in race_rows:
        dl = d.deadline.iloc[0]
        line = f"\n【{nm} {race}R】 締切 {dl or '-'}"
        if venue == "04":
            tn, lv, tr, easy, hard = L.attrs["tide"].get(race, (None, np.nan, 0, 0, 0))
            if tn:
                judge = "逃げやすい" if easy else ("逃げにくい" if hard else "判定なし")
                flow = "上げ" if tr > 3 else ("下げ" if tr < -3 else "止まり")
                line += f"   潮: {tn} {lv:.0f}cm({flow}) → {judge}"
        print(line)
        esc = d.win_prob.iloc[0]
        label = "逃げ有力" if esc >= 0.55 else ("逃げ不利" if esc < 0.35 else "拮抗")
        print("  決まり手: " + "  ".join(f"{n}{x*100:.0f}%" for n, x in zip(KIM_NAMES, pkim)) + f"   ▶ 1号艇の1着 {esc*100:.0f}% ({label})")
        mp = mando_prob(venue, d, pkim, p)
        if mp is not None:
            print(f"  万舟(3連単1万円以上)確率: {mp*100:.0f}%  (この場の平均 {17 if venue == '05' else 19}%、目安: 25%以上=万舟が出やすい / 13%以下=出にくい)")
        print(f"  {'艇':<2}{'選手':<10}{'級':<3}{'1着率':>6}{'3連対率':>8}{'展示':>7}{'モーター2率':>9}")
        for _, r in d.iterrows():
            ex = f"{r.exhibit_time:.2f}" if pd.notna(r.exhibit_time) else "  -  "
            print(f"  {int(r.boat):<2}{r['name']:<10}{r['class']:<3}{r.win_prob*100:>5.1f}%{r.top3_prob*100:>7.1f}%{ex:>7}{r.motor_2r:>9.1f}")
            boat_out.append({"date": date, "venue": venue, "race": race, "boat": int(r.boat), "regno": r.regno, "name": r["name"],
                             "class": r["class"], "mando_prob": None if mp is None else round(mp, 4),
                             "win_prob": round(r.win_prob, 4), "top3_prob": round(r.top3_prob, 4),
                             "exhibit_time": r.exhibit_time, "motor_2r": r.motor_2r})
        top = np.argsort(-p)[:show_top]
        print("  3連単: " + "  ".join(f"{COMBO[i]} {p[i]*100:.1f}%" for i in top))
        for rank, i in enumerate(np.argsort(-p)[:30], 1):
            tri_out.append({"date": date, "venue": venue, "race": race, "rank": rank, "combo": COMBO[i], "prob": round(p[i], 5)})
        collect_app(venue, date, race, d, pkim, p, mp, label, L)
    return boat_out, tri_out


APP = {"races": [], "boats": [], "tri": [], "second": [], "third": []}


def condition_tags(venue, race, pkim, p1, mp, easy):
    """検証で回収率が比較的高かった条件の目印(どれも100%未満。買いの推奨ではない)"""
    tags = []
    if venue == "05":
        if pkim[0] >= 0.40:
            tags.append("逃げ予想40%以上→3連単3点(検証88%)")
        if 0.45 <= p1 < 0.60:
            tags.append("1号艇1着45〜60%→3連単3点(検証90%)")
        if p1 >= 0.60:
            tags.append("1号艇1着60%以上→単勝1点(検証101%)")
    else:
        if p1 >= 0.60 and easy:
            tags.append("1号艇有力+潮が逃げやすい→3連単20点(検証92%)")
        if p1 >= 0.60 and race >= 9:
            tags.append("1号艇有力・9R以降→3連単20点(検証87%)")
    if mp is not None and mp >= 0.25:
        tags.append("万舟注意(25%以上)")
    return tags


def collect_app(venue, date, race, d, pkim, p, mp, label, L):
    """Webアプリ用のデータ(レース・艇・3連単の上位)を集める"""
    from trifecta_lib import PERMS
    p1 = float(p[PERMS[:, 0] == 0].sum())
    tn, lv, tr, easy, hard = (None, np.nan, 0.0, 0, 0)
    if venue == "04":
        tn, lv, tr, easy, hard = L.attrs["tide"].get(race, (None, np.nan, 0.0, 0, 0))
    top = np.argsort(-p)
    label = "逃げ有力" if p1 >= 0.55 else ("逃げ不利" if p1 < 0.35 else "拮抗")  # 3連単の確率ベース(検証の条件と同じ定義)
    tags = condition_tags(venue, race, pkim, p1, mp, easy)
    rid = f"{venue}-{int(race)}"
    APP["races"].append({
        "rid": rid, "venue": venue, "venue_name": NAMES[venue], "race": int(race), "date": date, "deadline": d.deadline.iloc[0] or "",
        "p_nige": round(pkim[0] * 100), "p_sashi": round(pkim[1] * 100), "p_makuri": round(pkim[2] * 100),
        "p_mz": round(pkim[3] * 100), "p_other": round(pkim[4] * 100),
        "p1": round(p1 * 100), "p1_label": label, "mando": None if mp is None else round(mp * 100),
        "tide": "" if not tn else f"{tn} {lv:.0f}cm",
        "tide_judge": "" if not tn else ("逃げやすい" if easy else ("逃げにくい" if hard else "判定なし")),
        "tags": " / ".join(tags), "n_tags": len(tags),
        "top_combo": COMBO[top[0]], "top_prob": round(float(p[top[0]]) * 100, 1),
    })
    for _, r in d.iterrows():
        APP["boats"].append({"rid": rid, "venue": venue, "race": int(race), "boat": int(r.boat), "name": r["name"], "class": r["class"],
                             "win": round(r.win_prob * 100, 1), "top3": round(r.top3_prob * 100, 1),
                             "exhibit": None if pd.isna(r.exhibit_time) else float(r.exhibit_time), "motor_2r": float(r.motor_2r)})
    # 各艇が1着のとき、2着がどの艇になるかの確率(%)
    for a in range(6):
        pa = float(p[PERMS[:, 0] == a].sum())
        for b in range(6):
            if b == a:
                continue
            pb = float(p[(PERMS[:, 0] == a) & (PERMS[:, 1] == b)].sum())
            APP["second"].append({"rid": rid, "first": a + 1, "second": b + 1, "p_first": round(pa * 100, 1),
                                  "prob": round(pb / pa * 100, 1) if pa > 0 else 0.0})
    # 1着・2着が決まったとき、3着がどの艇になるかの確率(%)
    for a in range(6):
        for b in range(6):
            if b == a:
                continue
            pab = float(p[(PERMS[:, 0] == a) & (PERMS[:, 1] == b)].sum())
            for c in range(6):
                if c in (a, b):
                    continue
                pc = float(p[(PERMS[:, 0] == a) & (PERMS[:, 1] == b) & (PERMS[:, 2] == c)].sum())
                APP["third"].append({"rid": rid, "first": a + 1, "second": b + 1, "third": c + 1, "p_pair": round(pab * 100, 2),
                                     "prob": round(pc / pab * 100, 1) if pab > 0 else 0.0})
    for rank, i in enumerate(top[:12], 1):
        APP["tri"].append({"rid": rid, "venue": venue, "race": int(race), "rank": rank, "combo": COMBO[i], "prob": round(float(p[i]) * 100, 2)})


def write_app(date, trained):
    import json
    Path("data/app").mkdir(parents=True, exist_ok=True)
    meta = [{"label": "予想日", "value": date}, {"label": "モデルの学習", "value": f"{trained}まで"},
            {"label": "作成", "value": "predict.py(指数v4+潮ルール+モーター成績)"}]
    for k in ("races", "boats", "tri", "second", "third"):
        json.dump({"rows": APP[k], "meta": meta}, open(f"data/app/{k}.json", "w", encoding="utf-8"), ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", default=dt.date.today().isoformat())
    ap.add_argument("--venues", nargs="+", default=["04", "05"])
    ap.add_argument("--race", type=int)
    ap.add_argument("--no-before", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    date = a.date
    target = dt.date.fromisoformat(date)
    ymd = target.strftime("%y%m%d")

    if a.update:
        print("■ データ更新")
        update_data(target)
    print("■ 準備: 学習データと特徴量を読み込み中...")
    E, mu, sd = ft.build_dataset()
    H = pd.read_csv("data/history.csv.gz", dtype={"venue": str, "regno": str, "finish": str, "course": str})
    H = H[H.date < date]
    print(f"  成績データ: {H.date.max()} まで / 学習データ: {E.date.max()} まで")

    bpath = Path(f"data/raw/B{ymd}.TXT")
    if not bpath.exists() and not fetchmod.fetch_one("B", target):
        print(f"{date} の番組表がありません(開催なし、または未公開)")
        return
    bent, brace = parse_b(bpath, date)
    all_boat, all_tri = [], []
    for v in a.venues:
        if not any(k[0] == v for k in bent):
            print(f"\n{NAMES[v]}: {date} は開催なし")
            continue
        races = sorted({k[1] for k in bent if k[0] == v})
        if a.race:
            races = [a.race]
            bent_v = {k: d for k, d in bent.items() if not (k[0] == v and k[1] != a.race)}
        else:
            bent_v = bent
        before = {} if a.no_before else get_before(v, target.strftime("%Y%m%d"), races)
        L = build_live(date, v, bent_v, brace, E, H, mu, sd, before)
        if a.check:
            check(E, L, v, date)
            continue
        rr, pk = predict_venue(L, v, date)
        no_exh = L.exhibit_time.isna().all()
        print(f"\n{'='*70}\n■ {NAMES[v]} {date}  ({len(rr)}レース)  モデル学習: {pk['trained_through']}まで"
              + ("  ※展示タイム未取得(展示前の予想)" if no_exh else ""))
        b, t = show(v, date, rr, L, a.top)
        all_boat += b
        all_tri += t
    if all_boat:
        pd.DataFrame(all_boat).to_csv(f"data/predict_{target:%Y%m%d}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(all_tri).to_csv(f"data/predict_{target:%Y%m%d}_trifecta.csv", index=False, encoding="utf-8-sig")
        write_app(date, pk["trained_through"])
        print(f"\n出力: data/predict_{target:%Y%m%d}.csv, data/predict_{target:%Y%m%d}_trifecta.csv, data/app/*.json(Webアプリ用)")


def check(E, L, venue, date):
    """当日用の特徴量が、学習データの同じ行の特徴量と一致するか"""
    ref = E[(E.date == date) & (E.venue == venue)].merge(L[["race", "boat"]], on=["race", "boat"])
    cols = ft.FEATS + ft.MOTOR
    m = L.merge(ref[["race", "boat"] + cols], on=["race", "boat"], suffixes=("", "_ref"))
    print(f"\n■ 検証 {NAMES[venue]} {date}: 比較できる艇 {len(m)}")
    print(f"  {'特徴量':<14}{'一致(差<0.02)':>14}{'最大の差':>10}")
    for c in cols:
        d = (m[c] - m[c + "_ref"]).abs()
        print(f"  {c:<14}{(d < 0.02).mean()*100:>13.0f}%{d.max():>10.3f}")


if __name__ == "__main__":
    main()
