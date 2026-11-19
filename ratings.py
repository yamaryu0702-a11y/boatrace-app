"""自作レーティング(骨組みの部品2): 全国24場の全走から、1走ごとに選手の強さを更新する。

強さ s = 場×コースの基礎値 B + 選手の総合力 R + コース別の得手不得手 Rc(イン/差し/まくり/外) + 当地適性 Rv(場ごと)
1着確率 = softmax(s)。2着・3着は残りの艇で同じ計算(強さの差は係数κで縮める)。
走ったあと、実際の着順と期待の差で R, Rc, Rv, B を少しずつ動かす(Eloと同じ考え方)。
各走の「走る前の強さ」(更新前の値)を保存するので、未来の情報は混ざらない。
Rc, Rv は走数が少ないうちは0(総合力どおり)に寄る。

使い方:
  python ratings.py tune      # 学習率とκを、前半2年の着順の当てはまりで選ぶ(回収率は使わない)
  python ratings.py build     # 選んだ値で全期間を計算し、data/rating_feat.csv.gz と models/rating_state.pkl を作る
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GROUP = np.array([0, 1, 2, 2, 3, 3])   # コース1→イン, 2→差し, 3・4→まくり, 5・6→外
TUNE_END = "2025-10-21"                  # 前半2年(学習率などを選ぶ期間)の終わり
PARAMS_FILE = Path("models/rating_params.pkl")
STATE_FILE = "data/rating_state.pkl"   # 毎朝の自動更新でコミットされるよう data の中に置く
DEFAULT = dict(lr=0.04, kappa=0.7, sub=0.5, decay=0.02, lr_b=0.002)


def load():
    H = pd.read_csv("data/history.csv.gz", dtype={"venue": str, "regno": str, "finish": str, "course": str, "race": int})
    H = H[H.finish.str.fullmatch(r"0[1-6]", na=False) & H.course.str.fullmatch(r"[1-6]", na=False)].copy()
    H["fin"] = H.finish.astype(int)
    H["crs"] = H.course.astype(int) - 1
    H = H.sort_values(["date", "race", "venue", "fin"]).reset_index(drop=True)
    H["rkey"] = H.date + "_" + H.venue + "_" + H.race.astype(str)
    return H


def run(H, p, collect=False, until=None, skip=None):
    """H を時系列に1走ずつ処理。戻り値: (着順の対数損失[勝者,2着,3着], 走る前の強さの配列, 最終状態)"""
    regs = {r: i for i, r in enumerate(sorted(H.regno.unique()))}
    vens = {v: i for i, v in enumerate(sorted(H.venue.unique()))}
    R = np.zeros(len(regs)); Rc = np.zeros((len(regs), 4)); Rv = np.zeros((len(regs), len(vens)))
    B = np.zeros((len(vens), 6))
    ri_all = H.regno.map(regs).to_numpy(); vi_all = H.venue.map(vens).to_numpy()
    ci_all = H.crs.to_numpy(); fin_all = H.fin.to_numpy()
    bounds = np.flatnonzero(np.r_[True, H.rkey.to_numpy()[1:] != H.rkey.to_numpy()[:-1], True])
    pre = np.full(len(H), np.nan)
    pre_full = np.full(len(H), np.nan)
    lossk = np.zeros(3); nk = np.zeros(3)
    dates = H.date.to_numpy()
    kap = np.array([1.0, p["kappa"], p["kappa"]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 4:
            continue
        ri, vi, ci, fin = ri_all[a:b], vi_all[a:b], ci_all[a:b], fin_all[a:b]
        g = GROUP[ci]
        rt = R[ri] + Rc[ri, g] + Rv[ri, vi]
        s = B[vi, ci] + rt
        if collect:
            pre[a:b] = rt
            pre_full[a:b] = s
        if skip is not None and skip[a]:
            continue   # このレースの強さは計算するが、レーティングの更新には使わない(例: 女子戦)
        order = np.argsort(fin)
        grad = np.zeros(b - a)
        rem = list(range(b - a))
        track = until is None or dates[a] <= until
        for st in range(min(3, b - a - 1)):
            z = kap[st] * s[rem]
            z = z - z.max()
            e = np.exp(z); pr = e / e.sum()
            win = order[st]
            j = rem.index(win)
            if track:
                lossk[st] += -np.log(max(pr[j], 1e-12)); nk[st] += 1
            gv = -pr; gv[j] += 1.0
            grad[rem] += kap[st] * gv
            rem.remove(win)
        lr = p["lr"]
        R[ri] += lr * grad
        Rc[ri, g] += p["sub"] * lr * grad - p["decay"] * Rc[ri, g]
        Rv[ri, vi] += p["sub"] * lr * grad - p["decay"] * Rv[ri, vi]
        B[vi, ci] += p["lr_b"] * grad
    state = dict(regs=regs, vens=vens, R=R, Rc=Rc, Rv=Rv, B=B, params=p)
    return lossk / np.maximum(nk, 1), pre, pre_full, state


def tune():
    H = load()
    Hh = H[H.date <= TUNE_END].reset_index(drop=True)   # 前半2年だけで選ぶ
    best = None
    for lr in (0.08, 0.12, 0.16):
        for kappa in (0.6, 0.7, 0.85):
            p = dict(DEFAULT, lr=lr, kappa=kappa)
            # 前半2年の後半1年(最後の1年)の損失で比べる(更新は最初から)
            loss, *_ = run(Hh, p, until=None)
            tot = loss.sum()
            print(f"lr={lr} kappa={kappa}  損失 勝者 {loss[0]:.4f} 2着 {loss[1]:.4f} 3着 {loss[2]:.4f}  合計 {tot:.4f}", flush=True)
            if best is None or tot < best[0]:
                best = (tot, p)
    print("選んだ値:", best[1])
    Path("models").mkdir(exist_ok=True)
    pickle.dump(best[1], open(PARAMS_FILE, "wb"))


def build():
    H = load()
    p = pickle.load(open(PARAMS_FILE, "rb")) if PARAMS_FILE.exists() else DEFAULT
    loss, pre, pre_full, state = run(H, p, collect=True)
    print("パラメータ", p)
    print(f"全期間の損失 勝者 {loss[0]:.4f} 2着 {loss[1]:.4f} 3着 {loss[2]:.4f}")
    H["rt"], H["rt_full"] = pre, pre_full
    out = H[H.venue.isin(["04", "05"])][["date", "venue", "race", "boat", "rt", "rt_full"]].round(5)
    out.to_csv("data/rating_feat.csv.gz", index=False)
    pickle.dump(state, open(STATE_FILE, "wb"))
    print("data/rating_feat.csv.gz", len(out), "行 /", STATE_FILE)


def live_strength(state, regnos, venue, boats):
    """当日の予想用: 各艇の『走る前の強さ(場×コースの基礎値は含めない)』。進入コース=枠番として計算。
    レーティングに無い選手(新人など)は0(平均)。"""
    v = state["vens"].get(venue)
    out = []
    for r, b in zip(regnos, boats):
        i = state["regs"].get(r)
        if i is None or v is None:
            out.append(0.0)
            continue
        g = GROUP[int(b) - 1]
        out.append(float(state["R"][i] + state["Rc"][i, g] + state["Rv"][i, v]))
    return out


def build_nowomen():
    """女子戦(6艇が全員女子選手のレース)をレーティングの更新に使わない版。女子戦の走る前の強さは計算する"""
    H = load()
    W = pd.read_csv("data/women_races.csv", dtype={"venue": str}).drop_duplicates(["date", "venue", "race"])
    H = H.merge(W[["date", "venue", "race", "all_women"]], on=["date", "venue", "race"], how="left")
    H["all_women"] = H.all_women.fillna(0).astype(int)
    p = pickle.load(open(PARAMS_FILE, "rb")) if PARAMS_FILE.exists() else DEFAULT
    skip = (H.all_women == 1).to_numpy()
    print("更新に使わない女子戦の走", int(skip.sum()), "/", len(H))
    loss, pre, pre_full, state = run(H, p, collect=True, skip=skip)
    print(f"損失(一般戦のみ) 勝者 {loss[0]:.4f} 2着 {loss[1]:.4f} 3着 {loss[2]:.4f}")
    H["rt_nw"] = pre
    out = H[H.venue.isin(["04", "05"])][["date", "venue", "race", "boat", "rt_nw"]].round(5)
    out.to_csv("data/rating_feat_nw.csv.gz", index=False)
    pickle.dump(state, open("models/rating_state_nw.pkl", "wb"))
    print("data/rating_feat_nw.csv.gz", len(out), "行")


if __name__ == "__main__":
    {"tune": tune, "build": build, "build_nw": build_nowomen}[sys.argv[1]]()
