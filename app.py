"""ボートレース予想(Streamlit)。平和島・多摩川。
- 一覧(表紙)でレースを選ぶと詳細ページへ。展示タイムは、ボタンで公式サイトから取り込む/手入力で、その場で再計算する。
- 予想の計算は predict.py と同じもの(指数v4)。成績データは、GitHub Actionsが毎朝更新する。
"""
import datetime as dt
import itertools
import json
import re
import threading
import unicodedata

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import features as ft
import fetch as fetchmod
import predict as P
from parse import parse_b

JST = dt.timezone(dt.timedelta(hours=9))
NAMES = {"04": "平和島", "05": "多摩川"}
BOAT = ["#ffffff", "#1a1a1a", "#d93025", "#1a56db", "#f4c20d", "#1e9e4a"]
EMOJI = ["⚪", "⚫", "🔴", "🔵", "🟡", "🟢"]
KIM = [("逃げ", "p_nige", "#0e7490"), ("差し", "p_sashi", "#f59e0b"), ("まくり", "p_makuri", "#e11d48"),
       ("まくり差し", "p_mz", "#7c3aed"), ("その他", "p_other", "#94a3b8")]
@st.cache_resource
def _get_lock():
    """予想の計算中に、別の計算が割り込まないようにするための鍵。
    ページの更新のたびにこのファイルは読み直されるので、鍵も作り直されると意味がない(平和島と多摩川の結果が混ざる原因になる)。
    cache_resource に入れて、全員・全回で同じ鍵を使う。"""
    return threading.Lock()


LOCK = _get_lock()

st.set_page_config(page_title="ボートレース予想", page_icon="🚤", layout="wide")
st.markdown("""
<style>
  :root { color-scheme: dark; }
  .stApp {
    background:
      radial-gradient(900px 480px at 8% -8%, rgba(34,211,238,.16), transparent 62%),
      radial-gradient(800px 460px at 100% 0%, rgba(99,102,241,.18), transparent 58%),
      radial-gradient(700px 500px at 50% 115%, rgba(14,116,144,.14), transparent 60%),
      linear-gradient(180deg,#06080c 0%,#0a0f17 55%,#06080c 100%);
    background-attachment: fixed;
    color: #e8edf5;
  }
  header[data-testid="stHeader"] { background: transparent; }
  .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
  .stApp div[data-testid="stMarkdownContainer"], .stApp div[data-testid="stCaptionContainer"] { color: #e8edf5; }
  .stApp h2, .stApp h3 { letter-spacing: .03em; }
  .stApp h2 { border-left: 4px solid #22d3ee; padding-left: 12px; }
  .stButton > button { background: rgba(34,211,238,.10); border: 1px solid #22d3ee; color: #67e8f9; border-radius: 999px; }
  .stButton > button:hover { background: rgba(34,211,238,.22); border-color: #67e8f9; color: #fff; }
  div[data-testid="stExpander"], div[data-testid="stDataFrame"] { background: rgba(17,21,28,.72); border: 1px solid #232a36; border-radius: 14px; }
  div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
  .hero { background: linear-gradient(135deg,#0a1426 0%,#0f2a4d 55%,#0b5d73 135%); border:1px solid #1c2b44; border-radius:16px; padding:18px 22px; margin-bottom:12px; box-shadow:0 14px 40px -18px rgba(34,211,238,.35); }
  .hero h1 { color:#fff; font-size:1.7rem; margin:0; letter-spacing:.04em; }
  .hero p { color:#a9bddb; margin:6px 0 0; font-size:.9rem; }
  .wrap { overflow-x:auto; border:1px solid #232a36; border-radius:14px; background:#11151c; box-shadow:0 8px 24px -16px #000; margin:6px 0 10px; }
  .wrap table { border-collapse:collapse; width:100%; font-size:.82rem; font-variant-numeric:tabular-nums; }
  .wrap th { background:#0d1219; color:#9fb0c6; font-weight:600; font-size:.74rem; letter-spacing:.03em; text-align:right; padding:8px 10px; border-bottom:1px solid #232a36; white-space:nowrap; }
  .wrap td { text-align:right; padding:7px 10px; border-bottom:1px solid #1b212b; white-space:nowrap; color:#e8edf5; }
  .wrap th.l, .wrap td.l { text-align:left; }
  .wrap tr:last-child td { border-bottom:0; }
  .wrap td.race { font-weight:700; color:#22d3ee; font-size:1.05em; }
  .wrap td.hot { color:#fbbf24; background:rgba(251,191,36,.14); font-weight:700; }
  .wrap td.good { color:#34d399; background:rgba(52,211,153,.14); font-weight:700; }
  .rl { color:#22d3ee; text-decoration:none; border-bottom:1px dashed #22d3ee; }
  .rl:hover { color:#fff; }
  .chip { display:inline-grid; place-items:center; width:22px; height:22px; border-radius:6px; font-size:12px; font-weight:700; border:1px solid rgba(255,255,255,.35); margin-right:2px; }
  .seg { display:flex; height:18px; border-radius:9px; overflow:hidden; background:#232a36; margin:6px 0; }
  .seg span { display:block; height:100%; }
  .legend { display:flex; flex-wrap:wrap; gap:4px 14px; font-size:.8rem; margin-bottom:8px; }
  .legend i { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }
  .card { border:1px solid #232a36; border-radius:14px; background:#11151c; padding:12px 16px; box-shadow:0 8px 24px -16px #000; margin-bottom:10px; }
  .card h4 { margin:0 0 6px; font-size:.95rem; }
  .tag { display:inline-block; padding:3px 10px; margin:2px 6px 2px 0; border-radius:10px; border-left:3px solid #22d3ee; background:#0d1219; font-size:.85rem; }
  .tag.warn { border-left-color:#fbbf24; color:#fbbf24; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ データ
@st.cache_resource(show_spinner="学習データと特徴量を読み込み中…(初回のみ)")
def _load():
    E, mu, sd = ft.build_dataset()
    H = pd.read_csv("data/history.csv.gz", dtype={"venue": str, "regno": str, "finish": str, "course": str})
    return E, mu, sd, H, ft.MOTOR_SD


def load_state():
    # build_dataset が features.MOTOR_SD を設定する。ファイル更新でモジュールが読み込み直されると、
    # キャッシュ済みの結果だけが残って MOTOR_SD が None に戻るので、毎回、設定し直す
    E, mu, sd, H, msd = _load()
    ft.MOTOR_SD = msd
    return E, mu, sd, H


@st.cache_data(ttl=900, show_spinner="番組表を取得中…")
def get_program(date_iso):
    day = dt.date.fromisoformat(date_iso)
    from pathlib import Path
    if not fetchmod.fetch_one("B", day):
        return None
    return parse_b(Path(f"data/raw/B{day:%y%m%d}.TXT"), date_iso)


@st.cache_data(show_spinner="予想を計算中…", max_entries=30)
def compute(date_iso, venue, before_json):
    E, mu, sd, H = load_state()
    prog = get_program(date_iso)
    bent, brace = prog
    before = {int(k): v for k, v in json.loads(before_json).items()}
    L = P.build_live(date_iso, venue, bent, brace, E, H[H.date < date_iso], mu, sd, before)
    with LOCK:
        for k in P.APP:
            P.APP[k] = []
        rr, pk = P.predict_venue(L, venue, date_iso)
        for race, d, pkim, p in rr:
            P.collect_app(venue, date_iso, race, d, pkim, p, P.mando_prob(venue, d, pkim, p), "", L)
        out = {k: pd.DataFrame(v) for k, v in P.APP.items()}
    # 念のため、この場(venue)のレースだけに絞る(rid は「場コード-レース番号」)
    for k, df in out.items():
        if "rid" in df.columns:
            out[k] = df[df.rid.astype(str).str.startswith(venue + "-")].reset_index(drop=True)
    return out, pk["trained_through"]


@st.cache_resource
def exh_store():
    """展示タイムの取り込み結果・手入力。ページを開き直しても残るよう、サーバー側に置く"""
    return {"before": {}, "manual": {}}


GAPS = [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12]


@st.cache_data(show_spinner="確率の変動を計算中…", max_entries=40)
def sensitivity(date_iso, venue, race, before_json):
    """1艇だけ展示タイムを動かして(平均より g 秒速い)、その艇の1着率・3連対率がどう変わるかをモデルで計算する"""
    E, mu, sd, H = load_state()
    bent, brace = get_program(date_iso)
    before = {int(k): v for k, v in json.loads(before_json).items()}
    L = P.build_live(date_iso, venue, bent, brace, E, H[H.date < date_iso], mu, sd, before)
    L = L[L.race == race].sort_values("boat").reset_index(drop=True)
    t = L.exhibit_time.to_numpy(float)
    real = not np.isnan(t).any()
    base = t if real else np.full(6, 6.75)
    out = {}
    with LOCK:
        for k in range(6):
            for g in GAPS:
                tt = base.copy()
                tt[k] = np.delete(base, k).mean() - g * 6 / 5
                raw = -(tt - tt.mean())
                L2 = L.copy()
                L2["exhibit_time"] = tt
                L2["exh_d"] = (raw - float(mu["exh_d"])) / float(sd["exh_d"])
                rr, _ = P.predict_venue(L2, venue, date_iso)
                d = rr[0][1].set_index("boat")
                out[(k + 1, g)] = (float(d.loc[k + 1, "win_prob"]) * 100, float(d.loc[k + 1, "top3_prob"]) * 100)
    cur = [float(np.mean(t) - x) for x in t] if real else None
    return out, cur


def merged_before(date_iso, venue):
    store = exh_store()
    fetched = store["before"].get((date_iso, venue), {})
    manual = store["manual"].get((date_iso, venue), {})
    out = {}
    for race in set(fetched) | set(manual):
        d = {int(b["boat"]): dict(b) for b in fetched.get(race, [])}
        for boat, t in manual.get(race, {}).items():
            d.setdefault(int(boat), {"boat": int(boat)})["exhibit_time"] = t
        out[int(race)] = list(d.values())
    return json.dumps(out, sort_keys=True, default=str)


def fetch_exhibit(date_iso, venue, races):
    ymd = date_iso.replace("-", "")
    got = P.get_before(venue, ymd, races)
    exh_store()["before"].setdefault((date_iso, venue), {}).update({r: v for r, v in got.items() if v})
    n = sum(1 for v in got.values() if any(str(b.get("exhibit_time", "")).strip() not in ("", "nan") for b in v))
    return n, len(races)


# ------------------------------------------------------------------ 表示部品
def pie(df, key, title):
    d = df.sort_values(key).copy()
    d["label"] = [f"{int(b)}号艇 {p}%" for b, p in zip(d[key], d.prob)]
    colors = [BOAT[int(b) - 1] for b in d[key]]
    ch = alt.Chart(d).mark_arc(stroke="#8a8a8a", strokeWidth=1, outerRadius=95).encode(
        theta=alt.Theta("prob:Q", stack=True),
        color=alt.Color("label:N", scale=alt.Scale(domain=list(d.label), range=colors), legend=alt.Legend(title=title, orient="right")),
        order=alt.Order(f"{key}:Q"),
        tooltip=["label:N"]).properties(height=230)
    return ch.configure(background="transparent").configure_view(strokeWidth=0).configure_legend(labelColor="#e8edf5", titleColor="#9fb0c6", labelFontSize=12)


def kim_bar(r):
    segs = "".join(f'<span style="width:{max(float(r[f]), 0.5)}%;background:{c}"></span>' for _, f, c in KIM)
    leg = "".join(f'<span><i style="background:{c}"></i>{n} <b>{int(r[f])}</b>%</span>' for n, f, c in KIM)
    return f'<div class="seg">{segs}</div><div class="legend">{leg}</div>'


def combo_emoji(c):
    return " ".join(EMOJI[int(x) - 1] + x for x in c.split("-"))


def chip(n):
    n = int(n)
    return f'<span class="chip" style="background:{BOAT[n - 1]};color:{"#1a1a1a" if n in (1, 5) else "#fff"}">{n}</span>'


def combo_html(c):
    return "".join(chip(x) for x in str(c).split("-"))


def shade(v, mx):
    if v is None or pd.isna(v):
        return ""
    w = max(0, min(100, float(v) / mx * 100))
    return f"background:linear-gradient(90deg,rgba(34,211,238,.28) {w:.0f}%,transparent {w:.0f}%)"


def num(v, fmt="{:.0f}"):
    return "—" if v is None or pd.isna(v) else fmt.format(v)


def html_table(heads, rows):
    """heads: [(ラベル, 左寄せか)]、rows: [[(html, class, style), ...]]。1行の文字列にして返す(markdownに崩されないため)"""
    th = "".join(f'<th class="{"l" if l else ""}">{h}</th>' for h, l in heads)
    body = "".join("<tr>" + "".join(f'<td class="{c}" style="{s}">{h}</td>' for h, c, s in row) + "</tr>" for row in rows)
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


# ------------------------------------------------------------------ 本体
today = dt.datetime.now(JST).date()
st.markdown('<div class="hero"><h1>🚤 ボートレース予想</h1><p>平和島・多摩川　指数v4(決まり手 × 艇の強さ × 潮 × モーター)</p></div>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 2])
qp = st.query_params
try:
    q_default = dt.date.fromisoformat(qp.get("d", ""))
except ValueError:
    q_default = today
date = c1.date_input("日付", q_default)
date_iso = date.isoformat()
E, mu, sd, H = load_state()
hmax = H.date.max()
note = f"成績データ: {hmax} まで"
if dt.date.fromisoformat(hmax) < date - dt.timedelta(days=2):
    note += "(⚠ 更新が止まっています。選手の直近成績が古い可能性があります)"
c2.caption(note)

try:
    prog = get_program(date_iso)
except Exception as e:  # noqa: BLE001
    prog = None
    st.error(f"番組表を取得できませんでした: {e}")
if prog is None:
    st.info(f"{date_iso} の番組表がありません(開催なし、または未公開)。")
    st.stop()
bent, brace = prog
venues = [v for v in NAMES if any(k[0] == v for k in bent)]
if not venues:
    st.info(f"{date_iso} は、平和島・多摩川の開催がありません。")
    st.stop()

venue = st.radio("場", venues, index=venues.index(qp.get("v")) if qp.get("v") in venues else 0, format_func=lambda v: NAMES[v], horizontal=True) if len(venues) > 1 else venues[0]
if len(venues) == 1:
    st.subheader(NAMES[venue])
races = sorted({k[1] for k in bent if k[0] == venue})

ss = st.session_state
if not ss.get("qp_done"):   # レース名のリンクから開いたとき、そのレースの詳細に飛ぶ
    ss["qp_done"] = True
    if qp.get("race", "").isdigit() and int(qp["race"]) in races and qp.get("d") == date_iso and qp.get("v") == venue:
        ss["ctx"] = (date_iso, venue)
        ss["race"] = int(qp["race"])
if ss.get("ctx") != (date_iso, venue):
    ss["ctx"] = (date_iso, venue)
    ss["race"] = None

data, trained = compute(date_iso, venue, merged_before(date_iso, venue))
R, B, T, S2, S3 = data["races"], data["boats"], data["tri"], data["second"], data["third"]
TA = data["tri_all"]   # 3連単120通りすべて(出目入力用)

# ---- 表紙(レース一覧) ----
if ss.get("race") is None:
    b1, b2 = st.columns([1, 3])
    if b1.button("🔄 全レースの展示タイムを取り込む", width="stretch"):
        with st.spinner("公式サイトから展示タイムを取得中…"):
            n, tot = fetch_exhibit(date_iso, venue, races)
        if n == 0:
            st.warning("展示タイムを取得できたレースがありません(展示前、またはサイトに接続できません)。レースを開いて、手入力もできます。")
        else:
            st.success(f"{n}/{tot} レースで展示タイムを取り込みました。")
        st.rerun() if n else None
    n_ex = int(B.groupby("race").exhibit.apply(lambda s: s.notna().all()).sum())
    b2.caption(f"展示タイム入りのレース: {n_ex}/{len(races)}　／　モデルの学習: {trained} まで")
    heads = [("R", True), ("締切", False), ("1号艇の1着(%)", False), ("逃げ(%)", False), ("差し(%)", False), ("まくり(%)", False),
             ("まくり差し(%)", False), ("万舟(%)", False), ("潮", True), ("潮の判定", True), ("目印", False), ("本命の3連単", False), ("確率(%)", False)]
    rows = []
    for _, x in R.iterrows():
        mando_hot = pd.notna(x.mando) and x.mando >= 25
        rows.append([(f'<a class="rl" href="?d={date_iso}&amp;v={venue}&amp;race={int(x.race)}" target="_self">{int(x.race)}R</a>', "l race", ""), (x.deadline or "—", "", ""), (num(x.p1), "", shade(x.p1, 100)),
                     (num(x.p_nige), "", ""), (num(x.p_sashi), "", ""), (num(x.p_makuri), "", ""), (num(x.p_mz), "", ""),
                     (num(x.mando), "hot" if mando_hot else "", ""), (x.tide or "—", "l", ""), (x.tide_judge or "—", "l", ""),
                     (str(int(x.n_tags)), "good" if x.n_tags > 0 else "", ""), (combo_html(x.top_combo), "", ""), (num(x.top_prob, "{:.1f}"), "", "")])
    st.markdown(html_table(heads, rows), unsafe_allow_html=True)
    st.caption("レース名(1R, 2R …)を押すと、そのレースの詳細ページに移ります。")
    st.query_params.clear()
    st.stop()

# ---- 詳細ページ ----
race = ss["race"]
r = R[R.race == race].iloc[0]
st.query_params.update({"d": date_iso, "v": venue, "race": str(race)})
idx = races.index(race)
n1, n2, n3 = st.columns(3)
if n1.button("← レース一覧", width="stretch"):
    ss["race"] = None
    st.query_params.clear()
    st.rerun()
if n2.button("‹ 前のレース", width="stretch", disabled=idx == 0):
    ss["race"] = races[idx - 1]
    st.rerun()
if n3.button("次のレース ›", width="stretch", disabled=idx == len(races) - 1):
    ss["race"] = races[idx + 1]
    st.rerun()
st.header(f"{NAMES[venue]} {race}R　締切 {r.deadline or '—'}")
if r.tide:
    st.caption(f"潮: {r.tide}　→　{r.tide_judge}")

# 展示タイム(取り込み/手入力)
bb = B[B.rid == r.rid].sort_values("boat")
f1, f2 = st.columns([1, 3])
if f1.button("このレースの展示タイムを取り込む", width="stretch"):
    with st.spinner("取得中…"):
        n, tot = fetch_exhibit(date_iso, venue, [race])
    if n:
        st.rerun()
    st.warning("展示タイムを取得できませんでした(展示前、またはサイトに接続できません)。下で手入力できます。")
with st.expander("展示タイムを手入力して、再計算する", expanded=False):
    with st.form(f"manual_{race}"):
        cols = st.columns(6)
        vals = {}
        for i, (_, b) in enumerate(bb.iterrows()):
            vals[int(b.boat)] = cols[i].number_input(f"{EMOJI[int(b.boat) - 1]} {int(b.boat)}号艇", min_value=0.0, max_value=9.99,
                                                    value=0.0 if pd.isna(b.exhibit) else float(b.exhibit), step=0.01, format="%.2f")
        st.caption("0.00 は未入力として扱います。6艇すべて入れると、展示タイムの差を使った予想になります。")
        if st.form_submit_button("この値で再計算"):
            m = exh_store()["manual"].setdefault((date_iso, venue), {}).setdefault(race, {})
            for k, v in vals.items():
                if v > 0:
                    m[k] = round(v, 2)
                else:
                    m.pop(k, None)
            st.rerun()

# 艇の表
heads = [("艇", True), ("選手", True), ("級", True), ("1着率(%)", False), ("3連対率(%)", False), ("展示タイム", False), ("モーター2連率(%)", False)]
rows = [[(chip(x.boat), "l", ""), (str(x["name"]), "l", ""), (str(x["class"]), "l", ""), (num(x.win, "{:.1f}"), "", shade(x.win, 80)),
         (num(x.top3, "{:.1f}"), "", shade(x.top3, 100)), (num(x.exhibit, "{:.2f}"), "", ""), (num(x.motor_2r, "{:.1f}"), "", "")]
        for _, x in bb.iterrows()]
st.markdown(html_table(heads, rows), unsafe_allow_html=True)

g1, g2 = st.columns(2)
with g1:
    tags = [x for x in str(r.tags).split(" / ") if x]
    tag_html = "".join(f'<span class="tag{" warn" if x.startswith("万舟") else ""}">{x}</span>' for x in tags) or "当てはまる目印はありません"
    st.markdown(f'<div class="card"><h4>決まり手の確率(%)</h4>{kim_bar(r)}<p style="font-size:.8rem;margin:0 0 8px">1号艇の1着 <b>{r.p1}%</b>　万舟 <b>{"—" if pd.isna(r.mando) else int(r.mando)}%</b></p>'
                f'<h4>目印</h4><div style="font-size:.78rem;color:#8b97a8;margin-bottom:4px">検証で回収率が比較的高かった条件。買いの推奨ではありません</div>{tag_html}</div>', unsafe_allow_html=True)
with g2:
    tt = T[T.rid == r.rid].sort_values("rank")
    rows = [[(str(int(x["rank"])), "l", ""), (combo_html(x.combo), "l", ""), (num(x.prob, "{:.2f}"), "", shade(x.prob, 10))] for _, x in tt.iterrows()]
    st.markdown('<div class="card"><h4>3連単の確率 上位12通り</h4>' + html_table([("順位", True), ("出目", True), ("確率(%)", False)], rows) + "</div>", unsafe_allow_html=True)

st.markdown("**買い目の確率を見る(直接入力・フォーメーション・ボックス)**")
st.caption("買い目は最大3セットまで。セットごとに、入力のしかたを選べます。確率はモデルの推定値です。")
ta = TA[TA.rid == r.rid].set_index("combo")


def parse_direct(txt):
    seen, bad = [], []
    for tok in re.split(r"[,\s、，]+", txt.strip()):
        if not tok:
            continue
        ds = re.findall(r"[1-6１-６]", tok)
        ds = [str("123456".index(unicodedata.normalize("NFKC", x)) + 1) for x in ds]
        if len(ds) != 3 or len(set(ds)) != 3 or len(re.sub(r"[1-6１-６\-－ー・/]", "", tok)) > 0:
            bad.append(tok)
        elif "-".join(ds) not in seen:
            seen.append("-".join(ds))
    return seen, bad


def boat_pick(label, key):
    """艇のボタンをタップして選ぶ(複数選択。もう一度タップすると外れる)"""
    st.markdown(f'<div style="font-weight:700;margin:6px 0 2px">{label}</div>', unsafe_allow_html=True)
    got = st.pills(label, [1, 2, 3, 4, 5, 6], selection_mode="multi", key=key, format_func=lambda x: f"{EMOJI[x - 1]} {x}", label_visibility="collapsed")
    return got or []


def set_ui(i):
    """1セット分の買い目を選ぶ。戻り値: (出目のリスト, 読み取れなかった入力)"""
    mode = st.radio("買い方", ["フォーメーション", "ボックス", "直接入力"], horizontal=True, key=f"bm_{race}_{i}")
    if mode == "直接入力":
        txt = st.text_area("出目(例: 1-2-3, 1-3-2, 2-1-3 のように、カンマ・スペース・改行で区切って、いくつでも)", key=f"bd_{race}_{i}", height=80,
                           placeholder="1-2-3, 1-3-2\n2-1-3")
        return parse_direct(txt) if txt.strip() else ([], [])
    if mode == "ボックス":
        bs = boat_pick("ボックスにする艇(3艇以上をタップ)", f"bb_{race}_{i}")
        if len(bs) < 3:
            return [], []
        return ["-".join(map(str, p)) for p in itertools.permutations(sorted(bs), 3)], []
    f1 = boat_pick("1着", f"bf1_{race}_{i}")
    f2 = boat_pick("2着", f"bf2_{race}_{i}")
    f3 = boat_pick("3着", f"bf3_{race}_{i}")
    out = ["-".join(map(str, p)) for p in itertools.product(sorted(f1), sorted(f2), sorted(f3)) if len(set(p)) == 3]
    return out, []


def show_set(i, combos, bad):
    if bad:
        st.warning("読み取れなかった入力: " + "、".join(bad) + "　(例: 1-2-3)")
    if not combos:
        st.info("買い目を選ぶと、ここに確率が出ます。")
        return
    ok = [c for c in combos if c in ta.index]
    df = pd.DataFrame({"combo": ok, "prob": [float(ta.loc[c, "prob"]) for c in ok], "rank": [int(ta.loc[c, "rank"]) for c in ok]}).sort_values("prob", ascending=False)
    tot = df.prob.sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("点数", f"{len(df)}点")
    m2.metric("どれかが当たる確率", f"{tot:.2f}%")
    m3.metric("1点あたりの平均確率", f"{tot / len(df):.2f}%")
    has_odds = bool(ODDS)
    heads = [("出目", True), ("確率(%)", False), ("順位", False), ("損益分岐(倍)", False)] + ([("オッズ", False), ("期待値", False)] if has_odds else [])

    def mk_rows(d):
        out = []
        for _, x in d.iterrows():
            row = [(combo_html(x.combo), "l", ""), (f"{x.prob:.2f}", "", shade(x.prob, 10)), (f"{int(x['rank'])}位 / 120", "", ""),
                   (f"{100 / x.prob:.0f}倍" if x.prob > 0 else "—", "", "")]
            if has_odds:
                o = ODDS.get(x.combo)
                if o is None:
                    row += [("—", "", ""), ("—", "", "")]
                else:
                    ev = x.prob / 100 * o
                    mark = "background:rgba(46,160,67,.28);font-weight:700" if ev >= EV_MIN else ""
                    row += [(f"{o:g}", "", ""), (f"{ev:.2f}", "", mark)]
            out.append(row)
        return out

    if has_odds:
        d2 = df[df.combo.isin(ODDS)].copy()
        d2["ev"] = d2.prob / 100 * d2.combo.map(ODDS)
        good = d2[d2.ev >= EV_MIN].sort_values("ev", ascending=False)
        e1, e2, e3 = st.columns(3)
        e1.metric("オッズ入力済み", f"{len(d2)} / {len(df)}点")
        e2.metric(f"期待値{EV_MIN:g}以上の点", f"{len(good)}点")
        e3.metric("入力済み分の平均期待値", f"{d2.ev.mean():.2f}" if len(d2) else "—")
        if len(good):
            st.success("買う価値あり(期待値の高い順): " + "、".join(f"{c}(×{e:.2f})" for c, e in zip(good.combo, good.ev)))
        elif len(d2):
            st.info(f"この買い目に、期待値{EV_MIN:g}以上の出目はありません。見送りも選択肢です。")
    show = df if len(df) <= 30 else df.head(30)
    st.markdown(html_table(heads, mk_rows(show)), unsafe_allow_html=True)
    if len(df) > 30:
        st.caption(f"確率の高い順に、上位30点を表示しています(全{len(df)}点)。")
        with st.expander(f"残りの{len(df) - 30}点も見る"):
            st.markdown(html_table(heads, mk_rows(df.iloc[30:])), unsafe_allow_html=True)


def parse_odds(txt):
    """『1-2-3 25.4』のような行を {出目: オッズ} に。全角・区切り違いも読む"""
    t = unicodedata.normalize("NFKC", txt)
    out = {}
    for a, b, c, o in re.findall(r"([1-6])\s*[-ー=・/]\s*([1-6])\s*[-ー=・/]\s*([1-6])[\s:,=×x]+(\d+(?:\.\d+)?)", t):
        if len({a, b, c}) == 3 and float(o) > 0:
            out[f"{a}-{b}-{c}"] = float(o)
    return out


with st.expander("オッズを入力して、期待値を見る(期待値＝確率×オッズ。1を超えれば、確率上は買う価値あり)"):
    st.caption("1行に1点、「出目 オッズ」の形で入力します(例: 1-2-3 12.5)。買い目に入れた出目だけで構いません。入力すると、下の買い目の表に、オッズと期待値が出ます。")
    odds_txt = st.text_area("オッズ", key=f"od_{race}", height=110, placeholder="1-2-3 12.5\n1-3-2 18.0\n2-1-3 30.2", label_visibility="collapsed")
    EV_MIN = st.number_input("期待値の基準(モデルの誤差を見込んで、1.1〜1.2くらいが目安)", min_value=0.5, max_value=3.0, value=1.1, step=0.05, key="evmin")
    ODDS = parse_odds(odds_txt)
    if odds_txt.strip():
        st.caption(f"読み取れたオッズ: {len(ODDS)}点")

tabs = st.tabs(["買い目1", "買い目2", "買い目3"])
sets = []
for i, tb in enumerate(tabs, 1):
    with tb:
        cm, bd = set_ui(i)
        sets.append(cm)
        show_set(i, cm, bd)
used = [(i + 1, [c for c in cm if c in ta.index]) for i, cm in enumerate(sets) if cm]
if len(used) >= 2:
    allc = sorted({c for _, cm in used for c in cm})
    rows_s = [[(f"買い目{i}", "l", ""), (f"{len(cm)}点", "", ""), (f"{sum(float(ta.loc[c, 'prob']) for c in cm):.2f}", "", "")] for i, cm in used]
    rows_s.append([("<b>全セットの合計(重複は1点と数える)</b>", "l", ""), (f"<b>{len(allc)}点</b>", "", ""), (f"<b>{sum(float(ta.loc[c, 'prob']) for c in allc):.2f}</b>", "", "")])
    st.markdown("**セットごとの点数と、どれかが当たる確率(%)**")
    st.markdown(html_table([("買い目", True), ("点数", False), ("当たる確率(%)", False)], rows_s), unsafe_allow_html=True)
if sets and any(sets):
    st.caption("「損益分岐」は、その出目のオッズが、この倍率より高ければ、確率から見て期待値がプラスになる目安です(モデルの確率が正しい場合)。"
               "「どれかが当たる確率」は、選んだ出目のどれか1つが的中する確率の合計です。")

st.markdown("**各艇が1着のときの、2着の確率(%)**")
a = st.radio("1着の艇", [1, 2, 3, 4, 5, 6], horizontal=True, format_func=lambda x: f"{EMOJI[x - 1]} {x}号艇", key="pie1", label_visibility="collapsed")
s2 = S2[(S2.rid == r.rid) & (S2["first"] == a)]
p1v = s2.p_first.iloc[0] if len(s2) else 0
pc1, pc2 = st.columns([1, 2])
pc1.metric(f"{a}号艇が1着になる確率", f"{p1v}%")
pc2.altair_chart(pie(s2, "second", "2着"), width="stretch")

st.markdown("**展示タイムの「平均からの差」による、確率の変動**")
if st.checkbox("この数字を見る(各艇の展示タイムを、平均より速く/遅くしたときの1着率・3連対率)", key=f"sens_{race}"):
    sens, cur = sensitivity(date_iso, venue, race, merged_before(date_iso, venue))
    which = st.radio("見る確率", ["1着率", "3連対率"], horizontal=True, key=f"sensw_{race}", label_visibility="collapsed")
    j = 0 if which == "1着率" else 1
    heads = [("艇", True)] + ([("いまの差(秒)", False)] if cur else []) + [(("±0.00(平均)" if g == 0 else f"{g:+.2f}"), False) for g in GAPS]
    rows = []
    for k in range(1, 7):
        v0 = sens[(k, 0.0)][j]
        row = [(chip(k), "l", "")]
        if cur:
            row.append((f"{cur[k - 1]:+.3f}", "", "color:#9fb0c6"))
        for g in GAPS:
            v = sens[(k, g)][j]
            row.append((f'{v:.1f}<span style="color:#8b97a8;font-size:.7rem"> ({v - v0:+.1f})</span>', "", "background:rgba(34,211,238,.10)" if g == 0 else shade(v, 100 if j else 60)))
        rows.append(row)
    st.markdown(html_table(heads, rows), unsafe_allow_html=True)
    st.caption("列の「+0.04」は、その艇の展示タイムが、そのレースの平均より0.04秒速い場合です(−は遅い場合)。1艇だけを動かし、ほかの艇は今の値のままです。"
               + ("" if cur else " ※展示タイム未取得のため、全艇が同じタイムの状態を基準にしています。") + " 括弧の中は、平均と同じ場合との差(ポイント)です。")

st.caption("数字はモデルの推定値です。検証でも、確率の高い出目を買った回収率は100%に届いていません。")

