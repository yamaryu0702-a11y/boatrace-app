"""ボートレース予想(Streamlit)。平和島・多摩川。
- 一覧(表紙)でレースを選ぶと詳細ページへ。展示タイムは、ボタンで公式サイトから取り込む/手入力で、その場で再計算する。
- 予想の計算は predict.py と同じもの(指数v4)。成績データは、GitHub Actionsが毎朝更新する。
"""
import datetime as dt
import json
import threading

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
LOCK = threading.Lock()

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

