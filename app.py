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
  .hero { background: linear-gradient(135deg,#0a1426 0%,#0f2a4d 55%,#0b5d73 135%); border:1px solid #1c2b44; border-radius:16px; padding:18px 22px; margin-bottom:12px; }
  .hero h1 { color:#fff; font-size:1.7rem; margin:0; letter-spacing:.04em; }
  .hero p { color:#a9bddb; margin:6px 0 0; font-size:.9rem; }
  .tag { display:inline-block; padding:3px 10px; margin:2px 6px 2px 0; border-radius:10px; border-left:3px solid #22d3ee; background:#0d1219; font-size:.85rem; }
  .tag.warn { border-left-color:#fbbf24; color:#fbbf24; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ データ
@st.cache_resource(show_spinner="学習データと特徴量を読み込み中…(初回のみ)")
def load_state():
    E, mu, sd = ft.build_dataset()
    H = pd.read_csv("data/history.csv.gz", dtype={"venue": str, "regno": str, "finish": str, "course": str})
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


def merged_before(date_iso, venue):
    ss = st.session_state
    fetched = ss.get("before", {}).get((date_iso, venue), {})
    manual = ss.get("manual", {}).get((date_iso, venue), {})
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
    st.session_state.setdefault("before", {}).setdefault((date_iso, venue), {}).update(
        {r: v for r, v in got.items() if v})
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
    return ch


def kim_bar(r):
    d = pd.DataFrame([{"決まり手": n, "確率": r[f]} for n, f, _ in KIM])
    return alt.Chart(d).mark_bar().encode(
        y=alt.Y("決まり手:N", sort=[n for n, _, _ in KIM], title=None),
        x=alt.X("確率:Q", title="確率(%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("決まり手:N", scale=alt.Scale(domain=[n for n, _, _ in KIM], range=[c for _, _, c in KIM]), legend=None),
        tooltip=["決まり手", "確率"]).properties(height=170)


def combo_emoji(c):
    return " ".join(EMOJI[int(x) - 1] + x for x in c.split("-"))


# ------------------------------------------------------------------ 本体
today = dt.datetime.now(JST).date()
st.markdown('<div class="hero"><h1>🚤 ボートレース予想</h1><p>平和島・多摩川　指数v4(決まり手 × 艇の強さ × 潮 × モーター)</p></div>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 2])
date = c1.date_input("日付", today)
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

venue = st.radio("場", venues, format_func=lambda v: NAMES[v], horizontal=True) if len(venues) > 1 else venues[0]
if len(venues) == 1:
    st.subheader(NAMES[venue])
races = sorted({k[1] for k in bent if k[0] == venue})

ss = st.session_state
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
    show = pd.DataFrame({
        "R": R.race, "締切": R.deadline, "1号艇1着%": R.p1, "逃げ%": R.p_nige, "差し%": R.p_sashi, "まくり%": R.p_makuri,
        "まくり差し%": R.p_mz, "万舟%": R.mando, "潮": R.tide, "潮の判定": R.tide_judge, "目印": R.n_tags,
        "本命の3連単": R.top_combo.map(combo_emoji), "確率%": R.top_prob})
    st.caption("行をクリックすると、そのレースの詳細ページに移ります。")
    ev = st.dataframe(show, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                      column_config={"1号艇1着%": st.column_config.ProgressColumn("1号艇1着%", min_value=0, max_value=100, format="%d"),
                                     "万舟%": st.column_config.NumberColumn("万舟%", help="25%以上は万舟が出やすい目安")})
    if ev.selection.rows:
        ss["race"] = int(R.iloc[ev.selection.rows[0]].race)
        st.rerun()
    st.stop()

# ---- 詳細ページ ----
race = ss["race"]
r = R[R.race == race].iloc[0]
if st.button("← レース一覧にもどる"):
    ss["race"] = None
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
with st.expander("展示タイムを手入力して、再計算する", expanded=bb.exhibit.isna().any()):
    with st.form(f"manual_{race}"):
        cols = st.columns(6)
        vals = {}
        for i, (_, b) in enumerate(bb.iterrows()):
            vals[int(b.boat)] = cols[i].number_input(f"{EMOJI[int(b.boat) - 1]} {int(b.boat)}号艇", min_value=0.0, max_value=9.99,
                                                    value=0.0 if pd.isna(b.exhibit) else float(b.exhibit), step=0.01, format="%.2f")
        st.caption("0.00 は未入力として扱います。6艇すべて入れると、展示タイムの差を使った予想になります。")
        if st.form_submit_button("この値で再計算"):
            m = ss.setdefault("manual", {}).setdefault((date_iso, venue), {}).setdefault(race, {})
            for k, v in vals.items():
                if v > 0:
                    m[k] = round(v, 2)
                else:
                    m.pop(k, None)
            st.rerun()

# 艇の表
tb = pd.DataFrame({"艇": [EMOJI[int(x) - 1] + f" {int(x)}" for x in bb.boat], "選手": bb.name, "級": bb["class"],
                   "1着率%": bb.win, "3連対率%": bb.top3, "展示タイム": bb.exhibit, "モーター2連率%": bb.motor_2r})
st.dataframe(tb, hide_index=True, width="stretch",
             column_config={"1着率%": st.column_config.ProgressColumn("1着率%", min_value=0, max_value=80, format="%.1f"),
                            "3連対率%": st.column_config.ProgressColumn("3連対率%", min_value=0, max_value=100, format="%.1f"),
                            "展示タイム": st.column_config.NumberColumn("展示タイム", format="%.2f")})

g1, g2 = st.columns(2)
with g1:
    st.markdown("**決まり手の確率(%)**　1号艇の1着 **{}%**　万舟 **{}%**".format(r.p1, "—" if pd.isna(r.mando) else int(r.mando)))
    st.altair_chart(kim_bar(r), width="stretch")
    st.markdown("**目印**(検証で回収率が比較的高かった条件。買いの推奨ではありません)")
    tags = [t for t in str(r.tags).split(" / ") if t]
    st.markdown("".join(f'<span class="tag{" warn" if t.startswith("万舟") else ""}">{t}</span>' for t in tags) or "当てはまる目印はありません",
                unsafe_allow_html=True)
with g2:
    st.markdown("**3連単の確率 上位12通り**")
    tt = T[T.rid == r.rid].sort_values("rank")
    st.dataframe(pd.DataFrame({"順位": tt["rank"], "出目": tt.combo.map(combo_emoji), "確率%": tt.prob}),
                 hide_index=True, width="stretch", height=300,
                 column_config={"確率%": st.column_config.ProgressColumn("確率%", min_value=0, max_value=10, format="%.2f")})

st.markdown("**各艇が1着のときの、2着の確率(%)**")
a = st.radio("1着の艇", [1, 2, 3, 4, 5, 6], horizontal=True, format_func=lambda x: f"{EMOJI[x - 1]} {x}号艇", key="pie1", label_visibility="collapsed")
s2 = S2[(S2.rid == r.rid) & (S2["first"] == a)]
p1v = s2.p_first.iloc[0] if len(s2) else 0
pc1, pc2 = st.columns([1, 2])
pc1.metric(f"{a}号艇が1着になる確率", f"{p1v}%")
pc2.altair_chart(pie(s2, "second", "2着"), width="stretch")

with st.expander("3着の確率を見る(1着・2着の艇を選ぶ)"):
    x1, x2 = st.columns(2)
    f = x1.radio("1着", [1, 2, 3, 4, 5, 6], horizontal=True, key="th1", format_func=lambda x: f"{EMOJI[x - 1]}{x}")
    opts = [k for k in range(1, 7) if k != f]
    s = x2.radio("2着", opts, horizontal=True, key="th2", format_func=lambda x: f"{EMOJI[x - 1]}{x}")
    s3 = S3[(S3.rid == r.rid) & (S3["first"] == f) & (S3.second == s)]
    if len(s3):
        st.caption(f"{f}-{s}と入る確率は {s3.p_pair.iloc[0]}%。そのとき3着は:")
        st.altair_chart(pie(s3, "third", "3着"), width="stretch")

st.caption("数字はモデルの推定値です。検証でも、確率の高い出目を買った回収率は100%に届いていません。")

