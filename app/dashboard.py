"""Market Monitor — 한 화면 모니터링 대시보드 (읽기 전용: data/ 캐시만 사용, 외부 호출 없음).

레이아웃 (1920×1080 기준 스크롤 최소화)
┌ 헤더: 기준일 · 데이터 상태 · DEMO 표시 ───────────────────────────────────────────┐
├ [주요지표] 타일 4×2 ─────────┬ [원자재] 표 ─────────────┬ [국채] 수익률곡선 + bp 변화 ┤
├ [산업별] 기간별 히트맵 ──────────────────────┬ [예상] 다음 거래일 예측구간 + 이벤트 ────┤
├ [10년 포트폴리오] 확률 팬차트 + KPI ─────────┬ [모델 신뢰도] 커버리지 · 롤링 · PIT ────┤
└──────────────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm import pipeline  # noqa: E402
from mm.config import series_by_id, universe  # noqa: E402
from mm.data import store  # noqa: E402
from mm.eval import calibration as cal  # noqa: E402
from mm.events import calendar as evcal  # noqa: E402
from mm.sim import portfolio as pf  # noqa: E402

st.set_page_config(page_title="Market Monitor", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

# ── 디자인 토큰 (dark, 검증된 팔레트) ─────────────────────────────────────────────
C = dict(surface="#1a1a19", card="#222220", border="#33332f", text="#f0efec", text2="#c3c2b7",
         muted="#8d8c84", grid="#2c2c29", s1="#3987e5", s2="#d95926", s3="#199e70",
         up="#e66767", down="#3987e5", neutral="#383835", good="#0ca30c", warn="#fab219",
         crit="#d03b3b")
DIVERGING = [[0, "#1c5cab"], [0.25, "#3987e5"], [0.5, C["neutral"]], [0.75, "#e66767"], [1, "#b83a3a"]]

st.markdown(f"""
<style>
  .stApp {{ background:{C['surface']}; color:{C['text']}; }}
  header[data-testid="stHeader"] {{ display:none; }}
  .block-container {{ padding:0.6rem 1.0rem 0.4rem 1.0rem; max-width:100%; }}
  div[data-testid="stVerticalBlock"] {{ gap:0.45rem; }}
  div[data-testid="stVerticalBlockBorderWrapper"] > div {{ padding:0.15rem 0.2rem; }}
  .panel-title {{ font-size:0.82rem; font-weight:700; color:{C['text']}; letter-spacing:.02em;
                 margin:0 0 .25rem 0; display:flex; justify-content:space-between; align-items:baseline; }}
  .panel-title small {{ font-weight:400; color:{C['muted']}; font-size:.70rem; }}
  .tiles {{ display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }}
  .tile {{ background:{C['card']}; border:1px solid {C['border']}; border-radius:6px; padding:5px 8px; }}
  .tile .n {{ font-size:.70rem; color:{C['text2']}; }}
  .tile .v {{ font-size:1.02rem; font-weight:700; font-variant-numeric:tabular-nums; color:{C['text']}; }}
  .tile .d {{ font-size:.72rem; font-variant-numeric:tabular-nums; }}
  .up {{ color:{C['up']}; }} .down {{ color:{C['down']}; }} .flat {{ color:{C['muted']}; }}
  table.mt {{ width:100%; border-collapse:collapse; font-size:.74rem; font-variant-numeric:tabular-nums; }}
  table.mt th {{ color:{C['muted']}; font-weight:500; text-align:right; padding:2px 4px; border-bottom:1px solid {C['border']}; }}
  table.mt td {{ text-align:right; padding:2px 4px; color:{C['text']}; border-bottom:1px solid {C['grid']}; }}
  table.mt td:first-child, table.mt th:first-child {{ text-align:left; color:{C['text2']}; }}
  .badge {{ display:inline-block; padding:1px 7px; border-radius:9px; font-size:.70rem; margin-left:6px;
           border:1px solid {C['border']}; color:{C['text2']}; }}
  .badge.ok {{ border-color:{C['good']}; }} .badge.err {{ border-color:{C['crit']}; }}
  .badge.demo {{ border-color:{C['warn']}; color:{C['warn']}; }}
  .hdr {{ display:flex; align-items:baseline; gap:10px; margin-bottom:2px; }}
  .hdr h1 {{ font-size:1.15rem; margin:0; padding:0; color:{C['text']}; }}
  .hdr .sub {{ color:{C['muted']}; font-size:.75rem; }}
  .kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:6px; margin-bottom:4px; }}
  .ev {{ font-size:.72rem; color:{C['text2']}; line-height:1.45; }}
  .ev b {{ color:{C['text']}; font-weight:600; }}
  .tag {{ display:inline-block; font-size:.66rem; padding:0 5px; border-radius:4px; background:{C['neutral']};
         color:{C['text']}; margin-right:3px; }}
  .tag.t1 {{ background:#5a2a2a; }}
</style>""", unsafe_allow_html=True)

PLOT_CFG = {"displayModeBar": False, "responsive": True}


def base_layout(h: int, **kw) -> dict:
    lay = dict(height=h, margin=dict(l=8, r=8, t=6, b=6), paper_bgcolor=C["card"], plot_bgcolor=C["card"],
               font=dict(color=C["text2"], size=11), hoverlabel=dict(bgcolor="#2c2c29", font_color=C["text"]),
               xaxis=dict(gridcolor=C["grid"], zeroline=False, linecolor=C["border"]),
               yaxis=dict(gridcolor=C["grid"], zeroline=False, linecolor=C["border"]),
               legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    lay.update(kw)
    return lay


def title(text: str, note: str = "") -> None:
    st.markdown(f'<div class="panel-title">{text}<small>{note}</small></div>', unsafe_allow_html=True)


def cls(x: float) -> str:
    return "up" if x > 0 else "down" if x < 0 else "flat"


def arrow(x: float) -> str:
    return "▲" if x > 0 else "▼" if x < 0 else "–"


def spark(v: pd.Series, w: int = 92, h: int = 22) -> str:
    v = v.dropna().tail(60)
    if len(v) < 2:
        return ""
    y = (v - v.min()) / (v.max() - v.min() + 1e-12)
    pts = " ".join(f"{i * w / (len(v) - 1):.1f},{h - 2 - y.iloc[i] * (h - 4):.1f}" for i in range(len(v)))
    col = C["up"] if v.iloc[-1] >= v.iloc[0] else C["down"]
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polyline points="{pts}" fill="none" '
            f'stroke="{col}" stroke-width="1.5" stroke-linejoin="round"/></svg>')


# ── 데이터 로딩 (status.json mtime 을 캐시 키로 → 파이프라인 실행 후 자동 갱신) ─────────────
def status_mtime() -> float:
    return pipeline.STATUS_PATH.stat().st_mtime if pipeline.STATUS_PATH.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_all(_mt: float):
    ids = list(series_by_id())
    px = store.load_panel(ids)
    status = json.loads(pipeline.STATUS_PATH.read_text(encoding="utf-8")) if pipeline.STATUS_PATH.exists() else {}
    return px, status, pipeline.load_forecasts(), evcal.load()


def changes(s: pd.Series, kind: str) -> dict:
    s = s.dropna()
    if len(s) < 2:
        return {}
    last = s.iloc[-1]
    ytd_base = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
    def chg(ref):
        return (last - ref) * 100 if kind == "yield" else (last / ref - 1) * 100
    out = {"last": last}
    for k, n in (("1D", 1), ("1W", 5), ("1M", 21), ("3M", 63)):
        out[k] = chg(s.iloc[-1 - n]) if len(s) > n else np.nan
    out["YTD"] = chg(ytd_base.iloc[-1]) if len(ytd_base) else np.nan
    return out


mt = status_mtime()
if mt == 0:
    st.error("데이터가 없습니다. 먼저 `python -m jobs.run daily` 를 실행하세요.")
    st.stop()
px, status, fc, ev = load_all(mt)
SB = series_by_id()
U = universe()
asof = px.dropna(how="all").index.max()

# ── 헤더 ────────────────────────────────────────────────────────────────────────
n_ok, n_err = status.get("series_ok", 0), status.get("series_err", 0)
badge = f'<span class="badge {"ok" if n_err == 0 else "err"}">{"● 정상" if n_err == 0 else "● 오류"} {n_ok}/{n_ok + n_err}</span>'
demo = '<span class="badge demo">DEMO · 합성 데이터</span>' if status.get("offline") else ""
st.markdown(f'<div class="hdr"><h1>Market Monitor</h1><span class="sub">기준 {asof:%Y-%m-%d} 마감 · '
            f'갱신 {status.get("run_at", "-")[:16].replace("T", " ")} UTC</span>{badge}{demo}</div>',
            unsafe_allow_html=True)

# ── ROW 1: 주요지표 · 원자재 · 국채 ────────────────────────────────────────────────
H1, H2, H3 = 300, 300, 336   # 행별 고정 높이 → 패널 정렬, 넘치면 패널 내부 스크롤
r1 = st.columns([5, 3.6, 3.4])
with r1[0]:
    with st.container(border=True, height=H1):
        title("주요지표", "전일 대비 · 60일 추이")
        html = '<div class="tiles">'
        for it in U["panels"]["indicators"]["items"]:
            s = px.get(it["id"])
            if s is None:
                continue
            c = changes(s, "price")
            if not c:
                continue
            fmt = f"{c['last']:,.2f}" if c["last"] < 100 else f"{c['last']:,.1f}" if c["last"] < 10000 else f"{c['last']:,.0f}"
            html += (f'<div class="tile"><div class="n">{it["name"]}</div><div class="v">{fmt}</div>'
                     f'<div class="d {cls(c["1D"])}">{arrow(c["1D"])} {c["1D"]:+.2f}%</div>{spark(s)}</div>')
        st.markdown(html + "</div>", unsafe_allow_html=True)

with r1[1]:
    with st.container(border=True, height=H1):
        title("원자재", "선물 근월물")
        rows = "".join(
            f'<tr><td>{it["name"]}</td><td>{changes(px[it["id"]], "price")["last"]:,.2f}</td>'
            + "".join(f'<td class="{cls(changes(px[it["id"]], "price")[k])}">{changes(px[it["id"]], "price")[k]:+.1f}%</td>'
                      for k in ("1D", "1M", "YTD"))
            + f'<td>{spark(px[it["id"]], 70, 16)}</td></tr>'
            for it in U["panels"]["commodities"]["items"] if it["id"] in px)
        st.markdown(f'<table class="mt"><tr><th></th><th>가격</th><th>1D</th><th>1M</th><th>YTD</th><th>60D</th></tr>{rows}</table>',
                    unsafe_allow_html=True)

with r1[2]:
    with st.container(border=True, height=H1):
        tr = [it for it in U["panels"]["treasuries"]["items"] if it["id"] in px]
        curve = px[[t["id"] for t in tr]].dropna(how="all").ffill()
        spread = (curve["UST10Y"] - curve["UST2Y"]).iloc[-1] * 100 if {"UST10Y", "UST2Y"} <= set(curve) else np.nan
        title("미 국채 금리", f"10Y−2Y {spread:+.0f}bp")
        mats = [t["maturity"] for t in tr]
        fig = go.Figure()
        for lab, lag, col, dash in (("1년 전", 252, C["s3"], "dot"), ("1개월 전", 21, C["s2"], "dash"), ("현재", 0, C["s1"], "solid")):
            y = curve.iloc[-1 - lag]
            fig.add_trace(go.Scatter(x=mats, y=y.values, name=lab, mode="lines+markers",
                                     line=dict(color=col, width=2, dash=dash), marker=dict(size=8, color=col, line=dict(color=C["card"], width=2)),
                                     hovertemplate="%{x}년: %{y:.2f}%<extra>" + lab + "</extra>"))
        fig.update_layout(**base_layout(112, xaxis=dict(type="log", tickvals=mats, ticktext=[t["name"] for t in tr],
                                                         gridcolor=C["grid"]),
                                        yaxis=dict(ticksuffix="%", gridcolor=C["grid"])))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
        rows = ""
        for t in tr:
            c = changes(px[t["id"]], "yield")
            rows += f'<tr><td>{t["name"]}</td><td>{c["last"]:.2f}%</td>' + "".join(
                f'<td class="{cls(c[k])}">{c[k]:+.0f}</td>' for k in ("1D", "1W", "1M", "YTD")) + "</tr>"
        st.markdown(f'<table class="mt"><tr><th></th><th>금리</th><th>1D bp</th><th>1W</th><th>1M</th><th>YTD</th></tr>{rows}</table>',
                    unsafe_allow_html=True)

# ── ROW 2: 산업별 · 예상 ──────────────────────────────────────────────────────────
r2 = st.columns([5.4, 6.6])
with r2[0]:
    with st.container(border=True, height=H2):
        title("산업별", "SPDR 섹터 ETF 수익률(%) · 색=기간 내 상대강도 · 1M 정렬")
        sec = U["panels"]["sectors"]["items"]
        periods = ["1D", "1W", "1M", "3M", "YTD"]
        tab = pd.DataFrame({it["name"]: changes(px[it["id"]], "price") for it in sec if it["id"] in px}).T[periods]
        tab = tab.sort_values("1M", ascending=True)
        z = tab.values / tab.abs().max().replace(0, 1).values  # 열별 정규화(기간마다 스케일 다름) → 색은 상대강도
        fig = go.Figure(go.Heatmap(z=z, x=periods, y=tab.index, colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
                                   text=tab.map(lambda v: f"{v:+.1f}").values, texttemplate="%{text}",
                                   textfont=dict(size=11, color=C["text"]), customdata=tab.values,
                                   hovertemplate="%{y} · %{x}: %{customdata:+.2f}%<extra></extra>",
                                   showscale=False, xgap=2, ygap=2))
        fig.update_layout(**base_layout(252, xaxis=dict(side="top", gridcolor=C["card"]), yaxis=dict(gridcolor=C["card"])))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)

with r2[1]:
    with st.container(border=True, height=H2):
        tgt = U["forecast_targets"]
        live = fc[fc.actual_pct.isna()] if len(fc) else fc
        nxt = live.target_date.max() if len(live) else None
        title("예상 · 다음 거래일 마감 등락률", f"FHS 몬테카를로 10,000회 · {nxt:%m/%d}" if nxt is not None else "")
        c1, c2 = st.columns([2.7, 2.3])
        with c1:
            fig = go.Figure()
            names = []
            for k, t in enumerate(tgt):
                r = live[live.target == t]
                if r.empty:
                    continue
                r = r.iloc[-1]
                q = r.q
                nm = SB[t].name
                names.append(nm)
                for (lo, hi), op, lab in (((1, 99), 0.18, "98%"), ((5, 95), 0.30, "90%"), ((10, 90), 0.45, "80%"), ((25, 75), 0.75, "50%")):
                    fig.add_trace(go.Bar(y=[nm], x=[q[hi - 1] - q[lo - 1]], base=[q[lo - 1]], orientation="h",
                                         marker=dict(color=C["s1"], opacity=op, line=dict(width=0)), width=0.55,
                                         name=f"{lab} 구간", showlegend=(k == 0), legendgroup=lab,
                                         hovertemplate=f"{nm} {lab}: %{{base:+.2f}}% ~ {q[hi - 1]:+.2f}%<extra></extra>"))
                fig.add_trace(go.Scatter(y=[nm], x=[q[49]], mode="markers",
                                         marker=dict(symbol="line-ns", size=18, line=dict(width=2, color=C["text"])),
                                         showlegend=False, hovertemplate=f"{nm} 중앙값 %{{x:+.2f}}%<extra></extra>"))
            fig.add_vline(x=0, line=dict(color=C["muted"], width=1, dash="dot"))
            fig.update_layout(**base_layout(240, barmode="overlay",
                                            xaxis=dict(ticksuffix="%", gridcolor=C["grid"], zeroline=False),
                                            yaxis=dict(autorange="reversed", gridcolor=C["card"]),
                                            legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=10))))
            st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
        with c2:
            rows = ""
            for t in tgt:
                r = live[live.target == t]
                if r.empty:
                    continue
                r = r.iloc[-1]
                q = r.q
                rows += (f'<tr><td>{SB[t].name}</td><td class="{cls(q[49])}">{q[49]:+.2f}%</td>'
                         f'<td>{q[9]:+.1f}~{q[89]:+.1f}</td><td>{r.p_up * 100:.0f}%</td>'
                         f'<td>×{r.event_mult:.2f}</td><td>×{r.calib_scale:.2f}</td></tr>')
            st.markdown(f'<table class="mt"><tr><th></th><th>중앙</th><th>80%구간</th><th>P(↑)</th><th>이벤트</th><th>보정</th></tr>{rows}</table>',
                        unsafe_allow_html=True)
            today = pd.Timestamp(asof)
            up = ev[(ev.date > today) & (ev.date <= today + pd.Timedelta(days=14)) & (ev.tier <= 2)].head(9)
            items = "".join(
                f'<div><b>{d:%m/%d}</b> <span class="tag {"t1" if tr_ == 1 else ""}">{reg}</span>{nm}{" ?" if stt == "tentative" else ""}</div>'
                for d, nm, tr_, reg, stt in zip(up.date, up.name, up.tier, up.region, up.status))
            st.markdown(f'<div class="panel-title" style="margin-top:6px">예정 이벤트 <small>14일 · tier≤2</small></div>'
                        f'<div class="ev">{items or "없음"}</div>', unsafe_allow_html=True)

# ── ROW 3: 10년 포트폴리오 · 모델 신뢰도 ───────────────────────────────────────────
with st.sidebar:
    st.subheader("포트폴리오 설정")
    cfg = U["default_portfolio"]
    names = {it["id"]: it["name"] for it in U["panels"]["assets"]["items"]}
    w = {a: st.slider(f"{names.get(a, a)} ({a})", 0, 100, int(v * 100), 5) for a, v in cfg["weights"].items()}
    initial = st.number_input("초기 투자금(원)", value=int(cfg["initial"]), step=10_000_000)
    monthly = st.number_input("월 적립(원)", value=int(cfg["monthly_contribution"]), step=100_000)
    years = st.slider("기간(년)", 1, 30, cfg["years"])
    reb = st.selectbox("리밸런싱", ["quarterly", "monthly", "annual", "none"])
    infl = st.slider("인플레이션(연, %)", 0.0, 5.0, cfg["inflation"] * 100, 0.1) / 100
    hair = st.slider("수익률 할인(연, %p) — 과거 낙관편향 보정", 0.0, 4.0, 1.0, 0.5) / 100
    goal = st.number_input("목표 실질자산(원, 0=미사용)", value=500_000_000, step=50_000_000)


@st.cache_data(show_spinner=False)
def run_pf(_mt: float, weights: tuple, initial: float, monthly: float, years: int, reb: str, infl: float, hair: float, goal: float):
    spec = pf.PortfolioSpec(weights=dict(weights), initial=initial, monthly_contribution=monthly, years=years,
                            rebalance=reb, inflation=infl, return_haircut=hair, goal=goal or None, n_paths=4000)
    prices = px[[a for a, x in weights if x > 0]].dropna()
    return pf.simulate(pf.monthly_returns(prices), spec)


wt = tuple((a, v / 100) for a, v in w.items() if v > 0) or (("SPY", 1.0),)
res = run_pf(mt, wt, initial, monthly, years, reb, infl, hair, goal)

r3 = st.columns([6.2, 5.8])
with r3[0]:
    with st.container(border=True, height=H3):
        h0, h1, n = res["hist_window"]
        title(f"{years}년 후 포트폴리오 (실질가치)",
              f"블록 부트스트랩 4,000경로 · 표본 {h0:%Y-%m}~{h1:%Y-%m} · 할인 {hair * 100:.1f}%p · ⚙ 좌측 사이드바에서 조정")
        fr = res["final_real"]
        inv_real = res["invested"][-1] / (1 + infl) ** years
        k = [("중앙값", f"{np.median(fr) / 1e8:,.2f}억"), ("5%~95%", f"{np.percentile(fr, 5) / 1e8:,.1f}~{np.percentile(fr, 95) / 1e8:,.1f}억"),
             ("실질 원금손실", f"{res['p_loss_real'] * 100:.1f}%"),
             ("목표 달성", f"{res.get('p_goal', np.nan) * 100:.0f}%" if goal else "–"),
             ("MDD 중앙값", f"{np.median(res['mdd']) * 100:.0f}%")]
        st.markdown('<div class="kpis">' + "".join(f'<div class="tile"><div class="n">{a}</div><div class="v">{b}</div></div>' for a, b in k) + "</div>",
                    unsafe_allow_html=True)
        b = res["bands_real"]
        x = pd.Timestamp(asof) + pd.to_timedelta(res["months"] * 30.44, unit="D")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=b[95] / 1e8, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=b[5] / 1e8, fill="tonexty", fillcolor="rgba(57,135,229,0.16)", line=dict(width=0),
                                 name="5–95%", hovertemplate="%{x|%Y-%m} 5%: %{y:.2f}억<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=b[75] / 1e8, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=b[25] / 1e8, fill="tonexty", fillcolor="rgba(57,135,229,0.34)", line=dict(width=0),
                                 name="25–75%", hovertemplate="%{x|%Y-%m} 25%: %{y:.2f}억<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=b[50] / 1e8, line=dict(color=C["s1"], width=2), name="중앙값",
                                 hovertemplate="%{x|%Y-%m} 중앙값: %{y:.2f}억<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=res["invested"] / (1 + infl) ** (res["months"] / 12) / 1e8, name="납입원금(실질)",
                                 line=dict(color=C["s2"], width=2, dash="dash"), hovertemplate="%{x|%Y-%m} 원금: %{y:.2f}억<extra></extra>"))
        if goal:
            fig.add_hline(y=goal / 1e8, line=dict(color=C["muted"], width=1, dash="dot"),
                          annotation_text="목표", annotation_font_color=C["text2"], annotation_position="top left")
        fig.update_layout(**base_layout(196, yaxis=dict(ticksuffix="억", gridcolor=C["grid"]), hovermode="x unified"))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)

with r3[1]:
    with st.container(border=True, height=H3):
        done = fc.dropna(subset=["actual_pct"]) if len(fc) else fc
        title("모델 신뢰도 · 예측 vs 마감 실제", f"최근 250거래일 워크포워드 · 이상적 = 명목값과 일치")
        c1, c2 = st.columns([1, 1.25])
        with c1:
            fig = go.Figure()
            nominal = [50, 80, 90, 98]
            fig.add_trace(go.Scatter(x=nominal, y=nominal, mode="lines", line=dict(color=C["muted"], dash="dot", width=1),
                                     name="이상적", hoverinfo="skip"))
            for t, col in zip(U["forecast_targets"], (C["s1"], C["s2"], C["s3"])):
                s = cal.summary(done[done.target == t])
                if not s:
                    continue
                fig.add_trace(go.Scatter(x=nominal, y=[s[f"cov{b}"] * 100 for b in nominal], mode="lines+markers", name=SB[t].name,
                                         line=dict(color=col, width=2), marker=dict(size=8, line=dict(color=C["card"], width=2)),
                                         hovertemplate=SB[t].name + " %{x}% 구간 → 실제 %{y:.1f}%<extra></extra>"))
            fig.update_layout(**base_layout(165, xaxis=dict(title=dict(text="명목 구간 %", font=dict(size=10)), gridcolor=C["grid"], tickvals=nominal),
                                            yaxis=dict(title=dict(text="실제 포함률 %", font=dict(size=10)), gridcolor=C["grid"])))
            st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
        with c2:
            fig = go.Figure()
            for t, col in zip(U["forecast_targets"], (C["s1"], C["s2"], C["s3"])):
                d = done[done.target == t].set_index("actual_date")
                if d.empty:
                    continue
                roll = d["in90"].rolling(60).mean() * 100
                fig.add_trace(go.Scatter(x=roll.index, y=roll, name=SB[t].name, line=dict(color=col, width=2),
                                         hovertemplate=SB[t].name + " %{x|%Y-%m-%d}: %{y:.0f}%<extra></extra>"))
            fig.add_hrect(y0=85, y1=95, fillcolor=C["neutral"], opacity=0.35, line_width=0)
            fig.add_hline(y=90, line=dict(color=C["muted"], width=1, dash="dot"))
            fig.update_layout(**base_layout(165, yaxis=dict(title=dict(text="90%구간 포함률(60일)", font=dict(size=10)), gridcolor=C["grid"], range=[60, 100]),
                                            hovermode="x unified"))
            st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
        rows = ""
        for t in U["forecast_targets"]:
            s = cal.summary(done[done.target == t])
            if s:
                ph = np.array(s["pit_hist"]); flat = ph.std() / ph.mean()
                rows += (f'<tr><td>{SB[t].name}</td><td>{s["cov90"] * 100:.1f}%</td><td>{s["crps"]:.3f}</td>'
                         f'<td>{s["brier"]:.3f}</td><td>{s["hit_dir"] * 100:.1f}%</td><td>{flat:.2f}</td><td>{s["n"]}</td></tr>')
        st.markdown(f'<table class="mt"><tr><th></th><th>90%포함</th><th>CRPS</th><th>Brier</th><th>방향적중</th><th>PIT편차</th><th>n</th></tr>{rows}</table>'
                    f'<div class="ev" style="margin-top:3px">Brier 0.25 = 동전던지기 수준 · 방향적중은 참고용(목표는 구간 보정) · PIT편차 낮을수록 균등</div>',
                    unsafe_allow_html=True)
