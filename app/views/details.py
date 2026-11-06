"""패널별 전체화면 상세 페이지. 개요 패널 제목의 ⤢ 링크 → 새 탭에서 열림.

공통 규칙: 한 차트 = 한 y축(단위가 다르면 소규모 다중 차트), 상승=빨강/하락=파랑 + 부호 병기,
범례는 2개 이상 계열일 때만, 모든 차트에 hover 툴팁.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import (C, DIVERGING, PLOT_CFG, base_layout, br, cal, changes, cls, context, evcal, gl, goals_for,
                    header, li, news_html, pipeline, portfolio_controls, rg, title)

SERIES = [C["s1"], C["s2"], C["s3"], "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]  # 검증된 범주 팔레트(dark)
PERIODS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "전체": None}


def _period_picker(key: str, default: str = "1Y") -> int | None:
    p = st.segmented_control("기간", list(PERIODS), default=default, key=key, label_visibility="collapsed")
    return PERIODS[p or default]


def _tail(s: pd.Series, n: int | None) -> pd.Series:
    s = s.dropna()
    return s if n is None else s.iloc[-n - 1:]


def _small_multiples(series: dict[str, pd.Series], cols: int = 4, height: int = 200, suffix: str = "",
                     ref: dict[str, float] | None = None) -> None:
    names = list(series)
    rows = int(np.ceil(len(names) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, horizontal_spacing=0.05, vertical_spacing=0.14)
    for k, nm in enumerate(names):
        s = series[nm]
        if s.empty:
            continue
        col = C["up"] if s.iloc[-1] >= s.iloc[0] else C["down"]
        r, c = k // cols + 1, k % cols + 1
        fig.add_trace(go.Scatter(x=s.index, y=s.values, line=dict(color=col, width=1.6), showlegend=False,
                                 hovertemplate=f"{nm} %{{x|%Y-%m-%d}}: %{{y:,.2f}}{suffix}<extra></extra>"), r, c)
        if ref and nm in ref:
            fig.add_hline(y=ref[nm], line=dict(color=C["muted"], width=1, dash="dot"), row=r, col=c)
    fig.update_annotations(font=dict(size=12, color=C["text"]))
    fig.update_xaxes(gridcolor=C["grid"], showgrid=False, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=C["grid"], tickfont=dict(size=9), ticksuffix=suffix)
    fig.update_layout(**base_layout(rows * height, margin=dict(l=8, r=8, t=28, b=6)))
    st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)


def _stats_table(ctx, items: list[dict], kind: str) -> None:
    rows = ""
    for it in items:
        s = ctx.px[it["id"]].dropna()
        if s.empty:
            continue
        c = changes(s, kind)
        y1 = ((s.iloc[-1] - s.iloc[-253]) * 100 if kind == "yield" else (s.iloc[-1] / s.iloc[-253] - 1) * 100) if len(s) > 253 else np.nan
        w52 = s.iloc[-252:]
        vol = np.log(s).diff().iloc[-20:].std() * np.sqrt(252) * 100 if kind == "price" else s.diff().iloc[-20:].std() * np.sqrt(252) * 100
        unit = "bp" if kind == "yield" else "%"
        fmt = (lambda v: f"{v:+.0f}") if kind == "yield" else (lambda v: f"{v:+.2f}%")
        rows += (f'<tr><td>{it["name"]}</td><td>{c["last"]:,.2f}</td>'
                 + "".join(f'<td class="{cls(c[k])}">{fmt(c[k])}</td>' for k in ("1D", "1W", "1M", "3M", "YTD"))
                 + f'<td class="{cls(y1)}">{fmt(y1)}</td><td>{w52.max():,.2f}</td><td>{w52.min():,.2f}</td>'
                 f'<td>{(s.iloc[-1] / w52.max() - 1) * 100:+.1f}%</td><td>{vol:.1f}{unit if kind == "yield" else "%"}</td></tr>')
    st.markdown(f'<table class="mt"><tr><th></th><th>현재</th><th>1D</th><th>1W</th><th>1M</th><th>3M</th><th>YTD</th><th>1Y</th>'
                f'<th>52주 고</th><th>52주 저</th><th>고점대비</th><th>20일 변동성(연)</th></tr>{rows}</table>',
                unsafe_allow_html=True)


# ── [주요지표] / [원자재] ─────────────────────────────────────────────────────────
def _price_panel(panel: str, page_title: str) -> None:
    ctx = context()
    header(ctx, page_title)
    items = [it for it in ctx.U["panels"][panel]["items"] if it["id"] in ctx.px]
    n = _period_picker(f"{panel}_p")
    title(page_title, "각 차트는 자체 축(단위 상이) · 색=기간 등락 방향")
    _small_multiples({it["name"]: _tail(ctx.px[it["id"]], n) for it in items}, cols=4, height=210)
    title("기간별 변화 · 위험 지표")
    _stats_table(ctx, items, "price")


def page_indicators():
    _price_panel("indicators", "주요지표")


def page_commodities():
    _price_panel("commodities", "원자재")


# ── [국채] ───────────────────────────────────────────────────────────────────────
def page_treasury():
    ctx = context()
    header(ctx, "미 국채 금리")
    tr = [it for it in ctx.U["panels"]["treasuries"]["items"] if it["id"] in ctx.px]
    curve = ctx.px[[t["id"] for t in tr]].dropna(how="all").ffill()
    mats, labels = [t["maturity"] for t in tr], [t["name"] for t in tr]
    c1, c2 = st.columns([1, 1.3])
    with c1:
        title("수익률 곡선 비교", "현재 vs 과거 시점")
        fig = go.Figure()
        for (lab, lag), col in zip((("1년 전", 252), ("3개월 전", 63), ("1개월 전", 21), ("1주 전", 5), ("현재", 0)), SERIES[::-1][3:] + [C["s1"]]):
            if len(curve) <= lag:
                continue
            y = curve.iloc[-1 - lag]
            fig.add_trace(go.Scatter(x=mats, y=y.values, name=lab, mode="lines+markers",
                                     line=dict(color=col, width=3 if lag == 0 else 1.6, dash="solid" if lag == 0 else "dot"),
                                     marker=dict(size=8, line=dict(color=C["card"], width=2)),
                                     hovertemplate="%{x}년: %{y:.2f}%<extra>" + lab + "</extra>"))
        fig.update_layout(**base_layout(400, xaxis=dict(type="log", tickvals=mats, ticktext=labels, gridcolor=C["grid"]),
                                        yaxis=dict(ticksuffix="%", gridcolor=C["grid"])))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
    with c2:
        n = _period_picker("tr_p", "3Y")
        title("만기별 금리 추이")
        fig = go.Figure()
        for t, col in zip(tr, SERIES):
            s = _tail(ctx.px[t["id"]], n)
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=t["name"], line=dict(color=col, width=1.8),
                                     hovertemplate=t["name"] + " %{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>"))
        fig.update_layout(**base_layout(360, yaxis=dict(ticksuffix="%", gridcolor=C["grid"]), hovermode="x unified"))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
    title("장단기 스프레드", "0 아래 = 역전 (경기침체 선행 신호로 자주 인용)")
    spreads = {}
    for a, b, nm in (("UST10Y", "UST2Y", "10Y−2Y"), ("UST10Y", "UST1Y", "10Y−1Y"), ("UST30Y", "UST5Y", "30Y−5Y")):
        if a in curve and b in curve:
            spreads[nm] = _tail((curve[a] - curve[b]) * 100, n)
    _small_multiples(spreads, cols=3, height=230, suffix="bp", ref={k: 0 for k in spreads})
    title("만기별 변화 (bp)")
    _stats_table(ctx, tr, "yield")


# ── [산업별] ─────────────────────────────────────────────────────────────────────
def page_sectors():
    ctx = context()
    header(ctx, "산업별")
    sec = [it for it in ctx.U["panels"]["sectors"]["items"] if it["id"] in ctx.px]
    periods = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
    tab = {}
    for it in sec:
        s = ctx.px[it["id"]].dropna()
        row = {k: (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100 if len(s) > n else np.nan for k, n in periods.items()}
        row["YTD"] = changes(s, "price")["YTD"]
        tab[it["name"]] = row
    tab = pd.DataFrame(tab).T.sort_values("1M")
    c1, c2 = st.columns([1, 1.25])
    with c1:
        title("기간별 수익률(%)", "색=열 내 상대강도 · 1M 정렬")
        z = tab.values / np.nanmax(np.abs(tab.values), axis=0)
        fig = go.Figure(go.Heatmap(z=z, x=list(tab.columns), y=tab.index, colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
                                   text=tab.map(lambda v: f"{v:+.1f}").values, texttemplate="%{text}",
                                   textfont=dict(size=12, color=C["text"]), customdata=tab.values,
                                   hovertemplate="%{y} · %{x}: %{customdata:+.2f}%<extra></extra>",
                                   showscale=False, xgap=2, ygap=2))
        fig.update_layout(**base_layout(520, xaxis=dict(side="top", gridcolor=C["card"]), yaxis=dict(gridcolor=C["card"])))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
    with c2:
        n = _period_picker("sec_p", "1Y")
        title("S&P 500 대비 상대강도", "섹터/S&P500, 기간 시작=100 · 100 위=시장 대비 강세")
        spx = ctx.px["SPX"].dropna()
        rel = {}
        for it in sec:
            s = ctx.px[it["id"]].dropna()
            r = (s / spx.reindex(s.index).ffill()).dropna()
            r = _tail(r, n)
            rel[it["name"]] = r / r.iloc[0] * 100
        _small_multiples(rel, cols=4, height=150, ref={k: 100 for k in rel})
    title("섹터 지표")
    _stats_table(ctx, sec, "price")


# ── [예상] ───────────────────────────────────────────────────────────────────────
def page_forecast():
    ctx = context()
    header(ctx, "예상 · 다음 거래일 예측")
    fc = ctx.fc
    tgt = ctx.U["forecast_targets"]
    tabs = st.tabs([ctx.SB[t].name for t in tgt])
    for t, tab in zip(tgt, tabs):
        with tab:
            g = fc[fc.target == t].sort_values("base_date") if len(fc) else fc
            if g.empty:
                st.info("예측 이력 없음")
                continue
            live = g[g.actual_pct.isna()]
            done = g.dropna(subset=["actual_pct"])
            c1, c2 = st.columns([1, 1.6])
            with c1:
                if len(live):
                    r = live.iloc[-1]
                    q = r.q
                    title(f"{pd.Timestamp(r.target_date):%Y-%m-%d} 예측분포", f"σ연율 {r.sigma * np.sqrt(252) * 100:.0f}% · 이벤트 ×{r.event_mult:.2f} · 보정 ×{r.calib_scale:.2f}")
                    edges = q[[0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 98]]  # 1,5,…,95,99%
                    probs = np.diff([0.01, 0.05] + [k / 100 for k in range(10, 100, 5)] + [0.99])
                    mids, dens = (edges[1:] + edges[:-1]) / 2, probs / np.maximum(np.diff(edges), 1e-9)
                    q_w = np.diff(edges)
                    fig = go.Figure(go.Bar(x=mids, y=dens, width=q_w, marker=dict(color=[C["up"] if m > 0 else C["down"] for m in mids], line=dict(width=0)),
                                           hovertemplate="%{x:+.2f}% 부근 밀도 %{y:.2f}<extra></extra>"))
                    for p_, lab in ((5, "5%"), (50, "중앙"), (95, "95%")):
                        fig.add_vline(x=q[p_ - 1], line=dict(color=C["text2"], width=1, dash="dot"),
                                      annotation_text=f"{lab} {q[p_ - 1]:+.2f}%", annotation_font_color=C["text2"])
                    fig.update_layout(**base_layout(330, xaxis=dict(ticksuffix="%", gridcolor=C["grid"]),
                                                    yaxis=dict(gridcolor=C["grid"]),
                                                    bargap=0))
                    st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
                    st.markdown(f'<table class="mt"><tr><th>P(상승)</th><th>중앙</th><th>50%</th><th>80%</th><th>98%</th><th>이벤트</th></tr>'
                                f'<tr><td>{r.p_up * 100:.0f}%</td><td class="{cls(q[49])}">{q[49]:+.2f}%</td><td>{q[24]:+.2f}~{q[74]:+.2f}</td>'
                                f'<td>{q[9]:+.2f}~{q[89]:+.2f}</td><td>{q[0]:+.2f}~{q[98]:+.2f}</td><td>{r.events or "-"}</td></tr></table>',
                                unsafe_allow_html=True)
            with c2:
                k = st.slider("최근 거래일", 30, 250, 120, 10, key=f"fc_n_{t}")
                d = done.tail(k)
                title("예측구간 vs 실제 마감 등락", "밴드=80%/98% 예측구간 · ●=구간 안 · ✕=98% 밖")
                fig = go.Figure()
                x = pd.to_datetime(d.actual_date)
                q = np.vstack(d.q.values)
                fig.add_trace(go.Scatter(x=x, y=q[:, 98], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=x, y=q[:, 0], fill="tonexty", fillcolor="rgba(57,135,229,0.15)", line=dict(width=0), name="98% 구간",
                                         hovertemplate="%{x|%m-%d} 98%% 하단 %{y:+.2f}%<extra></extra>"))
                fig.add_trace(go.Scatter(x=x, y=q[:, 89], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=x, y=q[:, 9], fill="tonexty", fillcolor="rgba(57,135,229,0.32)", line=dict(width=0), name="80% 구간",
                                         hovertemplate="%{x|%m-%d} 80%% 하단 %{y:+.2f}%<extra></extra>"))
                inside = (d.actual_pct >= q[:, 0]) & (d.actual_pct <= q[:, 98])
                fig.add_trace(go.Scatter(x=x[inside.values], y=d.actual_pct[inside], mode="markers", name="실제(98% 안)",
                                         marker=dict(size=6, color=C["text"], line=dict(color=C["card"], width=1)),
                                         hovertemplate="%{x|%Y-%m-%d} 실제 %{y:+.2f}%<extra></extra>"))
                fig.add_trace(go.Scatter(x=x[~inside.values], y=d.actual_pct[~inside], mode="markers", name="실제(98% 밖)",
                                         marker=dict(size=10, symbol="x", color=C["warn"]),
                                         hovertemplate="%{x|%Y-%m-%d} 실제 %{y:+.2f}% (구간 밖)<extra></extra>"))
                fig.update_layout(**base_layout(400, yaxis=dict(ticksuffix="%", gridcolor=C["grid"]), hovermode="closest"))
                st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
                s_ = cal.summary(d)
                if s_:
                    st.markdown(f'<div class="ev">최근 {s_["n"]}일 · 50/80/90/98% 포함률 {s_["cov50"] * 100:.0f}/{s_["cov80"] * 100:.0f}/'
                                f'{s_["cov90"] * 100:.0f}/{s_["cov98"] * 100:.0f}% · CRPS {s_["crps"]:.3f} · 방향적중 {s_["hit_dir"] * 100:.0f}%</div>',
                                unsafe_allow_html=True)
    title("이벤트별 변동성 배수", "이벤트일 |수익률| / 직전 20일 평균, 사전값과 축소추정 · 다음 14일 일정")
    from mm.data import store
    from mm.sim import daily
    c1, c2 = st.columns([1, 1])
    with c1:
        r_ = daily.log_returns(store.load("SPX")["value"].dropna())
        m = evcal.estimate_multipliers(r_, ctx.ev, "US")
        st.markdown('<table class="mt"><tr><th>이벤트</th><th>표본 n</th><th>실측 m̂</th><th>사전값</th><th>적용 배수</th></tr>'
                    + "".join(f'<tr><td>{k}</td><td>{int(v.n)}</td><td>{"" if np.isnan(v.m_hat) else f"{v.m_hat:.2f}"}</td>'
                              f'<td>{v.prior:.2f}</td><td>×{v.multiplier:.2f}</td></tr>' for k, v in m.iterrows())
                    + "</table>", unsafe_allow_html=True)
    with c2:
        up = ctx.ev[(ctx.ev.date > ctx.asof) & (ctx.ev.date <= ctx.asof + pd.Timedelta(days=14))]
        st.markdown('<table class="mt"><tr><th>날짜</th><th>이벤트</th><th>지역</th><th>tier</th><th>상태</th></tr>'
                    + "".join(f"<tr><td>{d_:%m-%d}</td><td>{n_}</td><td>{r__}</td><td>{t_}</td><td>{s__}</td></tr>"
                              for d_, n_, r__, t_, s__ in zip(up.date, up.name, up.region, up.tier, up.status))
                    + "</table>", unsafe_allow_html=True)


# ── [AI 브리핑] ──────────────────────────────────────────────────────────────────
def page_briefing():
    ctx = context()
    header(ctx, "AI 브리핑")
    files = sorted(br.BRIEF_DIR.glob("daily-*.json"), reverse=True) + sorted(br.BRIEF_DIR.glob("weekly-*.json"), reverse=True)
    opts = ["최신 일일", "최신 주간"] + [f.stem for f in files]
    pick = st.selectbox("브리핑 선택", opts, label_visibility="collapsed")
    if pick == "최신 일일":
        b = ctx.BRIEF or {}
    elif pick == "최신 주간":
        b = ctx.WEEKLY or {}
    else:
        b = json.loads((br.BRIEF_DIR / f"{pick}.json").read_text(encoding="utf-8"))
    if not b:
        st.info("브리핑 없음 — `python -m jobs.run briefing` / `weekly`")
        return
    meta = b.get("_meta", {})
    gen = f'Claude {meta.get("model")}' if meta.get("generator") == "claude" else "템플릿(LLM 미사용)"
    st.markdown(f'<div class="bigtitle">{b.get("headline", "")}</div><div class="ev">{meta.get("asof")} · {gen} · '
                f'생성 {meta.get("generated_at", "")[:16]} UTC</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        with st.container(border=True):
            title("시장 요약")
            st.markdown(f'<div class="brief">{li(b.get("market_brief", []))}<div class="h">한국</div>{li(b.get("korea_brief", []))}'
                        f'<div class="h">리스크</div>{li(b.get("risks", []))}</div>', unsafe_allow_html=True)
        with st.container(border=True):
            title("이벤트 관전 포인트")
            st.markdown('<div class="brief">' + "".join(
                f'<div><b>{e["date"]}</b> {e["event"]}{"<br><span class=src>" + e["why"] + "</span>" if e.get("why") else ""}</div>'
                for e in b.get("event_watch", [])) + "</div>", unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            title("시나리오", "확률=VIX 국면 전이행렬(코드) · 서술=LLM")
            probs = {s_["id"]: s_ for s_ in b.get("_scenarios", [])}
            html = '<div class="brief">'
            for sc in sorted(b.get("scenarios", []), key=lambda x: ["base", "up", "down"].index(x["id"])):
                pinfo = probs.get(sc["id"], {})
                p_ = f'{pinfo["prob_next_month"] * 100:.0f}%' if "prob_next_month" in pinfo else "–"
                html += (f'<div class="sc {sc["id"]}"><b>{pinfo.get("label", sc["id"])}</b> <span class="src">다음 달 {p_}</span><br>{sc["narrative"]}'
                         + (f'<div class="src">트리거: {" · ".join(sc["triggers"])}</div>' if sc.get("triggers") else "")
                         + (f'<div class="src">포트폴리오: {sc["portfolio_note"]}</div>' if sc.get("portfolio_note") else "") + "</div>")
            st.markdown(html + "</div>", unsafe_allow_html=True)
        with st.container(border=True):
            title("증시 뉴스")
            st.markdown(f'<div class="brief">{news_html(b.get("stock_news", []), meta)}</div>', unsafe_allow_html=True)
    with c3:
        with st.container(border=True):
            title("AI 트렌드")
            st.markdown(f'<div class="brief">{news_html(b.get("ai_trends", []), meta, "impact")}</div>', unsafe_allow_html=True)
        with st.container(border=True):
            title("출처 · 진단")
            src = meta.get("sources", [])
            st.markdown('<div class="brief"><ul>' + "".join(f'<li><a href="{s_["url"]}" target="_blank">{s_["title"] or s_["url"]}</a></li>'
                                                          for s_ in src[:25]) + "</ul></div>" if src else '<span class="src">웹 검색 출처 없음</span>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="ev">{br.diagnose()}</div>', unsafe_allow_html=True)


# ── [10년 포트폴리오] / [목표 판정] ─────────────────────────────────────────────────
def _hist(values, goal_line=None, suffix="", scale=1.0, bins=60, color=None, goal_label=""):
    v = np.asarray(values) * scale
    fig = go.Figure(go.Histogram(x=v, nbinsx=bins, marker=dict(color=color or C["s1"], line=dict(color=C["card"], width=1)),
                                 hovertemplate="%{x}" + suffix + ": %{y}경로<extra></extra>"))
    if goal_line is not None:
        fig.add_vline(x=goal_line * scale, line=dict(color=C["text2"], width=1, dash="dot"), annotation_text=goal_label,
                      annotation_font_color=C["text2"])
    for p_ in (5, 50, 95):
        fig.add_vline(x=np.percentile(v, p_), line=dict(color=C["muted"], width=1),
                      annotation_text=f"p{p_}", annotation_position="bottom right", annotation_font_color=C["muted"])
    fig.update_layout(**base_layout(260, xaxis=dict(ticksuffix=suffix, gridcolor=C["grid"]), yaxis=dict(gridcolor=C["grid"]), bargap=0.02))
    return fig


def page_portfolio():
    ctx = context()
    header(ctx, "10년 후 포트폴리오")
    left, right = st.columns([1, 3.4])
    S = portfolio_controls(ctx.U, left)
    RES, VERDICT, SWEEP, RM, rets = goals_for(ctx, S, n_paths=6000)
    res = RES.get(S["model"]) or RES["bootstrap"]
    with right:
        title(f"{S['years']}년 후 실질가치 경로", f"{gl.MODELS[res['model']]} · 6,000경로 · 할인 {S['hair'] * 100:.1f}%p")
        b = res["bands_real"]
        x = pd.Timestamp(ctx.asof) + pd.to_timedelta(res["months"] * 30.44, unit="D")
        fig = go.Figure()
        for lo, hi, op, nm in ((5, 95, 0.16, "5–95%"), (25, 75, 0.34, "25–75%")):
            fig.add_trace(go.Scatter(x=x, y=b[hi] / 1e8, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=b[lo] / 1e8, fill="tonexty", fillcolor=f"rgba(57,135,229,{op})", line=dict(width=0), name=nm,
                                     hovertemplate="%{x|%Y-%m} " + nm + " 하단 %{y:.2f}억<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=b[50] / 1e8, line=dict(color=C["s1"], width=2.5), name="중앙값", hovertemplate="%{x|%Y-%m} 중앙 %{y:.2f}억<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=res["invested"] / (1 + S["infl"]) ** (res["months"] / 12) / 1e8, name="납입원금(실질)",
                                 line=dict(color=C["s2"], width=2, dash="dash"), hovertemplate="%{x|%Y-%m} 원금 %{y:.2f}억<extra></extra>"))
        if S["goal"]:
            fig.add_hline(y=S["goal"] / 1e8, line=dict(color=C["muted"], dash="dot"), annotation_text="목표", annotation_font_color=C["text2"])
        fig.update_layout(**base_layout(420, yaxis=dict(ticksuffix="억", gridcolor=C["grid"]), hovermode="x unified"))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            title("최종 실질자산 분포")
            st.plotly_chart(_hist(res["final_real"], S["goal"] or None, "억", 1e-8, goal_label="목표"), config=PLOT_CFG, use_container_width=True)
        with c2:
            title("연평균 수익률(IRR) 분포", "적립식 투자자 기준")
            st.plotly_chart(_hist(res["irr"], 0.07, "%", 100, color=C["s3"], goal_label="7%"), config=PLOT_CFG, use_container_width=True)
        with c3:
            title("최대낙폭(MDD) 분포")
            st.plotly_chart(_hist(res["mdd"], -0.30, "%", 100, color=C["s2"], goal_label="−30%"), config=PLOT_CFG, use_container_width=True)
        title("모델 비교", "같은 설정 · 모델만 다름 → 차이 = 모델 불확실성")
        rows = ""
        for k, r in RES.items():
            rows += (f'<tr><td>{gl.MODELS[k]}</td><td>{np.median(r["final_real"]) / 1e8:.2f}억</td>'
                     f'<td>{np.percentile(r["final_real"], 5) / 1e8:.2f}억</td><td>{np.percentile(r["final_real"], 95) / 1e8:.2f}억</td>'
                     f'<td>{np.median(r["irr"]) * 100:.1f}%</td><td>{np.percentile(r["irr"], 5) * 100:.1f}%</td>'
                     f'<td>{r["p_loss_nominal"] * 100:.1f}%</td><td>{r["p_loss_real"] * 100:.1f}%</td><td>{np.median(r["mdd"]) * 100:.0f}%</td></tr>')
        st.markdown('<table class="mt"><tr><th>모델</th><th>실질 중앙</th><th>p5</th><th>p95</th><th>IRR 중앙</th><th>IRR p5</th>'
                    f'<th>원금손실</th><th>실질손실</th><th>MDD 중앙</th></tr>{rows}</table>'
                    f'<div class="ev">표본 {rets.index[0]:%Y-%m}~{rets.index[-1]:%Y-%m} ({len(rets)}개월) · 국면 장기비중 '
                    + " / ".join(f"{n_} {p_ * 100:.0f}%" for n_, p_ in zip(rg.NAMES, RM.stationary)) + "</div>", unsafe_allow_html=True)


def page_goals():
    ctx = context()
    header(ctx, "목표 판정")
    left, right = st.columns([1, 3.4])
    S = portfolio_controls(ctx.U, left)
    RES, VERDICT, SWEEP, RM, _ = goals_for(ctx, S, n_paths=6000)
    with right:
        title("판정표", "충족=3개 모델의 Wilson 95% 불리한 한계까지 기준 만족 · 경계=일부 모델만 만족")
        vc, icon = {"충족": "ok", "경계": "mid", "미충족": "bad"}, {"충족": "✓", "경계": "△", "미충족": "✕"}
        rows = ""
        for _, g in VERDICT.iterrows():
            op = "<" if g.op == "<" else "≥"
            rows += (f'<tr><td>{g.label}</td><td>{op} {g.threshold * 100:.0f}%</td>'
                     + "".join(f'<td>{g[f"v_{m}"] * 100:.1f}%</td>' for m in gl.MODELS if f"v_{m}" in g)
                     + f'<td>{g.worst_bound * 100:.1f}%</td><td class="verdict {vc[g.verdict]}">{icon[g.verdict]} {g.verdict}</td></tr>')
        st.markdown('<table class="mt"><tr><th>목표</th><th>기준</th>' + "".join(f"<th>{v}</th>" for v in gl.MODELS.values())
                    + f'<th>최악 한계</th><th>판정</th></tr>{rows}</table>', unsafe_allow_html=True)
        title("주식 비중별 목표 확률", "주식 바스켓(현 배분 내 비율 유지) vs 채권·금 바스켓 · 국면전환 모델 · 점선=기준 · 녹색=충족")
        goals = ctx.U["goals"]
        fig = make_subplots(rows=2, cols=2, subplot_titles=[g["label"] for g in goals], vertical_spacing=0.16, horizontal_spacing=0.08)
        cur_eq = sum(v for a, v in S["weights"] if a in ctx.U["equity_assets"]) * 100
        for k, g in enumerate(goals[:4]):
            r, c = k // 2 + 1, k % 2 + 1
            y = SWEEP[g["id"]] * 100
            ok = (SWEEP[g["id"]] < g["threshold"]) if g["op"] == "<" else (SWEEP[g["id"]] >= g["threshold"])
            fig.add_trace(go.Scatter(x=SWEEP.equity, y=y, mode="lines+markers", line=dict(color=C["s1"], width=2), showlegend=False,
                                     marker=dict(size=9, color=[C["good"] if o else C["muted"] for o in ok], line=dict(color=C["card"], width=2)),
                                     hovertemplate="주식 %{x}% → %{y:.1f}%<extra>" + g["label"] + "</extra>"), r, c)
            fig.add_hline(y=g["threshold"] * 100, line=dict(color=C["text2"], width=1, dash="dot"), row=r, col=c)
            fig.add_vline(x=cur_eq, line=dict(color=C["muted"], width=1), row=r, col=c)
        fig.update_annotations(font=dict(size=12, color=C["text"]))
        fig.update_xaxes(gridcolor=C["grid"], ticksuffix="%", dtick=20)
        fig.update_yaxes(gridcolor=C["grid"], ticksuffix="%")
        fig.update_layout(**base_layout(560, margin=dict(l=8, r=8, t=30, b=6)))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
        allok = np.all([(SWEEP[g["id"]] < g["threshold"]) if g["op"] == "<" else (SWEEP[g["id"]] >= g["threshold"]) for g in goals], axis=0)
        feas = SWEEP.equity[allok]
        st.markdown(f'<div class="ev">모든 목표 동시 충족 주식 비중: {f"{feas.min()}~{feas.max()}%" if len(feas) else "없음"} · '
                    f'현재 {cur_eq:.0f}% (세로선) · 목표 문장은 config/universe.yaml 의 goals 에서 수정</div>', unsafe_allow_html=True)


# ── [모델 상태] ──────────────────────────────────────────────────────────────────
def page_model():
    ctx = context()
    header(ctx, "모델 상태")
    U = ctx.U
    vix = ctx.px[U["regime"]["vix_id"]].dropna()
    rm = rg.fit(vix, U["regime"]["thresholds"])
    c1, c2 = st.columns([1.6, 1])
    with c1:
        n = _period_picker("vix_p", "5Y")
        title("VIX와 국면", f"월평균 기준 · 경계 {rm.thresholds[0]:.0f}/{rm.thresholds[1]:.0f} · 배경색=국면")
        v = _tail(vix, n)
        mreg = rm.monthly[rm.monthly.index >= v.index[0] - pd.offsets.MonthEnd(1)]
        fig = go.Figure()
        colors = {0: "rgba(25,158,112,0.10)", 1: "rgba(141,140,132,0.06)", 2: "rgba(208,59,59,0.16)"}
        for d_, k in mreg.items():
            fig.add_vrect(x0=d_ - pd.offsets.MonthBegin(1), x1=d_, fillcolor=colors[int(k)], line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=v.index, y=v.values, line=dict(color=C["s1"], width=1.5), name="VIX",
                                 hovertemplate="%{x|%Y-%m-%d} VIX %{y:.1f}<extra></extra>"))
        for th in rm.thresholds:
            fig.add_hline(y=th, line=dict(color=C["muted"], width=1, dash="dot"))
        fig.update_layout(**base_layout(360, yaxis=dict(gridcolor=C["grid"]), showlegend=False))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
    with c2:
        title("전이행렬 (현재→다음 달)", f"현재 {rg.NAMES[rm.current]} · VIX20 {rm.current_vix:.1f}")
        P = rm.P
        fig = go.Figure(go.Heatmap(z=P, x=rg.NAMES, y=rg.NAMES, colorscale=[[0, C["card"]], [1, C["s1"]]], zmin=0, zmax=1,
                                   text=[[f"{p * 100:.0f}%<br>n={int(c_)}" for p, c_ in zip(rp, rc)] for rp, rc in zip(P, rm.counts)],
                                   texttemplate="%{text}", textfont=dict(color=C["text"], size=12), showscale=False, xgap=3, ygap=3,
                                   hovertemplate="%{y} → %{x}: %{z:.1%}<extra></extra>"))
        fig.update_layout(**base_layout(250, yaxis=dict(autorange="reversed", title=dict(text="현재", font=dict(size=10))),
                                        xaxis=dict(side="top", title=dict(text="다음 달", font=dict(size=10)))))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
        st.markdown('<table class="mt"><tr><th>국면</th><th>평균 지속</th><th>장기 비중</th><th>표본(월)</th></tr>'
                    + "".join(f"<tr><td>{rg.NAMES[i]}</td><td>{rm.durations[i]:.1f}개월</td><td>{rm.stationary[i] * 100:.0f}%</td>"
                              f"<td>{int((rm.monthly == i).sum())}</td></tr>" for i in range(3)) + "</table>", unsafe_allow_html=True)
    title("국면별 자산 성과", "월간 수익률 기준 연율화 · 위기 국면에서 채권·금의 방어력 확인")
    from mm.sim import portfolio as pf
    assets = [it["id"] for it in U["panels"]["assets"]["items"] if it["id"] in ctx.px]
    rets = pf.monthly_returns(ctx.px[assets].dropna())
    cs = rg.conditional_stats(rets, rm.monthly)
    hdr = "".join(f"<th>{a} {b}</th>" for a, b in cs.columns)
    st.markdown(f'<table class="mt"><tr><th>자산</th>{hdr}</tr>' + "".join(
        f"<tr><td>{idx}</td>" + "".join(f'<td class="{cls(v) if "수익" in col[1] else ""}">{v:g}</td>' for col, v in row.items()) + "</tr>"
        for idx, row in cs.iterrows()) + "</table>", unsafe_allow_html=True)
    title("예측 보정 상태", "롤링 60일 90% 구간 포함률 (85~95% 음영=정상 범위) · PIT 히스토그램(평평할수록 잘 보정)")
    fc = ctx.fc.dropna(subset=["actual_pct"]) if len(ctx.fc) else ctx.fc
    c1, c2 = st.columns([1.6, 1])
    with c1:
        fig = go.Figure()
        for t, col in zip(U["forecast_targets"], SERIES):
            d = fc[fc.target == t].set_index("actual_date").sort_index()
            if d.empty:
                continue
            roll = d["in90"].rolling(60).mean() * 100
            fig.add_trace(go.Scatter(x=roll.index, y=roll, name=ctx.SB[t].name, line=dict(color=col, width=2),
                                     hovertemplate=ctx.SB[t].name + " %{x|%Y-%m-%d}: %{y:.0f}%<extra></extra>"))
        fig.add_hrect(y0=85, y1=95, fillcolor=C["neutral"], opacity=0.35, line_width=0)
        fig.add_hline(y=90, line=dict(color=C["muted"], width=1, dash="dot"))
        fig.update_layout(**base_layout(300, yaxis=dict(ticksuffix="%", gridcolor=C["grid"], range=[50, 100]), hovermode="x unified"))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
    with c2:
        tg = U["forecast_targets"]
        fig = make_subplots(rows=1, cols=len(tg), subplot_titles=[ctx.SB[t].name for t in tg], horizontal_spacing=0.06)
        for k, t in enumerate(tg):
            s_ = cal.summary(fc[fc.target == t])
            if not s_:
                continue
            ph = np.array(s_["pit_hist"])
            fig.add_trace(go.Bar(x=np.arange(0.05, 1, 0.1), y=ph, marker=dict(color=SERIES[k], line=dict(width=0)), showlegend=False,
                                 hovertemplate="PIT %{x:.2f}: %{y}회<extra></extra>"), 1, k + 1)
            fig.add_hline(y=ph.mean(), line=dict(color=C["muted"], width=1, dash="dot"), row=1, col=k + 1)
        fig.update_annotations(font=dict(size=12, color=C["text"]))
        fig.update_xaxes(gridcolor=C["grid"], range=[0, 1], dtick=0.5)
        fig.update_yaxes(gridcolor=C["grid"])
        fig.update_layout(**base_layout(300, margin=dict(l=8, r=8, t=28, b=6), bargap=0.05))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
    st.markdown(f'<div class="ev">일간 국면조건부 잔차: {"켜짐" if U["regime"].get("daily_conditioning", True) else "꺼짐"} '
                f'(config/universe.yaml › regime.daily_conditioning) · 데이터 상태: {pipeline.STATUS_PATH.name}</div>',
                unsafe_allow_html=True)
