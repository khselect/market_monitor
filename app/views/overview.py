"""개요: 9개 패널 한 화면. 각 패널 제목의 ⤢ → 해당 패널 전체화면 페이지(새 탭)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (C, DIVERGING, PLOT_CFG, arrow, base_layout, cal, changes, cls, context, gl, goals_for, header,  # noqa: F401
                    li, news_html, portfolio_controls, portfolio_params, rg, spark, title)

ctx = context()
px, status, fc, ev, REG, BRIEF, WEEKLY = ctx.px, ctx.status, ctx.fc, ctx.ev, ctx.REG, ctx.BRIEF, ctx.WEEKLY
SB, U, asof, mt = ctx.SB, ctx.U, ctx.asof, ctx.version

header(ctx)

# ── ROW 1: 주요지표 · 원자재 · 국채 ────────────────────────────────────────────────
H1, H2, H3 = 296, 300, 384   # 행별 고정 높이 → 패널 정렬, 넘치면 패널 내부 스크롤
r1 = st.columns([5, 3.6, 3.4])
with r1[0]:
    with st.container(border=True, height=H1):
        title("주요지표", "전일 대비 · 60일 추이", slug="indicators")
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
        title("원자재", "선물 근월물", slug="commodities")
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
        title("미 국채 금리", f"10Y−2Y {spread:+.0f}bp", slug="treasury")
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

# ── ROW 2: 산업별 · 예상 · AI 브리핑 ─────────────────────────────────────────────────
r2 = st.columns([3.5, 4.1, 4.4])
with r2[0]:
    with st.container(border=True, height=H2):
        title("산업별", "섹터 ETF 수익률(%) · 색=기간 내 상대강도", slug="sectors")
        sec = U["panels"]["sectors"]["items"]
        periods = ["1D", "1W", "1M", "YTD"]
        tab = pd.DataFrame({it["name"]: changes(px[it["id"]], "price") for it in sec if it["id"] in px}).T[periods]
        tab = tab.sort_values("1M", ascending=True)
        z = tab.values / tab.abs().max().replace(0, 1).values
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
        title("예상 · 다음 거래일 마감 등락률", f"FHS 몬테카를로 10,000회 · {nxt:%m/%d}" if nxt is not None else "", slug="forecast")
        fig = go.Figure()
        for k, t in enumerate(tgt):
            r = live[live.target == t]
            if r.empty:
                continue
            r = r.iloc[-1]
            q, nm = r.q, SB[t].name
            for (lo, hi), op, lab in (((1, 99), 0.18, "98%"), ((5, 95), 0.30, "90%"), ((10, 90), 0.45, "80%"), ((25, 75), 0.75, "50%")):
                fig.add_trace(go.Bar(y=[nm], x=[q[hi - 1] - q[lo - 1]], base=[q[lo - 1]], orientation="h",
                                     marker=dict(color=C["s1"], opacity=op, line=dict(width=0)), width=0.55,
                                     name=f"{lab} 구간", showlegend=(k == 0), legendgroup=lab,
                                     hovertemplate=f"{nm} {lab}: %{{base:+.2f}}% ~ {q[hi - 1]:+.2f}%<extra></extra>"))
            fig.add_trace(go.Scatter(y=[nm], x=[q[49]], mode="markers", showlegend=False,
                                     marker=dict(symbol="line-ns", size=18, line=dict(width=2, color=C["text"])),
                                     hovertemplate=f"{nm} 중앙값 %{{x:+.2f}}%<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=C["muted"], width=1, dash="dot"))
        fig.update_layout(**base_layout(150, barmode="overlay", xaxis=dict(ticksuffix="%", gridcolor=C["grid"]),
                                        yaxis=dict(autorange="reversed", gridcolor=C["card"]),
                                        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10))))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
        rows = ""
        for t in tgt:
            r = live[live.target == t]
            if r.empty:
                continue
            r = r.iloc[-1]
            q = r.q
            rn = rg.NAMES[int(r.regime)] if "regime" in r and r.regime >= 0 else "전체"
            rows += (f'<tr><td>{SB[t].name}</td><td class="{cls(q[49])}">{q[49]:+.2f}%</td>'
                     f'<td>{q[9]:+.1f}~{q[89]:+.1f}</td><td>{r.p_up * 100:.0f}%</td>'
                     f'<td>×{r.event_mult:.2f}</td><td>×{r.calib_scale:.2f}</td><td>{rn}</td></tr>')
        st.markdown(f'<table class="mt"><tr><th></th><th>중앙</th><th>80%구간</th><th>P(↑)</th><th>이벤트</th><th>보정</th><th>잔차풀</th></tr>{rows}</table>',
                    unsafe_allow_html=True)


with r2[2]:
    with st.container(border=True, height=H2):
        b = BRIEF or {}
        meta = b.get("_meta", {})
        title("AI 브리핑", f'{meta.get("asof", "-")} · {"Claude " + str(meta.get("model")) if meta.get("generator") == "claude" else "템플릿(LLM 미사용)"}', slug="briefing")
        tabs = st.tabs(["오늘", "시나리오", "증시 뉴스", "AI 트렌드", "주간"])
        with tabs[0]:
            ew = "".join(f'<div><b>{e["date"][5:]}</b> {e["event"]}{" — " + e["why"] if e.get("why") else ""}</div>'
                         for e in b.get("event_watch", []))
            st.markdown(f'<div class="brief"><div class="h">{b.get("headline", "브리핑 없음 — jobs.run briefing 실행")}</div>'
                        f'{li(b.get("market_brief", []) + b.get("korea_brief", []))}'
                        f'<div class="h" style="margin-top:4px">이벤트</div><div class="ev">{ew or "없음"}</div>'
                        f'<div class="h" style="margin-top:4px">리스크</div>{li(b.get("risks", []))}</div>',
                        unsafe_allow_html=True)
        with tabs[1]:
            probs = {s_["id"]: s_ for s_ in b.get("_scenarios", [])}
            html = '<div class="brief">'
            for sc in sorted(b.get("scenarios", []), key=lambda x: ["base", "up", "down"].index(x["id"])):
                pinfo = probs.get(sc["id"], {})
                p_ = f'{pinfo["prob_next_month"] * 100:.0f}%' if "prob_next_month" in pinfo else "–"
                trig = f'<div class="src">트리거: {" · ".join(sc["triggers"])}</div>' if sc.get("triggers") else ""
                note = f'<div class="src">포트폴리오: {sc["portfolio_note"]}</div>' if sc.get("portfolio_note") else ""
                html += (f'<div class="sc {sc["id"]}"><b>{pinfo.get("label", sc["id"])}</b> '
                         f'<span class="src">다음 달 {p_} (전이행렬)</span><br>{sc["narrative"]}{trig}{note}</div>')
            st.markdown(html + '<div class="src">확률=VIX 국면 전이행렬(코드 계산) · 서술=LLM</div></div>', unsafe_allow_html=True)
        with tabs[2]:
            st.markdown(f'<div class="brief">{news_html(b.get("stock_news", []), meta)}</div>', unsafe_allow_html=True)
        with tabs[3]:
            st.markdown(f'<div class="brief">{news_html(b.get("ai_trends", []), meta, "impact")}</div>', unsafe_allow_html=True)
        with tabs[4]:
            w_ = WEEKLY or {}
            if not w_:
                st.markdown('<div class="brief src">주간 다이제스트는 토요일 07:15 KST 실행 시 생성됩니다 '
                            '(<code>python -m jobs.run weekly</code> 로 수동 생성)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="brief"><div class="h">{w_.get("headline", "")}</div>{li(w_.get("market_brief", []))}'
                            f'<div class="h">증시</div>{news_html(w_.get("stock_news", []), w_.get("_meta"))}'
                            f'<div class="h">AI</div>{news_html(w_.get("ai_trends", []), w_.get("_meta"), "impact")}</div>', unsafe_allow_html=True)

# ── ROW 3: 10년 포트폴리오 · 목표 판정 · 모델 상태 ───────────────────────────────────
S = portfolio_controls(U, st.sidebar)
RES, VERDICT, SWEEP, RM, _ = goals_for(ctx, S)
res = RES.get(S["model"]) or RES["bootstrap"]
years, infl, hair, goal, wt = S["years"], S["infl"], S["hair"], S["goal"], S["weights"]
PFQ = portfolio_params(S)

r3 = st.columns([4.4, 4.2, 3.4])
with r3[0]:
    with st.container(border=True, height=H3):
        h0, h1, n = res["hist_window"]
        title(f"{years}년 후 포트폴리오 (실질가치)",
              f"{gl.MODELS[res['model']]} · 4,000경로 · 표본 {h0:%Y-%m}~{h1:%Y-%m} · 할인 {hair * 100:.1f}%p · ⚙ 사이드바", slug="portfolio", params=PFQ)
        fr = res["final_real"]
        k = [("중앙값", f"{np.median(fr) / 1e8:,.2f}억"), ("5%~95%", f"{np.percentile(fr, 5) / 1e8:,.1f}~{np.percentile(fr, 95) / 1e8:,.1f}억"),
             ("IRR 중앙", f"{np.median(res['irr']) * 100:.1f}%"), ("원금손실", f"{res['p_loss_nominal'] * 100:.1f}%"),
             ("MDD 중앙", f"{np.median(res['mdd']) * 100:.0f}%")]
        st.markdown('<div class="kpis">' + "".join(f'<div class="tile"><div class="n">{a}</div><div class="v">{b_}</div></div>' for a, b_ in k) + "</div>",
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
        fig.update_layout(**base_layout(250, yaxis=dict(ticksuffix="억", gridcolor=C["grid"]), hovermode="x unified"))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)

with r3[1]:
    with st.container(border=True, height=H3):
        title("목표 판정", "3개 모델 × Wilson 95% · 최악 한계로 판정", slug="goals", params=PFQ)
        vc = {"충족": "ok", "경계": "mid", "미충족": "bad"}
        icon = {"충족": "✓", "경계": "△", "미충족": "✕"}
        rows = ""
        for _, g in VERDICT.iterrows():
            vals = [g.get(f"v_{m}") for m in gl.MODELS if f"v_{m}" in g]
            rows += (f'<tr><td>{g.label}</td>' + "".join(f"<td>{v * 100:.1f}%</td>" for v in vals)
                     + f'<td class="verdict {vc[g.verdict]}">{icon[g.verdict]} {g.verdict}</td></tr>')
        st.markdown(f'<table class="mt"><tr><th>목표</th><th>부트</th><th>국면</th><th>위기</th><th>판정</th></tr>{rows}</table>',
                    unsafe_allow_html=True)
        gmap = {g["id"]: g for g in U["goals"]}
        fig = go.Figure()
        for gid, col in (("no_loss", C["s2"]), ("irr7", C["s1"])):
            if gid not in SWEEP:
                continue
            g = gmap[gid]
            fig.add_trace(go.Scatter(x=SWEEP.equity, y=SWEEP[gid] * 100, name=g["label"], mode="lines+markers",
                                     line=dict(color=col, width=2), marker=dict(size=8, line=dict(color=C["card"], width=2)),
                                     hovertemplate="주식 %{x}% → %{y:.1f}%<extra>" + g["label"] + "</extra>"))
            fig.add_hline(y=g["threshold"] * 100, line=dict(color=col, width=1, dash="dot"))
        ok = SWEEP[(SWEEP["no_loss"] < gmap["no_loss"]["threshold"]) & (SWEEP["irr7"] >= gmap["irr7"]["threshold"])] if {"no_loss", "irr7"} <= set(SWEEP) else SWEEP.iloc[:0]
        if len(ok):
            fig.add_vrect(x0=ok.equity.min() - 2, x1=ok.equity.max() + 2, fillcolor=C["good"], opacity=0.10, line_width=0)
        cur_eq = sum(v for a, v in wt if a in U["equity_assets"]) * 100
        fig.add_vline(x=cur_eq, line=dict(color=C["text2"], width=1), annotation_text="현재",
                      annotation_font_color=C["text2"], annotation_position="top")
        fig.update_layout(**base_layout(192, xaxis=dict(title=dict(text="주식 비중 % (국면전환 모델)", font=dict(size=10)), gridcolor=C["grid"], dtick=20),
                                        yaxis=dict(ticksuffix="%", gridcolor=C["grid"], range=[-2, 102]),
                                        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10))))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
        feas = f"두 목표 동시 충족 구간: 주식 {ok.equity.min()}~{ok.equity.max()}%" if len(ok) else "두 목표를 동시에 충족하는 주식 비중 없음"
        st.markdown(f'<div class="ev">{feas} (녹색 음영) · 점선=목표 기준</div>', unsafe_allow_html=True)

with r3[2]:
    with st.container(border=True, height=H3):
        done = fc.dropna(subset=["actual_pct"]) if len(fc) else fc
        title("모델 상태", "VIX 국면 전이 · 예측 보정", slug="model")
        P = RM.P
        cells = "".join(
            f'<tr><td>{rg.NAMES[i]}{" ◀" if i == RM.current else ""}</td>'
            + "".join(f'<td style="background:rgba(57,135,229,{P[i, j] * 0.8:.2f})">{P[i, j] * 100:.0f}%</td>' for j in range(3))
            + f"<td>{RM.durations[i]:.1f}개월</td><td>{RM.stationary[i] * 100:.0f}%</td></tr>" for i in range(3))
        st.markdown(f'<table class="mt"><tr><th>현재→다음달</th>' + "".join(f"<th>{n_}</th>" for n_ in rg.NAMES)
                    + f'<th>평균지속</th><th>장기비중</th></tr>{cells}</table>'
                    f'<div class="ev">경계 VIX {RM.thresholds[0]:.0f}/{RM.thresholds[1]:.0f} · 월 {len(RM.monthly)}개 표본</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        nominal = [50, 80, 90, 98]
        fig.add_trace(go.Scatter(x=nominal, y=nominal, mode="lines", line=dict(color=C["muted"], dash="dot", width=1),
                                 name="이상적", hoverinfo="skip"))
        for t, col in zip(U["forecast_targets"], (C["s1"], C["s2"], C["s3"])):
            s_ = cal.summary(done[done.target == t])
            if not s_:
                continue
            fig.add_trace(go.Scatter(x=nominal, y=[s_[f"cov{b_}"] * 100 for b_ in nominal], mode="lines+markers", name=SB[t].name,
                                     line=dict(color=col, width=2), marker=dict(size=8, line=dict(color=C["card"], width=2)),
                                     hovertemplate=SB[t].name + " %{x}% 구간 → 실제 %{y:.1f}%<extra></extra>"))
        fig.update_layout(**base_layout(118, xaxis=dict(gridcolor=C["grid"], tickvals=nominal, ticksuffix="%"),
                                        yaxis=dict(gridcolor=C["grid"], ticksuffix="%"),
                                        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=10))))
        st.plotly_chart(fig, config=PLOT_CFG, use_container_width=True)
        rows = ""
        for t in U["forecast_targets"]:
            s_ = cal.summary(done[done.target == t])
            if s_:
                rows += (f'<tr><td>{SB[t].name}</td><td>{s_["cov90"] * 100:.1f}%</td><td>{s_["crps"]:.3f}</td>'
                         f'<td>{s_["brier"]:.3f}</td><td>{s_["hit_dir"] * 100:.0f}%</td></tr>')
        st.markdown(f'<table class="mt"><tr><th>최근250일</th><th>90%포함</th><th>CRPS</th><th>Brier</th><th>방향</th></tr>{rows}</table>',
                    unsafe_allow_html=True)
