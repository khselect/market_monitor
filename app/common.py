"""공통: 디자인 토큰·CSS·차트 헬퍼·데이터 로딩·헤더·패널 링크 (개요/상세 페이지가 함께 사용)."""
from __future__ import annotations

import json
import sys
import textwrap
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # noqa: F401  (페이지에서 재사용)
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm import pipeline  # noqa: E402
from mm.config import series_by_id, universe  # noqa: E402
from mm.data import store  # noqa: E402
from mm.eval import calibration as cal  # noqa: E402
from mm.events import calendar as evcal  # noqa: E402
from mm.llm import briefing as br  # noqa: E402
from mm.sim import goals as gl  # noqa: E402
from mm.sim import portfolio as pf  # noqa: E402
from mm.sim import regime as rg  # noqa: E402

# ── 디자인 토큰 (dark, 검증된 팔레트) ─────────────────────────────────────────────
C = dict(surface="#1a1a19", card="#222220", border="#33332f", text="#f0efec", text2="#c3c2b7",
         muted="#8d8c84", grid="#2c2c29", s1="#3987e5", s2="#d95926", s3="#199e70",
         up="#e66767", down="#3987e5", neutral="#383835", good="#0ca30c", warn="#fab219",
         crit="#d03b3b")
DIVERGING = [[0, "#1c5cab"], [0.25, "#3987e5"], [0.5, C["neutral"]], [0.75, "#e66767"], [1, "#b83a3a"]]

def inject_css() -> None:
    st.markdown(textwrap.dedent(f"""
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
      .hdr {{ display:flex; align-items:baseline; gap:10px; margin-bottom:6px; flex-wrap:wrap; }}
      .hdr h1 {{ font-size:1.15rem; margin:0; padding:0; color:{C['text']}; }}
      .hdr .sub {{ color:{C['muted']}; font-size:.75rem; }}
      .kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:6px; margin-bottom:4px; }}
      .ev {{ font-size:.72rem; color:{C['text2']}; line-height:1.45; }}
      .ev b {{ color:{C['text']}; font-weight:600; }}
      .tag {{ display:inline-block; font-size:.66rem; padding:0 5px; border-radius:4px; background:{C['neutral']};
             color:{C['text']}; margin-right:3px; }}
      .tag.t1 {{ background:#5a2a2a; }}
      .badge.r0 {{ border-color:{C['s3']}; }} .badge.r1 {{ border-color:{C['muted']}; }} .badge.r2 {{ border-color:{C['crit']}; color:{C['text']}; }}
      .brief {{ font-size:.74rem; color:{C['text']}; line-height:1.5; }}
      .brief ul {{ margin:0 0 0 1rem; padding:0; }} .brief li {{ margin:0 0 2px 0; }}
      .brief .h {{ font-weight:700; font-size:.80rem; margin-bottom:3px; }}
      .brief a {{ color:{C['s1']}; text-decoration:none; }}
      .brief .src {{ color:{C['muted']}; font-size:.68rem; }}
      .sc {{ border-left:3px solid {C['border']}; padding:2px 0 2px 7px; margin-bottom:5px; }}
      .sc.up {{ border-color:{C['s3']}; color:{C['text']}; }} .sc.down {{ border-color:{C['crit']}; color:{C['text']}; }}
      .sc.base {{ border-color:{C['muted']}; }}
      .verdict {{ font-weight:700; }} .verdict.ok {{ color:{C['good']}; }} .verdict.mid {{ color:{C['warn']}; }} .verdict.bad {{ color:{C['crit']}; }}
      div[data-testid="stTabs"] button {{ padding:2px 8px; font-size:.74rem; }}
      div[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap:2px; }}
      .panel-title a.expand {{ color:{C['muted']}; text-decoration:none; font-weight:400; margin-left:6px; font-size:.85rem; }}
      .panel-title a.expand:hover {{ color:{C['s1']}; }}
      .hdr a.home {{ color:{C['text']}; text-decoration:none; }}
      .hdr .crumb {{ color:{C['text2']}; font-weight:500; }}
      .hdr .nav {{ margin-left:auto; font-size:.74rem; }} .hdr .nav a {{ color:{C['muted']}; text-decoration:none; }}
      .hdr .nav a.on, .hdr .nav a:hover {{ color:{C['text']}; }}
      .kpis.k4 {{ grid-template-columns:repeat(4,1fr); }}
      .bigtitle {{ font-size:.95rem; font-weight:700; color:{C['text']}; margin:.3rem 0 .2rem 0; }}
    </style>"""), unsafe_allow_html=True)

PLOT_CFG = {"displayModeBar": False, "responsive": True}


def base_layout(h: int, **kw) -> dict:
    lay = dict(height=h, margin=dict(l=8, r=8, t=6, b=6), paper_bgcolor=C["card"], plot_bgcolor=C["card"],
               font=dict(color=C["text2"], size=11), hoverlabel=dict(bgcolor="#2c2c29", font_color=C["text"]),
               xaxis=dict(gridcolor=C["grid"], zeroline=False, linecolor=C["border"]),
               yaxis=dict(gridcolor=C["grid"], zeroline=False, linecolor=C["border"]),
               legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    lay.update(kw)
    return lay


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
def data_version() -> float:
    """캐시 키: 상태·예측·브리핑·목표 파일 중 가장 최근 수정 시각.
    (주의: st.cache_data 는 밑줄로 시작하는 인자를 해시하지 않음 → 키 인자는 밑줄 없이)"""
    paths = [pipeline.STATUS_PATH, pipeline.FC_PATH, pipeline.GOALS_PATH, pipeline.REGIME_PATH,
             br.BRIEF_DIR / "latest.json", br.BRIEF_DIR / "latest_weekly.json"]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


@st.cache_data(show_spinner=False)
def load_all(version: float):
    ids = list(series_by_id())
    px = store.load_panel(ids)
    status = json.loads(pipeline.STATUS_PATH.read_text(encoding="utf-8")) if pipeline.STATUS_PATH.exists() else {}
    reg = json.loads(pipeline.REGIME_PATH.read_text(encoding="utf-8")) if pipeline.REGIME_PATH.exists() else None
    return px, status, pipeline.load_forecasts(), evcal.load(), reg, br.load_latest(), br.load_latest(weekly=True)


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




# ── 컨텍스트 · 헤더 · 패널 링크 ────────────────────────────────────────────────────
def context() -> SimpleNamespace:
    v = data_version()
    if v == 0:
        st.error("데이터가 없습니다. 먼저 `python -m jobs.run daily` 를 실행하세요.")
        st.stop()
    px, status, fc, ev, reg, brief, weekly = load_all(v)
    return SimpleNamespace(version=v, px=px, status=status, fc=fc, ev=ev, REG=reg, BRIEF=brief, WEEKLY=weekly,
                           SB=series_by_id(), U=universe(), asof=px.dropna(how="all").index.max())


def page_url(slug: str, params: dict | None = None) -> str:
    """서브디렉토리 게시(baseUrlPath)를 반영한 절대 경로 → 새 탭에서 열기."""
    base = (st.get_option("server.baseUrlPath") or "").strip("/")
    path = f"/{base}/{slug}" if base else f"/{slug}"
    return path + ("?" + urllib.parse.urlencode(params) if params else "")


def title(text: str, note: str = "", slug: str | None = None, params: dict | None = None) -> None:
    link = (f' <a class="expand" href="{page_url(slug, params)}" target="_blank" title="전체화면(새 탭)">⤢</a>'
            if slug else "")
    st.markdown(f'<div class="panel-title"><span>{text}{link}</span><small>{note}</small></div>', unsafe_allow_html=True)


NAV = [("indicators", "주요지표"), ("commodities", "원자재"), ("treasury", "미 국채 금리"), ("sectors", "산업별"),
       ("forecast", "예상 · 다음 거래일 예측"), ("briefing", "AI 브리핑"), ("portfolio", "10년 후 포트폴리오"),
       ("goals", "목표 판정"), ("model", "모델 상태")]


def header(ctx, page_title: str | None = None) -> None:
    status, REG, BRIEF = ctx.status, ctx.REG, ctx.BRIEF
    n_ok, n_err = status.get("series_ok", 0), status.get("series_err", 0)
    badge = f'<span class="badge {"ok" if n_err == 0 else "err"}">{"● 정상" if n_err == 0 else "● 오류"} {n_ok}/{n_ok + n_err}</span>'
    demo = '<span class="badge demo">DEMO · 합성 데이터</span>' if status.get("offline") else ""
    regb = (f'<span class="badge r{REG["current"]}">국면 {REG["current_name"]} · VIX {REG["current_vix"]:.1f}</span>' if REG else "")
    gen = (BRIEF or {}).get("_meta", {}).get("generator")
    llmb = f'<span class="badge">브리핑 {"Claude" if gen == "claude" else "템플릿"}</span>' if gen else ""
    home = f'<a class="home" href="{page_url("")}">Market Monitor</a>'
    sub = f' <span class="crumb">› {page_title}</span>' if page_title else ""
    nav = ""
    if page_title:
        nav = '<span class="nav">' + " · ".join(
            f'<a href="{page_url(sl)}"{" class=on" if nm == page_title else ""}>{nm}</a>' for sl, nm in NAV) + "</span>"
    st.markdown(f'<div class="hdr"><h1>{home}{sub}</h1><span class="sub">기준 {ctx.asof:%Y-%m-%d} 마감 · '
                f'갱신 {status.get("run_at", "-")[:16].replace("T", " ")} UTC</span>{badge}{regb}{llmb}{demo}{nav}</div>',
                unsafe_allow_html=True)


def li(items) -> str:
    items = [i for i in items if i]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>" if items else '<span class="src">없음</span>'


def news_html(items, meta: dict | None = None, extra_key: str | None = None) -> str:
    if not items:
        d = (meta or {}).get("diag", {})
        why = (meta or {}).get("error") or d.get("reason") or ""
        errs = [e for r in d.get("research", []) for e in r.get("errors", [])]
        if errs:
            why += f" 웹검색 오류 {sorted(set(errs))}"
        return (f'<span class="src">뉴스 없음{" — " + why if why else ""} · '
                f'진단: <code>python -m jobs.run llmcheck</code></span>')
    out = ""
    for n in items:
        link = f' <a href="{n["url"]}" target="_blank">↗</a>' if str(n.get("url", "")).startswith("http") else ""
        ex = f'<br><span class="src">{n[extra_key]}</span>' if extra_key and n.get(extra_key) else ""
        out += f'<li><b>{n["title"]}</b>{link} — {n.get("summary", "")}{ex}</li>'
    return f"<ul>{out}</ul>"


# ── 포트폴리오 설정 (사이드바/상세 페이지 공용, 쿼리스트링으로 새 탭에 전달) ───────────────────
PF_KEYS = ("initial", "monthly", "years", "reb", "infl", "hair", "goal", "model")


def portfolio_controls(U, container) -> dict:
    cfg = U["default_portfolio"]
    q = st.query_params
    names = {it["id"]: it["name"] for it in U["panels"]["assets"]["items"]}
    with container:
        st.subheader("포트폴리오 설정")
        w = {a: container.slider(f"{names.get(a, a)} ({a})", 0, 100, int(q.get(f"w_{a}", int(v * 100))), 5, key=f"w_{a}")
             for a, v in cfg["weights"].items()}
        s = dict(
            initial=container.number_input("초기 투자금(원)", value=int(q.get("initial", cfg["initial"])), step=10_000_000),
            monthly=container.number_input("월 적립(원)", value=int(q.get("monthly", cfg["monthly_contribution"])), step=100_000),
            years=container.slider("기간(년)", 1, 30, int(q.get("years", cfg["years"]))),
            reb=container.selectbox("리밸런싱", ["quarterly", "monthly", "annual", "none"],
                                    index=["quarterly", "monthly", "annual", "none"].index(q.get("reb", cfg["rebalance"]))),
            infl=container.slider("인플레이션(연, %)", 0.0, 5.0, float(q.get("infl", cfg["inflation"] * 100)), 0.1) / 100,
            hair=container.slider("수익률 할인(연, %p) — 과거 낙관편향 보정", 0.0, 4.0,
                                  float(q.get("hair", cfg.get("return_haircut", 0.01) * 100)), 0.5) / 100,
            goal=container.number_input("목표 실질자산(원, 0=미사용)", value=int(q.get("goal", 500_000_000)), step=50_000_000),
            model=container.selectbox("팬차트 모델", list(gl.MODELS), index=list(gl.MODELS).index(q.get("model", "regime")),
                                      format_func=lambda k: gl.MODELS[k]),
        )
    s["weights"] = tuple((a, v / 100) for a, v in w.items() if v > 0) or (("SPY", 1.0),)
    return s


def portfolio_params(s: dict) -> dict:
    p = {f"w_{a}": int(round(v * 100)) for a, v in s["weights"]}
    p.update(initial=int(s["initial"]), monthly=int(s["monthly"]), years=s["years"], reb=s["reb"],
             infl=round(s["infl"] * 100, 1), hair=round(s["hair"] * 100, 1), goal=int(s["goal"]), model=s["model"])
    return p


@st.cache_data(show_spinner="몬테카를로 시뮬레이션 중…")
def run_goals(version: float, weights: tuple, initial: float, monthly: float, years: int, reb: str,
              infl: float, hair: float, goal: float, n_paths: int = 4000):
    px = load_all(version)[0]
    U = universe()
    spec = pf.PortfolioSpec(weights=dict(weights), initial=initial, monthly_contribution=monthly, years=years,
                            rebalance=reb, inflation=infl, return_haircut=hair, goal=goal or None, n_paths=n_paths)
    rets = pf.monthly_returns(px[[a for a, _ in weights]].dropna())
    rm = rg.fit(px[U["regime"]["vix_id"]].dropna(), U["regime"]["thresholds"])
    res = gl.run_models(rets, spec, rm)
    verdict = gl.evaluate(res, U["goals"])
    sweep = gl.allocation_sweep(rets, spec, rm, U["equity_assets"], U["goals"])
    return res, verdict, sweep, rm, rets


def goals_for(ctx, s: dict, n_paths: int = 4000):
    return run_goals(ctx.version, s["weights"], s["initial"], s["monthly"], s["years"], s["reb"],
                     s["infl"], s["hair"], s["goal"], n_paths)
