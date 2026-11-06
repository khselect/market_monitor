"""LLM 일일 브리핑 — 이벤트 요약 · 시나리오 해설 · 증시/AI 뉴스 피드 (주간 다이제스트 포함).

설계 원칙 (숫자는 모델이, 해석은 LLM이)
  - 시나리오의 '확률'은 국면 전환 행렬·예측분포에서 코드로 계산해 FACTS로 전달.
    LLM은 확률을 만들지 않고, 각 시나리오의 동인·트리거·포트폴리오 영향만 서술.
  - 2단계 호출
      ① 리서치: web_search 서버 도구로 최근 뉴스 수집 (시장 / AI) → 메모 + 출처 URL
      ② 종합: FACTS + 리서치 메모 → JSON 스키마 강제(structured outputs) → 대시보드 패널
  - 키가 없거나 오프라인이면 FACTS만으로 만든 템플릿 브리핑(generator="template")으로 대체.
  - 거절(refusal) 대비 server-side fallbacks("default") 사용.

산출물: data/briefings/latest.json, daily-YYYY-MM-DD.json, weekly-YYYY-MM-DD.json
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from mm.config import DATA_DIR, offline_mode, series_by_id, universe

log = logging.getLogger(__name__)
BRIEF_DIR = DATA_DIR / "briefings"   # 생성은 mm.config 에서(읽기 전용 마운트 대비)

MODEL = os.environ.get("MM_LLM_MODEL", "claude-opus-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search"}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["headline", "market_brief", "korea_brief", "scenarios", "event_watch",
                 "stock_news", "ai_trends", "risks"],
    "properties": {
        "headline": {"type": "string"},
        "market_brief": {"type": "array", "items": {"type": "string"}},
        "korea_brief": {"type": "array", "items": {"type": "string"}},
        "scenarios": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "narrative", "triggers", "portfolio_note"],
            "properties": {"id": {"type": "string", "enum": ["up", "base", "down"]},
                           "narrative": {"type": "string"},
                           "triggers": {"type": "array", "items": {"type": "string"}},
                           "portfolio_note": {"type": "string"}}}},
        "event_watch": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["date", "event", "why"],
            "properties": {"date": {"type": "string"}, "event": {"type": "string"}, "why": {"type": "string"}}}},
        "stock_news": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["title", "summary", "url"],
            "properties": {"title": {"type": "string"}, "summary": {"type": "string"}, "url": {"type": "string"}}}},
        "ai_trends": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["title", "summary", "impact", "url"],
            "properties": {"title": {"type": "string"}, "summary": {"type": "string"},
                           "impact": {"type": "string"}, "url": {"type": "string"}}}},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM = """당신은 한국어로 쓰는 시장 모니터링 대시보드의 애널리스트다.
규칙:
- 수치(지수, 금리, 확률, 예측구간)는 반드시 FACTS에 있는 값만 인용한다. 새 수치를 만들지 않는다.
- 시나리오 확률은 FACTS.scenarios 에 주어진다. 확률을 바꾸거나 새로 제시하지 말고, 각 시나리오가 실현되는 동인·관찰할 트리거·포트폴리오 영향만 서술한다.
- 뉴스는 RESEARCH 메모에 근거한 것만 쓰고, url 은 메모의 출처 URL을 그대로 쓴다. 근거가 없으면 해당 배열을 비워 둔다.
- 대시보드 패널에 들어가므로 각 문장은 짧고 구체적으로(한 항목 1~2문장).
- 투자 권유가 아닌 정보 제공 관점으로 쓴다."""


# ── FACTS: 파이프라인 산출물에서 LLM 입력용 사실 묶음 구성 ─────────────────────────────
def build_facts() -> dict:
    from mm import pipeline
    from mm.data import store
    from mm.eval import calibration as cal
    from mm.events import calendar as evcal

    U, SB = universe(), series_by_id()
    px = store.load_panel(list(SB)).ffill()
    asof = px.dropna(how="all").index.max()

    def chg(sid, n):
        s = px[sid].dropna()
        if len(s) <= n:
            return None
        return round(float((s.iloc[-1] - s.iloc[-1 - n]) * 100 if SB[sid].kind == "yield"
                           else (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100), 2)

    panels = {}
    for pn, spec in U["panels"].items():
        if spec.get("hidden"):
            continue
        panels[pn] = [{"name": it["name"], "last": round(float(px[it["id"]].dropna().iloc[-1]), 3),
                       "chg_1d": chg(it["id"], 1), "chg_1w": chg(it["id"], 5), "chg_1m": chg(it["id"], 21),
                       "unit": "bp" if it["kind"] == "yield" else "%"} for it in spec["items"] if it["id"] in px]

    fc = pipeline.load_forecasts()
    live = fc[fc.actual_pct.isna()] if len(fc) else fc
    forecasts = []
    for _, r in live.iterrows():
        q = r.q
        forecasts.append({"target": SB[r.target].name, "date": str(pd.Timestamp(r.target_date).date()),
                          "median_pct": round(float(q[49]), 2), "p80": [round(float(q[9]), 2), round(float(q[89]), 2)],
                          "p_up": round(float(r.p_up), 3), "event_mult": round(float(r.event_mult), 2),
                          "events": r.events})
    calib = {}
    if len(fc):
        for t, g in fc.groupby("target"):
            s = cal.summary(g)
            if s:
                calib[SB[t].name] = {"cov90": round(float(s["cov90"]), 3), "n": int(s["n"])}

    reg = json.loads(pipeline.REGIME_PATH.read_text(encoding="utf-8")) if pipeline.REGIME_PATH.exists() else None
    goals = json.loads(pipeline.GOALS_PATH.read_text(encoding="utf-8")) if pipeline.GOALS_PATH.exists() else None
    ev = evcal.load()
    up = ev[(ev.date > asof) & (ev.date <= asof + pd.Timedelta(days=10)) & (ev.tier <= 2)]

    scenarios = []
    if reg:
        P, cur = np.array(reg["P"]), reg["current"]
        names = ["안정", "보통", "위기"]
        better = float(P[cur, :cur].sum()) if cur > 0 else 0.0
        worse = float(P[cur, cur + 1:].sum()) if cur < 2 else 0.0
        scenarios = [
            {"id": "base", "label": f"{names[cur]} 국면 유지", "prob_next_month": round(float(P[cur, cur]), 3)},
            {"id": "up", "label": "변동성 완화(국면 개선)", "prob_next_month": round(better, 3)},
            {"id": "down", "label": "변동성 확대(국면 악화)", "prob_next_month": round(worse, 3)},
        ]
    return {
        "asof": str(asof.date()),
        "panels": panels,
        "next_session_forecasts": forecasts,
        "forecast_calibration_90pct_coverage": calib,
        "regime": reg and {k: reg[k] for k in ("current_name", "current_vix", "durations", "thresholds")},
        "scenarios": scenarios,
        "portfolio_goals": goals and {"verdict": [{k: v for k, v in g.items() if k in ("label", "verdict", "worst_bound")}
                                                  for g in goals["verdict"]],
                                      "summary": goals["summary"]},
        "upcoming_events": [{"date": str(d.date()), "event": n, "region": r_, "status": s_}
                            for d, n, r_, s_ in zip(up.date, up.name, up.region, up.status)],
    }


# ── Claude 호출 ───────────────────────────────────────────────────────────────────
def _client():
    import anthropic
    return anthropic.Anthropic(max_retries=3, timeout=600)


def _research(client, topic: str, days: int, max_uses: int) -> dict:
    prompts = {
        "market": (f"최근 {days}일 동안의 미국 증시(나스닥·다우·S&P500)와 한국 증시(코스피) 주요 뉴스, 연준·금리·환율·원자재 관련 이슈, "
                   "그리고 앞으로 1주일 내 시장에 영향을 줄 일정을 조사해서 핵심만 메모해줘. 항목마다 출처 URL을 붙여줘."),
        "ai": (f"최근 {days}일 동안의 AI 산업 주요 뉴스(모델 출시, 빅테크 투자·실적, 반도체, 규제)를 조사해서 "
               "시장·산업 영향 관점으로 핵심만 메모해줘. 항목마다 출처 URL을 붙여줘."),
    }
    messages = [{"role": "user", "content": prompts[topic]}]
    sources, text = [], []
    for _ in range(5):                                     # pause_turn 재개 최대 5회
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=16000, betas=[FALLBACK_BETA], fallbacks="default",
            thinking={"type": "adaptive"}, output_config={"effort": "medium"},
            tools=[{**WEB_SEARCH, "max_uses": max_uses}], messages=messages)
        if resp.stop_reason == "refusal":
            log.warning("research %s refused: %s", topic, getattr(resp, "stop_details", None))
            break
        for b in resp.content:
            if b.type == "text":
                text.append(b.text)
            elif b.type == "web_search_tool_result" and isinstance(b.content, list):
                sources += [{"title": getattr(x, "title", ""), "url": getattr(x, "url", "")} for x in b.content]
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})
    seen, uniq = set(), []
    for s in sources:
        if s["url"] and s["url"] not in seen:
            seen.add(s["url"])
            uniq.append(s)
    return {"topic": topic, "notes": "\n".join(text).strip(), "sources": uniq[:40]}


def _synthesize(client, facts: dict, research: list[dict], extra_news: list[dict], weekly: bool) -> dict:
    n_news = "6~8" if weekly else "3~5"
    user = (f"FACTS:\n{json.dumps(facts, ensure_ascii=False)}\n\n"
            + "".join(f"RESEARCH[{r['topic']}]:\n{r['notes']}\nSOURCES:{json.dumps(r['sources'], ensure_ascii=False)}\n\n"
                      for r in research)
            + (f"NEWS_SENTIMENT(Alpha Vantage):\n{json.dumps(extra_news, ensure_ascii=False)}\n\n" if extra_news else "")
            + f"{'주간 다이제스트(지난 1주 회고 + 다음 주 관전 포인트)' if weekly else '일일 브리핑'}을 작성해줘. "
            f"stock_news·ai_trends는 각 {n_news}개, market_brief 3~5개, korea_brief 1~3개, "
            "scenarios는 FACTS.scenarios의 id(up/base/down) 각각 1개, event_watch는 upcoming_events 중 중요한 것 최대 5개, risks 2~4개.")
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000, betas=[FALLBACK_BETA], fallbacks="default",
        thinking={"type": "adaptive"}, output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM, messages=[{"role": "user", "content": user}])
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"synthesis refused: {getattr(resp, 'stop_details', None)}")
    txt = next(b.text for b in resp.content if b.type == "text")
    out = json.loads(txt)
    out["_usage"] = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens, "model": resp.model}
    return out


# ── 템플릿 폴백 (키 없음/오프라인/LLM 실패) ─────────────────────────────────────────
def template_briefing(facts: dict, weekly: bool = False) -> dict:
    ind = {x["name"]: x for x in facts["panels"].get("indicators", [])}
    def mv(n):
        x = ind.get(n)
        return f"{n} {x['chg_1d']:+.2f}%" if x and x["chg_1d"] is not None else None
    moves = ", ".join(m for m in (mv("S&P 500"), mv("나스닥100"), mv("코스피"), mv("VIX")) if m)
    sec = sorted(facts["panels"].get("sectors", []), key=lambda x: x["chg_1w"] or 0)
    rg = facts.get("regime") or {}
    sc = {s["id"]: s for s in facts.get("scenarios", [])}
    fcs = facts.get("next_session_forecasts", [])
    return {
        "headline": f"{facts['asof']} 마감: {moves}",
        "market_brief": [f"주간 섹터 강세 {sec[-1]['name']} {sec[-1]['chg_1w']:+.1f}%, 약세 {sec[0]['name']} {sec[0]['chg_1w']:+.1f}%" if sec else "",
                         f"현재 국면 {rg.get('current_name', '-')} (VIX 20일 평균 {rg.get('current_vix', float('nan')):.1f})"]
                        + [f"{f['target']} 다음 거래일 중앙 {f['median_pct']:+.2f}%, 80% 구간 {f['p80'][0]:+.1f}~{f['p80'][1]:+.1f}%"
                           + (f" · 이벤트: {f['events']}" if f["events"] else "") for f in fcs],
        "korea_brief": [],
        "scenarios": [{"id": k, "narrative": f"{sc[k]['label']} — 다음 달 확률 {sc[k]['prob_next_month'] * 100:.0f}% (국면 전이행렬)",
                       "triggers": [], "portfolio_note": ""} for k in ("base", "up", "down") if k in sc],
        "event_watch": [{"date": e["date"], "event": e["event"], "why": ""} for e in facts.get("upcoming_events", [])[:5]],
        "stock_news": [], "ai_trends": [],
        "risks": [f"{g['label']}: {g['verdict']}" for g in (facts.get("portfolio_goals") or {}).get("verdict", [])
                  if g["verdict"] != "충족"],
    }


def alphavantage_news(limit: int = 30) -> list[dict]:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key or offline_mode():
        return []
    import requests
    r = requests.get("https://www.alphavantage.co/query",
                     params={"function": "NEWS_SENTIMENT", "topics": "financial_markets,technology",
                             "sort": "LATEST", "limit": limit, "apikey": key}, timeout=30)
    r.raise_for_status()
    feed = r.json().get("feed", [])
    return [{"title": f.get("title"), "url": f.get("url"), "source": f.get("source"),
             "time": f.get("time_published"), "sentiment": f.get("overall_sentiment_label"),
             "score": f.get("overall_sentiment_score")} for f in feed[:limit]]


def generate(weekly: bool = False, client=None) -> dict:
    facts = build_facts()
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "weekly": weekly,
            "asof": facts["asof"], "model": None, "generator": "template", "sources": []}
    use_llm = client is not None or (os.environ.get("ANTHROPIC_API_KEY") and not offline_mode())
    body = None
    if use_llm:
        try:
            client = client or _client()
            days = 7 if weekly else 2
            research = [_research(client, "market", days, 8 if weekly else 5),
                        _research(client, "ai", days, 6 if weekly else 4)]
            try:
                news = alphavantage_news()
            except Exception as e:  # noqa: BLE001
                log.warning("alphavantage news: %r", e)
                news = []
            body = _synthesize(client, facts, research, news, weekly)
            meta.update(generator="claude", model=body.pop("_usage", {}).get("model", MODEL),
                        sources=[s for r in research for s in r["sources"]])
        except Exception as e:  # noqa: BLE001 — 브리핑 실패가 일일 파이프라인을 멈추면 안 됨
            log.exception("LLM briefing failed → template")
            meta["error"] = repr(e)[:300]
    if body is None:
        body = template_briefing(facts, weekly)
    out = {**body, "_meta": meta, "_scenarios": facts.get("scenarios", [])}
    name = f"{'weekly' if weekly else 'daily'}-{facts['asof']}.json"
    for p in (BRIEF_DIR / name, BRIEF_DIR / ("latest_weekly.json" if weekly else "latest.json")):
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load_latest(weekly: bool = False) -> dict | None:
    p = BRIEF_DIR / ("latest_weekly.json" if weekly else "latest.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
