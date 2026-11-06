"""경제 이벤트 캘린더 (하이브리드) + 이벤트별 변동성 배수 추정.

소스 우선순위
  1) manual (config/events.yaml)  — FOMC, 금통위 등 '연 8회' 류 정책 일정
  2) fred   (FRED releases/dates) — CPI/NFP/GDP/PCE… 과거+미래 일정 자동 (FRED_API_KEY)
  3) earnings (yfinance)          — 대형주 실적발표
  4) rule   (코드 생성)            — 옵션만기(셋째 금요일)
캐시: data/meta/events.parquet (주 1회 원격 갱신, 실패 시 마지막 캐시 사용)
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import requests

from mm.config import META_DIR, events_config, offline_mode

log = logging.getLogger(__name__)
EVENTS_CACHE = META_DIR / "events.parquet"
COLS = ["date", "type", "name", "tier", "region", "status", "source"]

# 표본이 적을 때 쓰는 사전(prior) 변동성 배수 — 문헌·경험치 수준의 보수적 값
PRIOR_MULT = {"FOMC": 1.35, "BOK": 1.15, "CPI": 1.40, "NFP": 1.25, "GDP": 1.10,
              "PCE": 1.10, "PPI": 1.10, "RETAIL": 1.10, "OPEX": 1.05, "EARN": 1.10}
PRIOR_WEIGHT = 12   # 사전분포의 '가상 표본수' (shrinkage 강도)


def _meta(t: str) -> dict:
    return events_config()["event_types"].get(t, {"name": t, "tier": 3, "region": "US"})


def manual_events() -> pd.DataFrame:
    rows = []
    for e in events_config().get("manual", []):
        m = _meta(e["type"])
        name = m["name"] + (" (+SEP)" if e.get("sep") else "")
        rows.append([pd.Timestamp(e["date"]), e["type"], name, m["tier"], m["region"],
                     e.get("status", "confirmed"), "manual"])
    return pd.DataFrame(rows, columns=COLS)


def opex_events(start: str, end: str) -> pd.DataFrame:
    months = pd.date_range(start, end, freq="MS")
    d = [m + pd.offsets.WeekOfMonth(week=2, weekday=4) for m in months]  # 셋째 금요일
    m = _meta("OPEX")
    return pd.DataFrame([[x, "OPEX", m["name"], m["tier"], m["region"], "confirmed", "rule"] for x in d],
                        columns=COLS)


def fred_release_events(start: str, end: str) -> pd.DataFrame:
    key = os.environ.get("FRED_API_KEY")
    if not key or offline_mode():
        return pd.DataFrame(columns=COLS)
    rows = []
    for typ, rid in events_config().get("fred_releases", {}).items():
        r = requests.get("https://api.stlouisfed.org/fred/release/dates",
                         params={"release_id": rid, "api_key": key, "file_type": "json",
                                 "realtime_start": start, "realtime_end": end,
                                 "include_release_dates_with_no_data": "true", "limit": 10000},
                         timeout=30)
        r.raise_for_status()
        m = _meta(typ)
        today = pd.Timestamp.today().normalize()
        for o in r.json().get("release_dates", []):
            d = pd.Timestamp(o["date"])
            rows.append([d, typ, m["name"], m["tier"], m["region"],
                         "confirmed" if d <= today else "scheduled", "fred"])
    return pd.DataFrame(rows, columns=COLS)


def earnings_events() -> pd.DataFrame:
    if offline_mode():
        return pd.DataFrame(columns=COLS)
    import yfinance as yf
    rows, m = [], _meta("EARN")
    for sym in events_config().get("earnings_watch", []):
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=12)
        except Exception as e:  # noqa: BLE001
            log.warning("earnings %s: %r", sym, e)
            continue
        if ed is None:
            continue
        for d in ed.index:
            rows.append([pd.Timestamp(d).tz_localize(None).normalize(), "EARN", f"{sym} 실적",
                         m["tier"], "KR" if sym.endswith(".KS") else "US", "scheduled", "yahoo"])
    return pd.DataFrame(rows, columns=COLS)


def synthetic_history(start: str, end: str) -> pd.DataFrame:
    """오프라인 모드: CPI(매월 ~12일), NFP(첫 금요일) 근사 일정 생성 — 데모 전용."""
    months = pd.date_range(start, end, freq="MS")
    rows = []
    for mth in months:
        nfp = mth + pd.offsets.WeekOfMonth(week=0, weekday=4)
        cpi = pd.bdate_range(mth + pd.Timedelta(days=9), periods=3)[1]
        for d, t in ((nfp, "NFP"), (cpi, "CPI")):
            m = _meta(t)
            rows.append([d, t, m["name"], m["tier"], m["region"], "synthetic", "synthetic"])
    return pd.DataFrame(rows, columns=COLS)


def refresh(start: str = "2004-01-01", horizon_days: int = 120) -> pd.DataFrame:
    end = (pd.Timestamp.today() + pd.Timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    parts = [manual_events(), opex_events(start, end)]
    remote = []
    for fn in (lambda: fred_release_events(start, end), earnings_events):
        try:
            remote.append(fn())
        except Exception as e:  # noqa: BLE001
            log.warning("event source failed: %r — 이전 캐시 유지", e)
    if offline_mode():
        remote.append(synthetic_history(start, end))
    if not any(len(r) for r in remote) and EVENTS_CACHE.exists():
        old = pd.read_parquet(EVENTS_CACHE)
        remote.append(old[old.source.isin(["fred", "yahoo", "synthetic"])])
    ev = pd.concat(parts + remote, ignore_index=True)
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev.drop_duplicates(["date", "type", "name"]).sort_values("date").reset_index(drop=True)
    ev.to_parquet(EVENTS_CACHE)
    return ev


def load() -> pd.DataFrame:
    if EVENTS_CACHE.exists():
        return pd.read_parquet(EVENTS_CACHE)
    return refresh()


def affected_days(ev: pd.DataFrame, market: str) -> pd.DataFrame:
    """이벤트 → 영향받는 '거래일'로 사상.
    US 시장: US 이벤트 당일. KR 시장: KR 이벤트 당일 + US 이벤트는 다음 영업일(시차)."""
    if market == "KR":
        kr = ev[ev.region == "KR"].copy()
        us = ev[ev.region == "US"].copy()
        us["date"] = us["date"] + pd.offsets.BDay(1)
        return pd.concat([kr, us], ignore_index=True)
    return ev[ev.region == "US"].copy()


def events_on(ev: pd.DataFrame, day: pd.Timestamp, market: str = "US") -> pd.DataFrame:
    a = affected_days(ev, market)
    return a[a.date == pd.Timestamp(day).normalize()]


def estimate_multipliers(returns: pd.Series, ev: pd.DataFrame, market: str = "US",
                         window: int = 20, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """이벤트일 |r| / 직전 window일 평균 |r| 의 평균 → 사전분포와 축소추정(shrinkage).

    m_post = (n·m_hat + k·m_prior) / (n + k)
    신뢰성 공학의 베이지안 고장률 갱신과 같은 구조: 표본이 쌓일수록 데이터가 지배.
    asof 를 주면 그 이전 표본만 사용(백테스트 look-ahead 방지).
    비율의 평균은 이벤트가 없어도 1보다 큼(E[X/Y] ≥ E[X]/E[Y]) → 전체 거래일 평균비율로 나눠 정규화.
    """
    absr = returns.abs()
    if asof is not None:
        absr = absr[absr.index < asof]
    base = absr.rolling(window).mean().shift(1)
    all_ratio = (absr / base).replace([np.inf, -np.inf], np.nan).dropna()
    baseline = float(all_ratio.mean()) if len(all_ratio) else 1.0   # 무이벤트 기준선(젠센 편향 ≈1.05) 보정
    a = affected_days(ev, market)
    rows = []
    for t in sorted(ev.type.unique()):
        idx = pd.DatetimeIndex(a.loc[a.type == t, "date"].unique()).intersection(absr.index)
        ratio = (absr.loc[idx] / base.loc[idx]).replace([np.inf, -np.inf], np.nan).dropna()
        n = len(ratio)
        m_hat = float(ratio.mean()) / baseline if n else np.nan
        prior = PRIOR_MULT.get(t, 1.0)
        post = prior if n == 0 else (n * m_hat + PRIOR_WEIGHT * prior) / (n + PRIOR_WEIGHT)
        rows.append({"type": t, "n": n, "m_hat": m_hat, "prior": prior,
                     "multiplier": float(np.clip(post, 0.8, 2.5))})
    return pd.DataFrame(rows).set_index("type")


def day_multiplier(mults: pd.DataFrame, todays: pd.DataFrame) -> float:
    """같은 날 여러 이벤트 → 분산 가산 근사: m = sqrt(1 + Σ(m_i² − 1)), 상한 2.5"""
    if todays.empty:
        return 1.0
    excess = sum(max(mults.loc[t, "multiplier"] ** 2 - 1, 0) for t in todays.type.unique() if t in mults.index)
    return float(min(np.sqrt(1 + excess), 2.5))
