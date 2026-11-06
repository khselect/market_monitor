"""일일 파이프라인: 수집 → 이벤트 → (어제 예측) 채점 → (다음 거래일) 예측 → 10년 시뮬 → 상태 기록.

산출물(대시보드는 이것만 읽음)
  data/forecasts/forecasts.parquet   예측·채점 이력 (target, base_date 키)
  data/forecasts/portfolio.npz       기본 포트폴리오 10년 시뮬 밴드
  data/meta/status.json              실행 상태·데이터 신선도
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from mm.config import FORECAST_DIR, META_DIR, offline_mode, universe
from mm.data import store
from mm.data.update import update_all
from mm.eval import calibration as cal
from mm.events import calendar as evcal
from mm.sim import daily, portfolio

log = logging.getLogger(__name__)
FC_PATH = FORECAST_DIR / "forecasts.parquet"
PF_PATH = FORECAST_DIR / "portfolio.npz"
STATUS_PATH = META_DIR / "status.json"
MARKET = {"SPX": "US", "NDX": "US", "DJI": "US", "KOSPI": "KR", "KOSDAQ": "KR"}
MULT_REFRESH = 21   # 백필 시 이벤트 배수 재추정 주기(거래일)


def load_forecasts() -> pd.DataFrame:
    if FC_PATH.exists():
        df = pd.read_parquet(FC_PATH)
        df["q"] = df["q"].apply(np.asarray)
        df["q_raw"] = df["q_raw"].apply(np.asarray)
        return df
    return pd.DataFrame()


def save_forecasts(df: pd.DataFrame) -> None:
    df = df.sort_values(["target", "base_date"]).drop_duplicates(["target", "base_date"], keep="last")
    tmp = FC_PATH.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(FC_PATH)


def _row(target: str, px: pd.Series, ev: pd.DataFrame, mults: pd.DataFrame, scale: float) -> dict:
    mkt = MARKET.get(target, "US")
    next_day = px.index[-1] + pd.offsets.BDay(1)
    todays = evcal.events_on(ev, next_day, mkt)
    m = evcal.day_multiplier(mults, todays)
    f = daily.forecast_next(px, event_mult=m, calib_scale=scale, seed=int(px.index[-1].strftime("%Y%m%d")))
    return {"target": target, "base_date": f["base_date"], "target_date": next_day,
            "base_close": f["base_close"], "sigma": f["sigma"], "event_mult": m, "calib_scale": scale,
            "events": ", ".join(todays.name.tolist()), "p_up": f["p_up"],
            "q": f["quantiles"], "q_raw": f["quantiles_raw"],
            "made_at": store.now_utc(), "actual_date": pd.NaT, "actual_pct": np.nan}


def _score_inplace(df: pd.DataFrame, pxs: dict[str, pd.Series]) -> pd.DataFrame:
    for i, r in df[df.actual_pct.isna()].iterrows():
        px = pxs.get(r.target)
        if px is None:
            continue
        after = px[px.index > r.base_date]
        if after.empty:
            continue
        a = (after.iloc[0] / r.base_close - 1) * 100
        df.loc[i, "actual_date"] = after.index[0]
        df.loc[i, "actual_pct"] = a
        for k, v in cal.score_one(r.q, a, r.p_up).items():
            df.loc[i, k] = v
    return df


def backfill(days: int = 500) -> pd.DataFrame:
    """워크포워드 백테스트로 이력 생성 → 첫날부터 보정·신뢰도 패널이 의미를 가짐(look-ahead 없음)."""
    ev = evcal.load()
    rows = []
    for t in universe()["forecast_targets"]:
        px = store.load(t)["value"].dropna()
        hist = pd.DataFrame(columns=["actual_date", "actual_pct", "q_raw"])
        r_all = daily.log_returns(px)
        mults = None
        start = len(px) - days - 1
        for k, i in enumerate(range(start, len(px) - 1)):
            sub = px.iloc[: i + 1]
            if mults is None or k % MULT_REFRESH == 0:
                mults = evcal.estimate_multipliers(r_all, ev, MARKET.get(t, "US"), asof=sub.index[-1] + pd.Timedelta(days=1))
            scale = cal.conformal_scale(hist, asof=sub.index[-1] + pd.Timedelta(days=1))
            row = _row(t, sub, ev, mults, scale)
            nxt = px.index[i + 1]
            row["actual_date"] = nxt
            row["actual_pct"] = (px.iloc[i + 1] / row["base_close"] - 1) * 100
            row.update(cal.score_one(row["q"], row["actual_pct"], row["p_up"]))
            rows.append(row)
            hist = pd.concat([hist, pd.DataFrame([{k2: row[k2] for k2 in ("actual_date", "actual_pct", "q_raw")}])],
                             ignore_index=True)
    df = pd.DataFrame(rows)
    df["backfill"] = True
    old = load_forecasts()
    if len(old):
        df = pd.concat([df, old[~old.get("backfill", False).astype(bool)]], ignore_index=True)
    save_forecasts(df)
    return df


def forecast_today() -> pd.DataFrame:
    ev = evcal.load()
    fc = load_forecasts()
    pxs = {t: store.load(t)["value"].dropna() for t in universe()["forecast_targets"]}
    if len(fc):
        fc = _score_inplace(fc, pxs)
    new = []
    for t, px in pxs.items():
        if len(fc) and ((fc.target == t) & (fc.base_date == px.index[-1])).any():
            continue
        hist = fc[fc.target == t] if len(fc) else pd.DataFrame(columns=["actual_date", "actual_pct", "q_raw"])
        mults = evcal.estimate_multipliers(daily.log_returns(px), ev, MARKET.get(t, "US"))
        row = _row(t, px, ev, mults, cal.conformal_scale(hist))
        row["backfill"] = False
        new.append(row)
    fc = pd.concat([fc, pd.DataFrame(new)], ignore_index=True) if new else fc
    save_forecasts(fc)
    return fc


def run_portfolio(spec: portfolio.PortfolioSpec | None = None) -> dict:
    cfg = universe()["default_portfolio"]
    spec = spec or portfolio.PortfolioSpec(weights=cfg["weights"], initial=cfg["initial"],
                                           monthly_contribution=cfg["monthly_contribution"],
                                           years=cfg["years"], rebalance=cfg["rebalance"],
                                           inflation=cfg["inflation"])
    px = store.load_panel(list(spec.weights)).dropna()
    res = portfolio.simulate(portfolio.monthly_returns(px), spec)
    np.savez_compressed(PF_PATH, months=res["months"], invested=res["invested"],
                        final_real=res["final_real"], mdd=res["mdd"], cagr=res["cagr"],
                        **{f"real_{p}": v for p, v in res["bands_real"].items()},
                        **{f"nom_{p}": v for p, v in res["bands_nominal"].items()})
    return res


def write_status(update_result: pd.DataFrame | None) -> dict:
    cat = store.catalog()
    st = {"run_at": store.now_utc(), "offline": offline_mode(),
          "series_ok": int((cat.status == "ok").sum()), "series_err": int((cat.status == "error").sum()),
          "errors": cat.loc[cat.status == "error", ["id", "message"]].to_dict("records"),
          "last_dates": dict(zip(cat.id, cat.last_date)),
          "update": None if update_result is None else update_result.to_dict("records")}
    STATUS_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return st


def run_daily(refresh_events: bool = True) -> dict:
    res = update_all()
    if refresh_events:
        evcal.refresh()
    if not FC_PATH.exists():
        backfill()
    forecast_today()
    run_portfolio()
    return write_status(res)
