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
from mm.sim import daily, goals, portfolio
from mm.sim import regime as rg

log = logging.getLogger(__name__)
FC_PATH = FORECAST_DIR / "forecasts.parquet"
PF_PATH = FORECAST_DIR / "portfolio.npz"
STATUS_PATH = META_DIR / "status.json"
REGIME_PATH = META_DIR / "regime.json"
GOALS_PATH = FORECAST_DIR / "goals.json"
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


def _vix() -> pd.Series:
    return store.load(universe()["regime"]["vix_id"])["value"].dropna()


def _regime_inputs(vix: pd.Series, base: pd.Timestamp) -> tuple[pd.Series, int]:
    th = universe()["regime"]["thresholds"]
    v = vix[vix.index <= base]
    cur = int(rg.classify(pd.Series([v.tail(5).mean()]), th).iloc[0])
    return rg.daily_regime(v, th), cur


def _row(target: str, px: pd.Series, ev: pd.DataFrame, mults: pd.DataFrame, scale: float,
         vix: pd.Series | None = None) -> dict:
    mkt = MARKET.get(target, "US")
    next_day = px.index[-1] + pd.offsets.BDay(1)
    todays = evcal.events_on(ev, next_day, mkt)
    m = evcal.day_multiplier(mults, todays)
    use = universe()["regime"].get("daily_conditioning", True) and vix is not None and len(vix)
    dr, cur = _regime_inputs(vix, px.index[-1]) if use else (None, None)
    f = daily.forecast_next(px, event_mult=m, calib_scale=scale, seed=int(px.index[-1].strftime("%Y%m%d")),
                            day_regime=dr, current_regime=cur)
    return {"target": target, "base_date": f["base_date"], "target_date": next_day,
            "base_close": f["base_close"], "sigma": f["sigma"], "event_mult": m, "calib_scale": scale,
            "events": ", ".join(todays.name.tolist()), "p_up": f["p_up"],
            "regime": -1 if f["regime"] is None else int(f["regime"]),
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


MIN_HISTORY = 300   # 예측에 필요한 최소 관측치(EWMA 초기화 + 표준화 잔차 표본)


def _target_prices(targets=None) -> dict[str, pd.Series]:
    """예측 대상 중 데이터가 충분한 것만 반환. 부족/누락은 경고만 남기고 건너뜀(한 대상 장애 격리)."""
    out = {}
    for t in targets or universe()["forecast_targets"]:
        px = store.load(t)["value"].dropna()
        if len(px) < MIN_HISTORY:
            log.warning("forecast target %s skipped: %d rows (< %d) — 수집 실패 또는 이력 부족", t, len(px), MIN_HISTORY)
            continue
        out[t] = px
    return out


def backfill(days: int = 500, targets: list[str] | None = None) -> pd.DataFrame:
    """워크포워드 백테스트로 이력 생성 → 첫날부터 보정·신뢰도 패널이 의미를 가짐(look-ahead 없음).
    targets 를 주면 해당 대상만 다시 만들고 나머지 대상의 기존 이력은 보존."""
    ev = evcal.load()
    vix = _vix()
    rows = []
    pxs = _target_prices(targets)
    for t, px in pxs.items():
        hist = pd.DataFrame(columns=["actual_date", "actual_pct", "q_raw"])
        r_all = daily.log_returns(px)
        mults = None
        start = max(len(px) - days - 1, MIN_HISTORY - 1)   # 이력이 짧으면 앞부분은 학습용으로만 사용
        for k, i in enumerate(range(start, len(px) - 1)):
            sub = px.iloc[: i + 1]
            if mults is None or k % MULT_REFRESH == 0:
                mults = evcal.estimate_multipliers(r_all, ev, MARKET.get(t, "US"), asof=sub.index[-1] + pd.Timedelta(days=1))
            scale = cal.conformal_scale(hist, asof=sub.index[-1] + pd.Timedelta(days=1))
            row = _row(t, sub, ev, mults, scale, vix)
            nxt = px.index[i + 1]
            row["actual_date"] = nxt
            row["actual_pct"] = (px.iloc[i + 1] / row["base_close"] - 1) * 100
            row.update(cal.score_one(row["q"], row["actual_pct"], row["p_up"]))
            rows.append(row)
            hist = pd.concat([hist, pd.DataFrame([{k2: row[k2] for k2 in ("actual_date", "actual_pct", "q_raw")}])],
                             ignore_index=True)
    df = pd.DataFrame(rows)
    if len(df):
        df["backfill"] = True
    old = load_forecasts()
    if len(old):
        is_bf = old["backfill"].fillna(False).astype(bool) if "backfill" in old else pd.Series(False, index=old.index)
        keep = old[~is_bf | ~old.target.isin(list(pxs))]     # 다시 만든 대상의 백필만 교체
        df = pd.concat([df, keep], ignore_index=True) if len(df) else keep
    if len(df):
        save_forecasts(df)
    return df


def missing_backfill_targets() -> list[str]:
    """예측 이력이 없는 대상(최초 실행 시 수집 실패했다가 복구된 대상 등)."""
    fc = load_forecasts()
    have = set(fc.target.unique()) if len(fc) else set()
    return [t for t in universe()["forecast_targets"] if t not in have]


def forecast_today() -> pd.DataFrame:
    ev = evcal.load()
    vix = _vix()
    fc = load_forecasts()
    pxs = _target_prices()
    if len(fc):
        fc = _score_inplace(fc, pxs)
    new = []
    for t, px in pxs.items():
        if len(fc) and ((fc.target == t) & (fc.base_date == px.index[-1])).any():
            continue
        hist = fc[fc.target == t] if len(fc) else pd.DataFrame(columns=["actual_date", "actual_pct", "q_raw"])
        mults = evcal.estimate_multipliers(daily.log_returns(px), ev, MARKET.get(t, "US"))
        row = _row(t, px, ev, mults, cal.conformal_scale(hist), vix)
        row["backfill"] = False
        new.append(row)
    fc = pd.concat([fc, pd.DataFrame(new)], ignore_index=True) if new else fc
    save_forecasts(fc)
    return fc


def default_spec(**over) -> portfolio.PortfolioSpec:
    cfg = universe()["default_portfolio"]
    kw = dict(weights=cfg["weights"], initial=cfg["initial"], monthly_contribution=cfg["monthly_contribution"],
              years=cfg["years"], rebalance=cfg["rebalance"], inflation=cfg["inflation"],
              return_haircut=cfg.get("return_haircut", 0.01))
    kw.update(over)
    return portfolio.PortfolioSpec(**kw)


def fit_regime() -> rg.RegimeModel:
    rm = rg.fit(_vix(), universe()["regime"]["thresholds"])
    REGIME_PATH.write_text(json.dumps(rm.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    return rm


def run_portfolio(spec: portfolio.PortfolioSpec | None = None) -> dict:
    """기본 포트폴리오: 3개 모델 시뮬 → 목표 판정 + 배분 탐색 → goals.json (대시보드·LLM 브리핑 입력)."""
    U = universe()
    spec = spec or default_spec()
    rm = fit_regime()
    rets = portfolio.monthly_returns(store.load_panel(list(spec.weights)).dropna())
    res = goals.run_models(rets, spec, rm)
    verdict = goals.evaluate(res, U["goals"])
    sweep = goals.allocation_sweep(rets, spec, rm, U["equity_assets"], U["goals"])
    main = res.get("regime") or res["bootstrap"]
    np.savez_compressed(PF_PATH, months=main["months"], invested=main["invested"],
                        final_real=main["final_real"], mdd=main["mdd"], irr=main["irr"],
                        **{f"real_{p}": v for p, v in main["bands_real"].items()})
    summary = {k: {"irr_p50": float(np.median(r["irr"])), "irr_p5": float(np.percentile(r["irr"], 5)),
                   "irr_p95": float(np.percentile(r["irr"], 95)), "p_loss_nominal": r["p_loss_nominal"],
                   "p_loss_real": r["p_loss_real"], "mdd_p50": float(np.median(r["mdd"])),
                   "final_real_p50": float(np.median(r["final_real"]))} for k, r in res.items()}
    GOALS_PATH.write_text(json.dumps({"spec": {**spec.__dict__}, "summary": summary,
                                      "verdict": verdict.to_dict("records"), "sweep": sweep.to_dict("records")},
                                     ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {"results": res, "verdict": verdict, "sweep": sweep, "regime": rm}


def write_status(update_result: pd.DataFrame | None) -> dict:
    cat = store.catalog()
    st = {"run_at": store.now_utc(), "offline": offline_mode(),
          "series_ok": int((cat.status == "ok").sum()), "series_err": int((cat.status == "error").sum()),
          "errors": cat.loc[cat.status == "error", ["id", "message"]].to_dict("records"),
          "last_dates": dict(zip(cat.id, cat.last_date)),
          "update": None if update_result is None else update_result.to_dict("records")}
    STATUS_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return st


def _stage(errors: list, name: str, fn, *a, **kw):
    """단계별 장애 격리: 한 단계가 실패해도 다음 단계와 상태 기록은 계속 진행."""
    try:
        return fn(*a, **kw)
    except Exception as e:  # noqa: BLE001
        log.exception("stage %s failed", name)
        errors.append({"stage": name, "error": repr(e)[:300]})
        return None


def run_daily(refresh_events: bool = True, briefing: bool = True, weekly: bool = False) -> dict:
    errors: list[dict] = []
    res = _stage(errors, "update", update_all)
    if refresh_events:
        _stage(errors, "events", evcal.refresh)
    missing = missing_backfill_targets()
    if missing:
        _stage(errors, "backfill", backfill, targets=missing)
    _stage(errors, "forecast", forecast_today)
    _stage(errors, "portfolio", run_portfolio)
    st = write_status(res)
    if briefing:
        from mm.llm import briefing as br
        b = _stage(errors, "briefing", br.generate, weekly=False)
        if b:
            st["briefing"] = b["_meta"]
        if weekly:
            bw = _stage(errors, "briefing_weekly", br.generate, weekly=True)
            if bw:
                st["briefing_weekly"] = bw["_meta"]
    st["stage_errors"] = errors
    st["skipped_targets"] = [t for t in universe()["forecast_targets"] if t not in _target_prices()]
    STATUS_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return st
