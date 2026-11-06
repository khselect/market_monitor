"""일간(다음 거래일 마감) 등락률 예측분포 — Filtered Historical Simulation (FHS).

  σ_t   = EWMA 변동성 (RiskMetrics λ=0.94)
  z_i   = r_i / σ_i  (과거 표준화 잔차 — 꼬리·비대칭을 '데이터 그대로' 보존)
  r*    = μ + σ_{t+1} · m_event · z*,   z* ~ 경험분포(z)에서 복원추출
  출력  = 분위수(1..99), P(상승), 사용된 σ·이벤트배수 (나중에 보정 학습에 사용)

정규분포 GBM 대비 장점: 두꺼운 꼬리 + 변동성 군집 반영, 파라미터 거의 없음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAMBDA = 0.94
Z_WINDOW = 1500           # 표준화 잔차 표본 기간(약 6년)
N_SIMS = 10_000
QUANTILES = np.arange(1, 100) / 100


def log_returns(px: pd.Series) -> pd.Series:
    return np.log(px).diff().dropna()


def ewma_sigma(r: pd.Series, lam: float = LAMBDA) -> pd.Series:
    """σ_t (t시점까지의 정보로 t+1 예측용). 초기값 = 첫 60일 표본분산."""
    v = np.empty(len(r))
    v0 = float(r.iloc[:60].var()) if len(r) > 60 else float(r.var())
    prev = v0
    x = r.values
    for i in range(len(r)):
        v[i] = prev                       # i일 수익률을 보기 전 예측분산
        prev = lam * prev + (1 - lam) * x[i] ** 2
    s = pd.Series(np.sqrt(v), index=r.index)
    s.attrs["next"] = float(np.sqrt(prev))  # 마지막 관측 이후(=다음 거래일) σ
    return s


MIN_REGIME_Z = 250        # 국면별 잔차 표본이 이보다 적으면 전체 표본 사용


def forecast_next(px: pd.Series, event_mult: float = 1.0, calib_scale: float = 1.0,
                  n: int = N_SIMS, seed: int | None = None,
                  day_regime: pd.Series | None = None, current_regime: int | None = None) -> dict:
    """day_regime/current_regime 이 주어지면 '같은 VIX 국면의 날'의 표준화 잔차만 사용."""
    r = log_returns(px)
    sig = ewma_sigma(r)
    zser = (r / sig).replace([np.inf, -np.inf], np.nan).dropna()
    used_regime = None
    if day_regime is not None and current_regime is not None:
        reg = day_regime.reindex(zser.index)
        zr = zser[reg == current_regime].iloc[-Z_WINDOW * 2:]
        if len(zr) >= MIN_REGIME_Z:
            zser, used_regime = zr, current_regime
        else:
            zser = zser.iloc[-Z_WINDOW:]
    else:
        zser = zser.iloc[-Z_WINDOW:]
    z = zser.values
    z = (z - z.mean()) / z.std()          # 위치·척도 정규화 (형태만 사용)
    mu = 0.0                              # 일간 드리프트는 노이즈 대비 무시 가능 → 0 (과적합 방지)
    sigma_next = sig.attrs["next"]
    rng = np.random.default_rng(seed)
    zs = rng.choice(z, size=n, replace=True)
    raw = np.expm1(mu + sigma_next * event_mult * zs) * 100                 # 보정 미적용
    pct = np.expm1(mu + sigma_next * event_mult * calib_scale * zs) * 100   # 보정 적용(%)
    return {
        "base_date": px.index[-1],
        "base_close": float(px.iloc[-1]),
        "sigma": sigma_next,
        "event_mult": event_mult,
        "calib_scale": calib_scale,
        "p_up": float((pct > 0).mean()),
        "quantiles": np.quantile(pct, QUANTILES),        # 보정 적용
        "quantiles_raw": np.quantile(raw, QUANTILES),                          # 보정 미적용(컨포멀 학습용)
        "samples": pct,
        "regime": used_regime,
        "z_n": len(z),
    }
