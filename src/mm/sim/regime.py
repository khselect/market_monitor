"""VIX 기반 시장 국면(regime) 모형.

국면 정의 (config: regime.thresholds, 기본 [17, 25])
  안정(calm)  : 월평균 VIX < 17
  보통(normal): 17 ≤ VIX < 25
  위기(stress): VIX ≥ 25

1) 월간 국면 시퀀스 → Markov 전이행렬 P (라플라스 평활) → 평균 지속기간, 정상분포
2) 10년 시뮬: 국면 경로를 P로 생성(현재 국면에서 출발) → 각 달 수익률을 '같은 국면의 과거 달'에서
   블록 단위로 추출 (국면이 이어지면 다음 달을 그대로 이어 붙여 자기상관 보존)
3) 일간 예측: 표준화 잔차 z를 '같은 국면의 날'에서만 추출 → 위기 국면의 두꺼운 꼬리 반영

왜 VIX인가: 관측 가능(은닉상태 추정 불필요), 시장 합의 변동성, 국면 판별이 즉시·설명가능.
HMM 대비 단순하지만 운영 신뢰성(재현성·설명력)이 높음.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NAMES = ["안정", "보통", "위기"]
KEYS = ["calm", "normal", "stress"]


@dataclass
class RegimeModel:
    thresholds: tuple[float, float]
    monthly: pd.Series            # 월말 index → 국면 0/1/2
    P: np.ndarray                 # 3×3 전이행렬 (행: 현재, 열: 다음)
    counts: np.ndarray
    current: int
    current_vix: float
    stationary: np.ndarray

    @property
    def durations(self) -> np.ndarray:   # 평균 지속기간(개월) = 1/(1−p_ii)
        return 1 / np.maximum(1 - np.diag(self.P), 1e-9)

    def to_dict(self) -> dict:
        return {"thresholds": list(self.thresholds), "current": int(self.current),
                "current_name": NAMES[self.current], "current_vix": self.current_vix,
                "P": self.P.round(4).tolist(), "counts": self.counts.astype(int).tolist(),
                "stationary": self.stationary.round(4).tolist(), "durations": self.durations.round(1).tolist(),
                "months": len(self.monthly)}


def classify(vix: pd.Series, thresholds=(17.0, 25.0)) -> pd.Series:
    lo, hi = thresholds
    return pd.Series(np.select([vix < lo, vix < hi], [0, 1], 2), index=vix.index)


def fit(vix_daily: pd.Series, thresholds=(17.0, 25.0), smoothing: float = 0.5) -> RegimeModel:
    vix_daily = vix_daily.dropna()
    m = vix_daily.resample("ME").mean().dropna()
    reg = classify(m, thresholds)
    cnt = np.full((3, 3), smoothing)
    for a, b in zip(reg.values[:-1], reg.values[1:]):
        cnt[a, b] += 1
    P = cnt / cnt.sum(1, keepdims=True)
    w, v = np.linalg.eig(P.T)
    st = np.real(v[:, np.argmin(np.abs(w - 1))])
    st = st / st.sum()
    cur_vix = float(vix_daily.tail(20).mean())   # 최근 20거래일 평균 = '현재 월' 근사
    cur = int(classify(pd.Series([cur_vix]), thresholds).iloc[0])
    return RegimeModel(tuple(thresholds), reg, P, cnt - smoothing, cur, cur_vix, st)


def daily_regime(vix_daily: pd.Series, thresholds=(17.0, 25.0), window: int = 5) -> pd.Series:
    """일간 국면(직전 5일 평균 VIX, 전일까지 정보만 → look-ahead 없음)."""
    return classify(vix_daily.rolling(window).mean().shift(1).dropna(), thresholds)


def simulate_regime_paths(model: RegimeModel, n_paths: int, n_steps: int, rng: np.random.Generator,
                          start: int | None = None) -> np.ndarray:
    s = np.empty((n_paths, n_steps), dtype=np.int8)
    cur = np.full(n_paths, model.current if start is None else start)
    cdf = np.cumsum(model.P, axis=1)
    for t in range(n_steps):
        u = rng.random(n_paths)
        cur = (u[:, None] > cdf[cur]).sum(1)     # 행별 역CDF 샘플링
        s[:, t] = cur
    return s


def regime_bootstrap_idx(month_regimes: np.ndarray, paths: np.ndarray, block: int,
                         rng: np.random.Generator) -> np.ndarray:
    """국면 경로에 맞춰 과거 월 인덱스 추출. 같은 국면이 이어지면 확률 (1−1/L)로 '다음 달'을 이어 붙임."""
    n_paths, T = paths.shape
    pools = [np.flatnonzero(month_regimes == k) for k in range(3)]
    # 표본이 없는 국면은 가장 가까운 국면 풀로 대체
    for k in range(3):
        if len(pools[k]) == 0:
            pools[k] = pools[1] if len(pools[1]) else np.arange(len(month_regimes))
    n_hist = len(month_regimes)
    idx = np.empty((n_paths, T), dtype=np.int64)
    for t in range(T):
        reg = paths[:, t]
        fresh = np.empty(n_paths, dtype=np.int64)
        for k in range(3):
            sel = reg == k
            if sel.any():
                fresh[sel] = rng.choice(pools[k], size=sel.sum())
        if t == 0:
            idx[:, t] = fresh
            continue
        nxt = idx[:, t - 1] + 1
        cont = (rng.random(n_paths) > 1 / block) & (paths[:, t] == paths[:, t - 1]) & (nxt < n_hist)
        cont &= month_regimes[np.minimum(nxt, n_hist - 1)] == reg
        idx[:, t] = np.where(cont, nxt, fresh)
    return idx


def conditional_stats(monthly_rets: pd.DataFrame, month_regimes: pd.Series) -> pd.DataFrame:
    """국면별 연율 수익률·변동성 (자산별) — 해석용 표."""
    reg = month_regimes.reindex(monthly_rets.index, method="ffill")
    out = {}
    for k in range(3):
        r = monthly_rets[reg == k]
        out[(NAMES[k], "연수익%")] = (r.mean() * 12 * 100).round(1)
        out[(NAMES[k], "연변동%")] = (r.std() * np.sqrt(12) * 100).round(1)
        out[(NAMES[k], "개월")] = pd.Series(len(r), index=monthly_rets.columns)
    return pd.DataFrame(out)
