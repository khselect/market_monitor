"""10년 포트폴리오 몬테카를로 — 정상(stationary) 블록 부트스트랩 (Politis & Romano, 1994).

- 월간 수익률 '행 전체'를 블록으로 복원추출 → 자산 간 상관·꼬리·국면 지속성 보존
- 평균 블록길이 L(기본 12개월)은 기하분포 → 정상성 유지
- 월 적립 + 주기적 리밸런싱 + 인플레이션 실질화
- return_haircut: 과거 수익률이 미래에도 반복된다는 낙관 편향을 줄이는 연간 차감치(시나리오 레버)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REB_MONTHS = {"monthly": 1, "quarterly": 3, "annual": 12, "none": 10_000}


@dataclass
class PortfolioSpec:
    weights: dict[str, float]
    initial: float = 100_000_000
    monthly_contribution: float = 1_000_000
    years: int = 10
    rebalance: str = "quarterly"
    inflation: float = 0.025
    return_haircut: float = 0.0      # 연율, 예: 0.01 → 모든 자산 연 1%p 차감
    goal: float | None = None        # 실질 목표금액
    block: int = 12
    n_paths: int = 5000
    seed: int = 42


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    m = prices.resample("ME").last()
    return m.pct_change().dropna(how="any")


def stationary_bootstrap_idx(n_hist: int, n_steps: int, n_paths: int, block: int,
                             rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / block
    idx = np.empty((n_paths, n_steps), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_hist, n_paths)
    jump = rng.random((n_paths, n_steps)) < p
    fresh = rng.integers(0, n_hist, (n_paths, n_steps))
    for t in range(1, n_steps):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n_hist)
    return idx


def simulate(rets: pd.DataFrame, spec: PortfolioSpec) -> dict:
    assets = [a for a in spec.weights if a in rets.columns]
    w = np.array([spec.weights[a] for a in assets], dtype=float)
    w = w / w.sum()
    R = rets[assets].values - spec.return_haircut / 12
    rng = np.random.default_rng(spec.seed)
    T = spec.years * 12
    idx = stationary_bootstrap_idx(len(R), T, spec.n_paths, spec.block, rng)
    every = REB_MONTHS[spec.rebalance]

    hold = np.tile(w * spec.initial, (spec.n_paths, 1))
    nav = np.empty((spec.n_paths, T + 1))
    nav[:, 0] = spec.initial
    invested = np.full(T + 1, spec.initial, dtype=float)
    for t in range(T):
        hold *= 1 + R[idx[:, t]]
        hold += w * spec.monthly_contribution
        invested[t + 1] = invested[t] + spec.monthly_contribution
        if (t + 1) % every == 0:
            hold = hold.sum(1, keepdims=True) * w
        nav[:, t + 1] = hold.sum(1)

    deflator = (1 + spec.inflation) ** (np.arange(T + 1) / 12)
    real = nav / deflator
    # 적립이 섞인 경로의 MDD는 '단위가격(TWR)' 기준이어야 왜곡이 없음
    unit = np.cumprod(np.c_[np.ones(spec.n_paths), (R[idx] @ w) + 1], axis=1)
    mdd = (unit / np.maximum.accumulate(unit, axis=1) - 1).min(1)
    final_real = real[:, -1]
    pct = [5, 25, 50, 75, 95]
    out = {
        "months": np.arange(T + 1),
        "bands_nominal": {p: np.percentile(nav, p, axis=0) for p in pct},
        "bands_real": {p: np.percentile(real, p, axis=0) for p in pct},
        "invested": invested,
        "final_real": final_real,
        "cagr": (unit[:, -1]) ** (1 / spec.years) - 1,
        "mdd": mdd,
        "p_loss_real": float((final_real < invested[-1] / deflator[-1]).mean()),
        "cvar5_real": float(final_real[final_real <= np.percentile(final_real, 5)].mean()),
        "hist_window": (rets.index[0], rets.index[-1], len(rets)),
        "assets": assets,
        "weights": w,
    }
    if spec.goal:
        out["p_goal"] = float((final_real >= spec.goal).mean())
    return out
