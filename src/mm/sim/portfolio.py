"""10년 포트폴리오 몬테카를로.

모델(model)
  "bootstrap"     : 정상(stationary) 블록 부트스트랩 (Politis & Romano, 1994) — 국면 무시
  "regime"        : VIX 국면 전환 — 현재 국면에서 출발, Markov 전이로 국면 경로 생성 후 국면조건부 추출
  "regime_stress" : 위기 국면에서 출발하는 스트레스 시나리오

투자자 관점 지표
  - 원금손실 확률: 최종 평가액 < 총 납입원금 (명목 / 실질)
  - IRR(금액가중 연수익률): 적립식 투자자가 '내 돈 기준'으로 실제 얻는 연평균 수익률
  - TWR CAGR(시간가중): 전략 자체의 연평균 수익률 (납입 타이밍 무관)
  - MDD: 단위가격(TWR) 기준 최대낙폭 — 적립금 유입이 낙폭을 가리지 않도록
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mm.sim import regime as rg

REB_MONTHS = {"monthly": 1, "quarterly": 3, "annual": 12, "none": 10_000}


@dataclass
class PortfolioSpec:
    weights: dict[str, float]
    initial: float = 100_000_000
    monthly_contribution: float = 1_000_000
    years: int = 10
    rebalance: str = "quarterly"
    inflation: float = 0.025
    return_haircut: float = 0.0      # 연율, 예: 0.01 → 모든 자산 연 1%p 차감 (과거 낙관편향 보정)
    goal: float | None = None        # 실질 목표금액
    block: int = 12
    n_paths: int = 5000
    seed: int = 42
    model: str = "bootstrap"         # bootstrap | regime | regime_stress
    extra: dict = field(default_factory=dict)


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


def irr_annual(initial: float, contrib: float, T: int, fv: np.ndarray) -> np.ndarray:
    """금액가중수익률(IRR) — 벡터화 이분법. 현금흐름: t0 초기투자, 매월 말 적립, T에 평가액."""
    lo = np.full(fv.shape, -0.30)
    hi = np.full(fv.shape, 0.30)
    k = np.arange(T)              # 월말 적립금의 남은 기간: T-1 … 0

    def f(r):
        g = 1 + r[:, None]
        return initial * g[:, 0] ** T + contrib * (g ** k[None, :]).sum(1) - fv

    for _ in range(60):
        mid = (lo + hi) / 2
        pos = f(mid) > 0          # 가치가 FV보다 크면 r이 너무 큼
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return (1 + (lo + hi) / 2) ** 12 - 1


def simulate(rets: pd.DataFrame, spec: PortfolioSpec, regime_model: rg.RegimeModel | None = None) -> dict:
    assets = [a for a in spec.weights if a in rets.columns]
    w = np.array([spec.weights[a] for a in assets], dtype=float)
    w = w / w.sum()
    R = rets[assets].values - spec.return_haircut / 12
    rng = np.random.default_rng(spec.seed)
    T = spec.years * 12

    reg_paths = None
    if spec.model.startswith("regime") and regime_model is not None:
        month_reg = regime_model.monthly.reindex(rets.index, method="ffill").fillna(1).astype(int).values
        start = 2 if spec.model == "regime_stress" else None
        reg_paths = rg.simulate_regime_paths(regime_model, spec.n_paths, T, rng, start=start)
        idx = rg.regime_bootstrap_idx(month_reg, reg_paths, spec.block, rng)
    else:
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
    unit = np.cumprod(np.c_[np.ones(spec.n_paths), (R[idx] @ w) + 1], axis=1)
    mdd = (unit / np.maximum.accumulate(unit, axis=1) - 1).min(1)
    final_nom, final_real = nav[:, -1], real[:, -1]
    irr = irr_annual(spec.initial, spec.monthly_contribution, T, final_nom)
    pct = [5, 25, 50, 75, 95]
    out = {
        "model": spec.model,
        "months": np.arange(T + 1),
        "bands_nominal": {p: np.percentile(nav, p, axis=0) for p in pct},
        "bands_real": {p: np.percentile(real, p, axis=0) for p in pct},
        "invested": invested,
        "final_nominal": final_nom,
        "final_real": final_real,
        "irr": irr,
        "cagr": unit[:, -1] ** (1 / spec.years) - 1,
        "mdd": mdd,
        "p_loss_nominal": float((final_nom < invested[-1]).mean()),
        "p_loss_real": float((final_real < invested[-1]).mean()),   # 실질가치 < 명목 납입원금
        "cvar5_real": float(final_real[final_real <= np.percentile(final_real, 5)].mean()),
        "hist_window": (rets.index[0], rets.index[-1], len(rets)),
        "assets": assets,
        "weights": w,
        "n_paths": spec.n_paths,
    }
    if reg_paths is not None:
        out["regime_share"] = np.bincount(reg_paths.ravel(), minlength=3) / reg_paths.size
    if spec.goal:
        out["p_goal"] = float((final_real >= spec.goal).mean())
    return out
