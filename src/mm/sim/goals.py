"""목표 확률 판정 — "10년 후 원금손실 확률 < 5%", "연평균 7% 이상 확률 ≥ 70%" 같은 문장을 정량 판정.

신뢰성 공학의 '보수적 판정' 구조를 그대로 차용
  1) 모델 불확실성: 3개 모델(블록부트스트랩 / 국면전환 / 위기출발)로 같은 지표를 계산 → 범위
  2) 표본 불확실성: 몬테카를로 표준오차 → Wilson 95% 구간
  3) 판정: 최악 모델의 '불리한 쪽' 신뢰한계가 기준을 만족하면 "충족", 일부만 만족하면 "경계", 아니면 "미충족"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mm.sim import portfolio as pf
from mm.sim import regime as rg

MODELS = {"bootstrap": "블록부트스트랩", "regime": "국면전환(현재출발)", "regime_stress": "국면전환(위기출발)"}


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def metric(res: dict, goal: dict) -> float:
    m = goal["metric"]
    if m == "p_loss_nominal":
        return res["p_loss_nominal"]
    if m == "p_loss_real":
        return res["p_loss_real"]
    if m == "p_irr_ge":
        return float((res["irr"] >= goal["param"]).mean())
    if m == "p_cagr_ge":
        return float((res["cagr"] >= goal["param"]).mean())
    if m == "p_mdd_le":       # 최대낙폭이 param(예: -0.30)보다 얕을 확률
        return float((res["mdd"] >= goal["param"]).mean())
    if m == "p_goal_real":
        return float((res["final_real"] >= goal["param"]).mean())
    raise ValueError(m)


def _ok(v: float, goal: dict) -> bool:
    return v < goal["threshold"] if goal["op"] == "<" else v >= goal["threshold"]


def evaluate(results: dict[str, dict], goals: list[dict]) -> pd.DataFrame:
    rows = []
    for g in goals:
        vals = {k: metric(r, g) for k, r in results.items()}
        n = min(r["n_paths"] for r in results.values())
        cis = {k: wilson(v, n) for k, v in vals.items()}
        # 불리한 쪽 한계: '<' 목표면 상한, '≥' 목표면 하한
        worst = max(c[1] for c in cis.values()) if g["op"] == "<" else min(c[0] for c in cis.values())
        n_ok = sum(_ok(v, g) for v in vals.values())
        verdict = "충족" if _ok(worst, g) else ("경계" if n_ok > 0 else "미충족")
        rows.append({"id": g["id"], "label": g["label"], "op": g["op"], "threshold": g["threshold"],
                     **{f"v_{k}": v for k, v in vals.items()}, "worst_bound": worst, "verdict": verdict})
    return pd.DataFrame(rows)


def run_models(rets: pd.DataFrame, spec: pf.PortfolioSpec, rmodel: rg.RegimeModel | None,
               models=tuple(MODELS)) -> dict[str, dict]:
    out = {}
    for m in models:
        if m.startswith("regime") and rmodel is None:
            continue
        s = pf.PortfolioSpec(**{**spec.__dict__, "model": m})
        out[m] = pf.simulate(rets, s, rmodel)
    return out


def allocation_sweep(rets: pd.DataFrame, spec: pf.PortfolioSpec, rmodel: rg.RegimeModel | None,
                     equity_ids: list[str], goals: list[dict], steps=np.arange(0, 101, 10),
                     model: str = "regime", n_paths: int = 2000) -> pd.DataFrame:
    """주식 비중 0~100%: 주식 바스켓(현 배분의 주식 내 비율 유지) vs 나머지(채권·금) 바스켓."""
    w = pd.Series(spec.weights, dtype=float)
    eq = w[w.index.isin(equity_ids)]
    ot = w[~w.index.isin(equity_ids)]
    eq = eq / eq.sum() if eq.sum() > 0 else pd.Series(1 / len(equity_ids), index=equity_ids)
    ot = ot / ot.sum() if ot.sum() > 0 else pd.Series({"IEF": 1.0})
    rows = []
    for s in steps:
        ww = pd.concat([eq * s / 100, ot * (100 - s) / 100]).groupby(level=0).sum()
        ww = ww[ww > 0]
        sp = pf.PortfolioSpec(**{**spec.__dict__, "weights": ww.to_dict(), "n_paths": n_paths, "model": model})
        r = pf.simulate(rets, sp, rmodel)
        row = {"equity": int(s), "irr_med": float(np.median(r["irr"])), "mdd_med": float(np.median(r["mdd"]))}
        for g in goals:
            row[g["id"]] = metric(r, g)
        rows.append(row)
    return pd.DataFrame(rows)
