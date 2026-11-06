"""예측 검증 & 보정 학습 — '방향 적중'이 아니라 '분포가 정직한가'를 본다.

지표
- 구간 커버리지 : 50/80/90/98% 구간에 실제값이 들어간 비율 (명목값과 같아야 정상)
- CRPS          : 분포 전체 점수 (분위수 pinball loss 의 2배 평균으로 근사, 낮을수록 좋음)
- PIT           : F(실제값). 균등분포면 잘 보정됨 / U자형=과신 / 종형=과소신
- Brier         : P(상승) 확률 보정
보정
- Split-conformal 스케일: 최근 W일의 정규화 오차 u=|a−q50|/half_width90 의 90% 분위수
  → 다음 예측의 분포 폭에 곱함. 분포가정 없이 '목표 커버리지'를 맞추는 가장 단순한 방법.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mm.sim.daily import QUANTILES

BANDS = {50: (25, 75), 80: (10, 90), 90: (5, 95), 98: (1, 99)}
CONFORMAL_WINDOW = 250
CONFORMAL_MIN = 40
SCALE_CLIP = (0.6, 2.0)


def _q(q: np.ndarray, pct: int) -> float:
    return float(q[pct - 1])     # QUANTILES = 0.01..0.99 → index pct-1


def score_one(q: np.ndarray, actual: float, p_up: float) -> dict:
    q = np.asarray(q)
    tau = QUANTILES
    pin = np.where(actual >= q, tau * (actual - q), (1 - tau) * (q - actual))
    out = {f"in{b}": float(_q(q, lo) <= actual <= _q(q, hi)) for b, (lo, hi) in BANDS.items()}
    out["crps"] = float(2 * pin.mean())
    out["pit"] = float(np.interp(actual, q, tau, left=0.0, right=1.0))
    out["brier"] = float((p_up - (actual > 0)) ** 2)
    out["abs_err_med"] = float(abs(actual - _q(q, 50)))
    out["hit_dir"] = float((_q(q, 50) > 0) == (actual > 0)) if _q(q, 50) != 0 else np.nan
    return out


def conformal_scale(hist: pd.DataFrame, asof: pd.Timestamp | None = None) -> float:
    """hist: 평가 완료된 예측(q_raw, actual_pct). asof 이전 표본만 사용."""
    h = hist.dropna(subset=["actual_pct"])
    if asof is not None:
        h = h[h.actual_date < asof]
    h = h.tail(CONFORMAL_WINDOW)
    if len(h) < CONFORMAL_MIN:
        return 1.0
    qr = np.vstack(h.q_raw.values)
    med, lo, hi = qr[:, 49], qr[:, 4], qr[:, 94]
    u = np.abs(h.actual_pct.values - med) / np.maximum((hi - lo) / 2, 1e-9)
    return float(np.clip(np.quantile(u, 0.90), *SCALE_CLIP))


def summary(scored: pd.DataFrame, window: int = 250) -> dict:
    s = scored.dropna(subset=["actual_pct"]).tail(window)
    if s.empty:
        return {}
    out = {f"cov{b}": s[f"in{b}"].mean() for b in BANDS}
    out.update(n=len(s), crps=s.crps.mean(), brier=s.brier.mean(), hit_dir=s.hit_dir.mean(),
               pit_hist=np.histogram(s.pit, bins=10, range=(0, 1))[0].tolist())
    return out
