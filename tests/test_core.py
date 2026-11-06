import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["MM_DATA"] = tempfile.mkdtemp()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mm.data import store  # noqa: E402
from mm.eval import calibration as cal  # noqa: E402
from mm.sim import daily, portfolio  # noqa: E402


def _df(dates, vals):
    return pd.DataFrame({"value": vals}, index=pd.DatetimeIndex(dates, name="date"))


def test_incremental_merge_overrides_overlap():
    d = pd.bdate_range("2024-01-01", periods=10)
    store.merge_and_save("T", _df(d, np.arange(10.0) + 1), source="x", symbol="T", run_id="a", req_start=d[0])
    d2 = pd.bdate_range(d[7], periods=5)      # 3일 겹침(수정치) + 2일 신규
    n = store.merge_and_save("T", _df(d2, [80.0, 90, 100, 11, 12]), source="x", symbol="T", run_id="b", req_start=d2[0])
    s = store.load("T")["value"]
    assert n == 2 and len(s) == 12 and s.iloc[7] == 80.0 and s.index.is_monotonic_increasing


def test_perfectly_calibrated_forecast_covers_nominal():
    rng = np.random.default_rng(0)
    q = np.quantile(rng.standard_normal(200_000), daily.QUANTILES)
    a = rng.standard_normal(4000)
    cov = np.mean([cal.score_one(q, x, 0.5)["in90"] for x in a])
    assert abs(cov - 0.90) < 0.02


def test_conformal_scale_widens_overconfident_model():
    rng = np.random.default_rng(1)
    q_narrow = np.quantile(rng.standard_normal(100_000) * 0.5, daily.QUANTILES)
    h = pd.DataFrame({"actual_date": pd.bdate_range("2024-01-01", periods=250),
                      "actual_pct": rng.standard_normal(250), "q_raw": [q_narrow] * 250})
    assert 1.6 < cal.conformal_scale(h) <= 2.0


def test_portfolio_no_contribution_zero_return_is_flat():
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    rets = pd.DataFrame({"A": 0.0, "B": 0.0}, index=idx)
    spec = portfolio.PortfolioSpec(weights={"A": .5, "B": .5}, initial=100, monthly_contribution=0,
                                   years=2, inflation=0.0, n_paths=50)
    r = portfolio.simulate(rets, spec)
    assert np.allclose(r["bands_nominal"][50], 100) and r["mdd"].min() == 0


def test_alphavantage_parse_and_budget(monkeypatch):
    from mm.data import sources

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"Time Series (Daily)": {"2026-11-05": {"4. close": "101.5", "5. volume": "10"},
                                            "2026-11-04": {"4. close": "100.0", "5. volume": "12"}}}

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "k")
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    av = sources.AlphaVantageSource()
    av.DAILY_BUDGET = 1
    out = av.fetch(["SPY", "^GSPC"], pd.Timestamp("2026-01-01"))
    assert list(out) == ["SPY"] and out["SPY"]["value"].iloc[-1] == 101.5 and out["SPY"].index.is_monotonic_increasing
    assert av.fetch(["XLK"], pd.Timestamp("2026-01-01")) == {}      # 일일 예산 소진 → 호출 안 함


def test_missing_forecast_target_is_skipped_not_crash(monkeypatch):
    """수집 실패로 데이터가 없는 예측 대상(예: ^NDX DNS 타임아웃)은 건너뛰고 나머지는 진행."""
    from mm import pipeline
    good = _df(pd.bdate_range("2020-01-01", periods=400), np.linspace(100, 120, 400))
    empty = _df([], [])
    monkeypatch.setattr(pipeline.store, "load", lambda sid: good if sid == "SPX" else empty)
    out = pipeline._target_prices(["SPX", "NDX"])
    assert list(out) == ["SPX"] and len(out["SPX"]) == 400
