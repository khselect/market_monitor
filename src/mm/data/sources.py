"""데이터 소스 어댑터. 모든 소스는 DataFrame(index=date, columns=[value, (volume)]) 을 반환.

- YahooSource : yfinance 일괄 다운로드(배치 + 재시도 + 백오프). 비공식 API → 실패를 정상 상황으로 취급.
- FredSource  : FRED API(키 있으면) / fredgraph.csv(키 없이) 폴백.
- SyntheticSource : 오프라인 개발·데모용 합성 데이터(화면에 DEMO 표시됨).
"""
from __future__ import annotations

import io
import os
import time
import zlib

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "market-monitor/0.1 (personal research)"}


def _retry(fn, tries: int = 4, base: float = 2.0):
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — 외부 소스 오류는 종류 불문 재시도
            last = e
            time.sleep(base ** k)
    raise last


class YahooSource:
    name = "yahoo"

    def fetch(self, symbols: list[str], start: pd.Timestamp) -> dict[str, pd.DataFrame]:
        out = self._fetch_batch(symbols, start)
        missing = [s for s in symbols if s not in out]
        if missing:                       # 배치 중 일부 실패(DNS 타임아웃 등)는 예외 없이 빈 열로 옴 → 개별 재시도
            time.sleep(3)
            for sym in missing:
                out.update(self._fetch_batch([sym], start))
                time.sleep(1)
        return out

    def _fetch_batch(self, symbols: list[str], start: pd.Timestamp) -> dict[str, pd.DataFrame]:
        import yfinance as yf
        from mm.config import META_DIR
        try:                              # 기본 캐시(~/Library/Caches/py-yfinance) 충돌 회피
            yf.set_tz_cache_location(str(META_DIR / "yf-cache"))
        except Exception:  # noqa: BLE001
            pass

        def _dl():
            return yf.download(symbols, start=start.strftime("%Y-%m-%d"), auto_adjust=True,
                               progress=False, group_by="ticker", threads=True)

        raw = _retry(_dl)
        out = {}
        for sym in symbols:
            try:
                sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            sub = sub.dropna(how="all")
            if sub.empty:
                continue
            df = pd.DataFrame({"value": sub["Close"].astype(float)})
            if "Volume" in sub:
                df["volume"] = sub["Volume"].astype(float)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            out[sym] = df
        return out


class FredSource:
    name = "fred"
    API = "https://api.stlouisfed.org/fred/series/observations"
    CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self):
        self.key = os.environ.get("FRED_API_KEY")

    def _one(self, sid: str, start: pd.Timestamp) -> pd.DataFrame:
        if self.key:
            r = requests.get(self.API, params={"series_id": sid, "api_key": self.key, "file_type": "json",
                                               "observation_start": start.strftime("%Y-%m-%d")},
                             headers=UA, timeout=30)
            r.raise_for_status()
            obs = r.json()["observations"]
            s = pd.Series({o["date"]: o["value"] for o in obs})
        else:
            r = requests.get(self.CSV, params={"id": sid, "cosd": start.strftime("%Y-%m-%d")},
                             headers=UA, timeout=30)
            r.raise_for_status()
            c = pd.read_csv(io.StringIO(r.text))
            s = pd.Series(c.iloc[:, 1].values, index=c.iloc[:, 0].values)
        s = pd.to_numeric(s.replace(".", np.nan), errors="coerce")
        df = pd.DataFrame({"value": s})
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df

    def fetch(self, symbols: list[str], start: pd.Timestamp) -> dict[str, pd.DataFrame]:
        out = {}
        for sid in symbols:
            out[sid] = _retry(lambda sid=sid: self._one(sid, start))
            time.sleep(0.3)
        return out


class SyntheticSource:
    """결정론적 합성 시계열. 공통 시장요인 + 개별요인 + 변동성 국면 → 레이아웃·파이프라인 검증용."""
    name = "synthetic"

    LEVELS = {"^GSPC": 6700, "^NDX": 24500, "^DJI": 46500, "^KS11": 3900, "^KQ11": 880, "^VIX": 17,
              "DX-Y.NYB": 99, "KRW=X": 1420, "CL=F": 61, "BZ=F": 65, "GC=F": 3950, "SI=F": 47,
              "HG=F": 5.0, "NG=F": 4.2, "DGS1": 3.65, "DGS2": 3.55, "DGS5": 3.70, "DGS10": 4.10,
              "DGS30": 4.65}
    VOL = {"^VIX": 0.9, "NG=F": 0.55, "CL=F": 0.35, "BZ=F": 0.33, "SI=F": 0.30, "^KQ11": 0.25,
           "^NDX": 0.22, "XLE": 0.26, "XLK": 0.24, "TLT": 0.15, "IEF": 0.07, "GLD": 0.15, "GC=F": 0.15,
           "KRW=X": 0.09, "DX-Y.NYB": 0.07, "EWY": 0.24, "^KS11": 0.20}
    BETA = {"^VIX": -4.0, "TLT": -0.2, "IEF": -0.1, "GLD": 0.05, "GC=F": 0.05, "KRW=X": -0.3,
            "DX-Y.NYB": -0.15, "XLU": 0.5, "XLP": 0.55, "XLV": 0.7, "XLK": 1.25, "^NDX": 1.2,
            "XLY": 1.1, "EWY": 1.0, "^KS11": 0.8, "^KQ11": 1.0}

    def __init__(self, start: str = "2004-01-01"):
        self.dates = pd.bdate_range(start, pd.Timestamp.today().normalize() - pd.offsets.BDay(1))
        rng = np.random.default_rng(7)
        n = len(self.dates)
        # 변동성 국면(평상/스트레스) 2-state Markov
        state = np.zeros(n, dtype=int)
        for t in range(1, n):
            p = 0.004 if state[t - 1] == 0 else 0.03
            state[t] = 1 - state[t - 1] if rng.random() < p else state[t - 1]
        self.vol_mult = np.where(state == 1, 2.3, 0.85)
        self.mkt = rng.standard_t(5, n) / np.sqrt(5 / 3) * 0.17 / np.sqrt(252) * self.vol_mult
        self.rate_f = rng.standard_normal(n) * 0.055 * np.sqrt(self.vol_mult)

    def _series(self, sym: str) -> pd.Series:
        rng = np.random.default_rng(zlib.crc32(sym.encode()))
        n = len(self.dates)
        if sym.startswith("DGS"):
            lvl = self.LEVELS[sym]
            mat = {"DGS1": 1, "DGS2": 2, "DGS5": 5, "DGS10": 10, "DGS30": 30}[sym]
            shocks = self.rate_f * (1.1 - 0.012 * mat) + rng.standard_normal(n) * 0.02
            path = np.cumsum(shocks[::-1])[::-1] * -1  # 끝점 고정, 과거로 역적분
            y = lvl + path - path[-1]
            y = 2.5 + (y - 2.5) * 0.35 + 0.35 * np.sin(np.linspace(0, 6, n)) * (1 + mat / 30)
            y = y - y[-1] + lvl
            return pd.Series(np.clip(y, 0.05, None), index=self.dates)
        vol = self.VOL.get(sym, 0.18) / np.sqrt(252)
        beta = self.BETA.get(sym, 1.0 if not sym.endswith("=F") else 0.3)
        idio_scale = np.sqrt(max(vol ** 2 - (beta * 0.17 / np.sqrt(252)) ** 2, (0.25 * vol) ** 2))
        r = beta * self.mkt + rng.standard_normal(n) * idio_scale * np.sqrt(self.vol_mult) + 0.07 / 252
        if sym == "^VIX":
            x = np.zeros(n); x[0] = np.log(17)
            for t in range(1, n):
                x[t] = x[t - 1] + 0.03 * (np.log(16 * self.vol_mult[t] / 0.85 ** 0.5) - x[t - 1]) + r[t]
            return pd.Series(np.exp(x), index=self.dates) * (17 / np.exp(x[-1]))
        p = np.exp(np.cumsum(r))
        lvl = self.LEVELS.get(sym, 100.0)
        return pd.Series(p / p[-1] * lvl, index=self.dates)

    def fetch(self, symbols: list[str], start: pd.Timestamp) -> dict[str, pd.DataFrame]:
        out = {}
        for s in symbols:
            v = self._series(s)
            v = v[v.index >= start]
            df = pd.DataFrame({"value": v})
            df.index.name = "date"
            out[s] = df
        return out


class AlphaVantageSource:
    """Yahoo 실패 시 폴백 (ETF·개별주만: ^지수, =F 선물, .KS 는 무료 티어 미지원).
    무료 키는 하루 25회 → 일일 사용량을 SQLite에 기록하고 DAILY_BUDGET 초과 시 호출하지 않음.
    compact(최근 100거래일)만 사용 → 증분 보충 용도."""
    name = "alphavantage"
    URL = "https://www.alphavantage.co/query"
    DAILY_BUDGET = 20

    def __init__(self):
        self.key = os.environ.get("ALPHAVANTAGE_API_KEY")

    @staticmethod
    def supports(symbol: str) -> bool:
        return not any(c in symbol for c in "^=.")

    def _budget_ok(self) -> bool:
        from mm.data import store
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        with store.db() as con:
            con.execute("CREATE TABLE IF NOT EXISTS av_usage (day TEXT PRIMARY KEY, n INTEGER)")
            row = con.execute("SELECT n FROM av_usage WHERE day=?", (today,)).fetchone()
            if row and row[0] >= self.DAILY_BUDGET:
                return False
            con.execute("INSERT INTO av_usage VALUES(?,1) ON CONFLICT(day) DO UPDATE SET n=n+1", (today,))
        return True

    def fetch(self, symbols: list[str], start: pd.Timestamp) -> dict[str, pd.DataFrame]:
        out = {}
        if not self.key:
            return out
        for sym in symbols:
            if not self.supports(sym) or not self._budget_ok():
                continue
            r = requests.get(self.URL, params={"function": "TIME_SERIES_DAILY", "symbol": sym,
                                               "outputsize": "compact", "apikey": self.key},
                             headers=UA, timeout=30)
            r.raise_for_status()
            ts = r.json().get("Time Series (Daily)")
            if not ts:                      # 한도 초과·오류 메시지는 JSON 본문으로 옴
                continue
            df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            # 주의: 무료 TIME_SERIES_DAILY 는 비조정 종가 → 분배금 지급일 전후 Yahoo(조정)와 소폭 불일치 가능
            out[sym] = pd.DataFrame({"value": df["4. close"], "volume": df["5. volume"]},
                                    index=df.index.rename("date")).loc[lambda d: d.index >= start]
            time.sleep(1.2)
        return out


def source_for(name: str):
    return {"yahoo": YahooSource, "fred": FredSource, "synthetic": SyntheticSource,
            "alphavantage": AlphaVantageSource}[name]()
