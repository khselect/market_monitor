"""증분 업데이트 오케스트레이터: 소스별로 묶어 배치 요청 → 캐시 병합.

실패는 시리즈 단위로 격리(한 티커 실패가 전체 배치를 멈추지 않음).
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict

import pandas as pd

from mm.config import all_series, offline_mode, universe
from mm.data import store
from mm.data.sources import AlphaVantageSource, SyntheticSource, source_for

log = logging.getLogger(__name__)
BATCH = 15


def update_all(ids: list[str] | None = None) -> pd.DataFrame:
    run_id = uuid.uuid4().hex[:8]
    hist_start = universe()["history_start"]
    targets = [s for s in all_series() if ids is None or s.id in ids]
    groups: dict[str, list] = defaultdict(list)
    for s in targets:
        groups["synthetic" if offline_mode() else s.source].append(s)

    results = []
    synth = SyntheticSource() if "synthetic" in groups else None
    for src_name, items in groups.items():
        src = synth if src_name == "synthetic" else source_for(src_name)
        # 같은 요청 시작일끼리 묶어 호출 수 최소화
        by_start: dict[pd.Timestamp, list] = defaultdict(list)
        for s in items:
            by_start[store.request_start(s.id, hist_start)].append(s)
        for start, ss in sorted(by_start.items()):
            for i in range(0, len(ss), BATCH):
                chunk = ss[i:i + BATCH]
                try:
                    got = src.fetch([s.symbol for s in chunk], start)
                except Exception as e:  # noqa: BLE001
                    for s in chunk:
                        store.log_error(s.id, source=src_name, symbol=s.symbol, run_id=run_id, message=repr(e))
                        results.append((s.id, src_name, "error", 0, repr(e)[:80]))
                    continue
                for s in chunk:
                    df = got.get(s.symbol)
                    if df is None or df.empty:
                        store.log_error(s.id, source=src_name, symbol=s.symbol, run_id=run_id, message="empty response")
                        results.append((s.id, src_name, "error", 0, "empty"))
                        continue
                    n = store.merge_and_save(s.id, df, source=src_name, symbol=s.symbol,
                                             run_id=run_id, req_start=start, kind=s.kind)
                    results.append((s.id, src_name, "ok", n, ""))
    results = _alphavantage_fallback(targets, results, run_id, hist_start)
    res = pd.DataFrame(results, columns=["id", "source", "status", "rows_new", "message"])
    log.info("update run=%s ok=%d err=%d", run_id, (res.status == "ok").sum(), (res.status != "ok").sum())
    return res


def _alphavantage_fallback(targets, results, run_id, hist_start):
    """Yahoo 실패분 중 ETF·개별주를 Alpha Vantage로 재시도 (키 있을 때만)."""
    av = AlphaVantageSource()
    if not av.key or offline_mode():
        return results
    failed = {r[0] for r in results if r[2] == "error" and r[1] == "yahoo"}
    by_id = {s.id: s for s in targets}
    out = [r for r in results if r[0] not in failed]
    for sid in failed:
        s = by_id[sid]
        start = store.request_start(sid, hist_start)
        try:
            got = av.fetch([s.symbol], start).get(s.symbol) if av.supports(s.symbol) else None
        except Exception as e:  # noqa: BLE001
            got, msg = None, repr(e)[:80]
        else:
            msg = "unsupported/empty"
        if got is not None and not got.empty:
            n = store.merge_and_save(sid, got, source="alphavantage", symbol=s.symbol, run_id=run_id,
                                     req_start=start, kind=s.kind)
            out.append((sid, "alphavantage", "ok", n, "yahoo 실패 → AV 폴백"))
        else:
            out.append((sid, "yahoo", "error", 0, msg))
    return out
