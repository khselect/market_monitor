"""로컬 캐시: 시계열 = Parquet(1 series = 1 file), 메타데이터·수집로그 = SQLite.

설계 원칙
- 대시보드는 이 캐시만 읽는다(외부 호출 0회) → 화면 응답성·장애 격리.
- 증분 업데이트: last_date - OVERLAP 일부터 다시 받아 덮어쓴다
  (Yahoo 종가 사후 수정·배당 조정 반영, FRED 개정치 반영).
- 원자적 쓰기: tmp 파일에 쓰고 rename → 수집 중 대시보드가 깨진 파일을 읽지 않음.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from mm.config import CATALOG_DB, PRICES_DIR

OVERLAP_DAYS = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS series_meta (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    first_date  TEXT,
    last_date   TEXT,
    n_rows      INTEGER,
    fetched_at  TEXT,
    status      TEXT,          -- ok | stale | error
    message     TEXT
);
CREATE TABLE IF NOT EXISTS fetch_log (
    run_id      TEXT NOT NULL,
    id          TEXT NOT NULL,
    ts          TEXT NOT NULL,
    req_start   TEXT,
    rows_recv   INTEGER,
    rows_new    INTEGER,
    status      TEXT,
    message     TEXT
);
CREATE TABLE IF NOT EXISTS quality_flags (
    id          TEXT NOT NULL,
    date        TEXT NOT NULL,
    rule        TEXT NOT NULL,   -- jump | nonpositive | gap
    value       REAL,
    detected_at TEXT NOT NULL,
    PRIMARY KEY (id, date, rule)
);
"""


@contextmanager
def db():
    con = sqlite3.connect(CATALOG_DB)
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def path_for(series_id: str):
    return PRICES_DIR / f"{series_id}.parquet"


def load(series_id: str) -> pd.DataFrame:
    """index=date(DatetimeIndex), columns=[value, volume?]"""
    p = path_for(series_id)
    if not p.exists():
        return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], name="date"))
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def load_panel(ids: list[str], column: str = "value") -> pd.DataFrame:
    frames = {i: load(i)[column] for i in ids if path_for(i).exists()}
    return pd.DataFrame(frames).sort_index()


def last_date(series_id: str) -> pd.Timestamp | None:
    with db() as con:
        row = con.execute("SELECT last_date FROM series_meta WHERE id=?", (series_id,)).fetchone()
    if row and row[0]:
        return pd.Timestamp(row[0])
    df = load(series_id)
    return df.index.max() if len(df) else None


def request_start(series_id: str, history_start: str) -> pd.Timestamp:
    ld = last_date(series_id)
    if ld is None:
        return pd.Timestamp(history_start)
    return ld - pd.Timedelta(days=OVERLAP_DAYS)


def merge_and_save(series_id: str, new: pd.DataFrame, *, source: str, symbol: str,
                   run_id: str, req_start: pd.Timestamp, kind: str = "price") -> int:
    """신규 수신분을 기존 캐시에 병합(겹치는 날짜는 신규값 우선). 반환: 신규 날짜 수."""
    old = load(series_id)
    new = new.dropna(subset=["value"])
    n_new = len(new.index.difference(old.index))
    merged = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    p = path_for(series_id)
    tmp = p.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp)
    tmp.replace(p)

    flags = quality_check(series_id, merged.tail(len(new) + 1), jump=None if kind == "yield" else 0.25)
    with db() as con:
        con.execute(
            """INSERT INTO series_meta(id, source, symbol, first_date, last_date, n_rows, fetched_at, status, message)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET source=excluded.source, symbol=excluded.symbol,
                 first_date=excluded.first_date, last_date=excluded.last_date, n_rows=excluded.n_rows,
                 fetched_at=excluded.fetched_at, status=excluded.status, message=excluded.message""",
            (series_id, source, symbol, str(merged.index.min().date()), str(merged.index.max().date()),
             len(merged), now_utc(), "ok", f"{flags} quality flags" if flags else ""),
        )
        con.execute("INSERT INTO fetch_log VALUES(?,?,?,?,?,?,?,?)",
                    (run_id, series_id, now_utc(), str(req_start.date()), len(new), n_new, "ok", ""))
    return n_new


def log_error(series_id: str, *, source: str, symbol: str, run_id: str, message: str) -> None:
    with db() as con:
        con.execute(
            """INSERT INTO series_meta(id, source, symbol, status, message, fetched_at) VALUES(?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET status='error', message=excluded.message, fetched_at=excluded.fetched_at""",
            (series_id, source, symbol, "error", message[:500], now_utc()))
        con.execute("INSERT INTO fetch_log VALUES(?,?,?,?,?,?,?,?)",
                    (run_id, series_id, now_utc(), None, 0, 0, "error", message[:500]))


def quality_check(series_id: str, df: pd.DataFrame, jump: float | None = 0.25) -> int:
    """간단한 데이터 무결성 규칙. 위반은 기록만 하고 데이터는 보존(사람이 판단)."""
    if df.empty:
        return 0
    v = df["value"]
    rows = []
    for d, x in (v[v <= 0] if jump is not None else v.iloc[:0]).items():
        rows.append((series_id, str(d.date()), "nonpositive", float(x)))
    if jump is not None:
        r = v.pct_change().abs()
        for d, x in r[r > jump].items():
            rows.append((series_id, str(d.date()), "jump", float(x)))
    gaps = v.index.to_series().diff().dt.days
    for d, x in gaps[gaps > 7].items():
        rows.append((series_id, str(d.date()), "gap", float(x)))
    if rows:
        with db() as con:
            con.executemany("INSERT OR IGNORE INTO quality_flags VALUES(?,?,?,?,?)",
                            [(*row, now_utc()) for row in rows])
    return len(rows)


def catalog() -> pd.DataFrame:
    with db() as con:
        return pd.read_sql("SELECT * FROM series_meta ORDER BY id", con)
