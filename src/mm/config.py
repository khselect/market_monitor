"""설정·경로 로딩. 모든 경로는 MM_HOME(기본: 프로젝트 루트) 기준."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("MM_HOME", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.environ.get("MM_DATA", ROOT / "data"))
PRICES_DIR = DATA_DIR / "prices"
META_DIR = DATA_DIR / "meta"
FORECAST_DIR = DATA_DIR / "forecasts"
CATALOG_DB = META_DIR / "catalog.sqlite"

for _d in (PRICES_DIR, META_DIR, FORECAST_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Series:
    id: str
    name: str
    source: str          # yahoo | fred | synthetic
    symbol: str
    kind: str            # price | yield
    panel: str
    maturity: float | None = None


@lru_cache
def universe() -> dict:
    return yaml.safe_load((CONFIG_DIR / "universe.yaml").read_text(encoding="utf-8"))


@lru_cache
def events_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "events.yaml").read_text(encoding="utf-8"))


def all_series() -> list[Series]:
    out = []
    for panel, spec in universe()["panels"].items():
        for it in spec["items"]:
            out.append(Series(panel=panel, **it))
    return out


def series_by_id() -> dict[str, Series]:
    return {s.id: s for s in all_series()}


def offline_mode() -> bool:
    """MM_OFFLINE=1 이면 외부 소스 대신 합성 데이터 사용(개발/데모 전용)."""
    return os.environ.get("MM_OFFLINE", "0") == "1"
