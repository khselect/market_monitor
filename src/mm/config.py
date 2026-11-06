"""설정·경로 로딩. 모든 경로는 MM_HOME(기본: 프로젝트 루트) 기준."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("MM_HOME", Path(__file__).resolve().parents[2]))


def _load_dotenv(path: Path) -> None:
    """프로젝트 루트 .env 를 환경변수로 로드(이미 설정된 값은 덮지 않음).
    → `python -m jobs.run …` / `streamlit run …` 을 셸에서 직접 실행해도 키가 적용됨."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_dotenv(ROOT / ".env")
CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.environ.get("MM_DATA", ROOT / "data"))
PRICES_DIR = DATA_DIR / "prices"
META_DIR = DATA_DIR / "meta"
FORECAST_DIR = DATA_DIR / "forecasts"
CATALOG_DB = META_DIR / "catalog.sqlite"

for _d in (PRICES_DIR, META_DIR, FORECAST_DIR, DATA_DIR / "briefings"):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:          # 대시보드 컨테이너는 data/ 를 읽기 전용으로 마운트 → worker가 생성
        pass


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
