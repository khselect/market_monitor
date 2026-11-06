#!/usr/bin/env bash
# 로컬(Mac) 운영 점검 — 실데이터로 전 과정을 한 번 돌리고 대시보드를 띄움
#   사용: bash scripts/local_check.sh          (최초 1회 2000년~ 전체 수집: 수 분 소요)
#         bash scripts/local_check.sh --docker  (docker compose 로 worker+dashboard 기동)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { cp .env.example .env; echo ".env 생성됨 → FRED_API_KEY / ANTHROPIC_API_KEY / ALPHAVANTAGE_API_KEY 입력 후 재실행"; exit 1; }
set -a; source .env; set +a
export MM_OFFLINE=0

if [ "${1:-}" = "--docker" ]; then
  docker network inspect docker_sielain_net >/dev/null 2>&1 || docker network create docker_sielain_net
  docker compose up -d --build
  echo "worker 로그: docker compose logs -f worker"
  echo "대시보드: http://localhost:${MM_PORT:-8502}"
  exit 0
fi

python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt pytest
echo "── 1/4 단위 테스트"; NO_PROXY=127.0.0.1 python -m pytest -q tests
echo "── 2/4 일일 파이프라인(실데이터)"; python -m jobs.run daily | tail -1
echo "── 3/4 상태 점검"; python -m jobs.run health || echo "⚠ 위 경고 확인"
echo "── 4/4 대시보드 http://localhost:8501  (Ctrl+C 종료)"
streamlit run app/dashboard.py
