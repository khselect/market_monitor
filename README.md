# Market Monitor

몬테카를로 기반 시장 모니터링 및 포트폴리오 확률 예측 대시보드입니다.
[주요지표] [원자재] [국채 1·2·5·10·30Y] [산업별] [예상] [10년 포트폴리오] [모델 신뢰도]를 한 화면(1920×1080)에 보여줍니다.

![layout](docs/dashboard_layout.png)
> 스크린샷은 `MM_OFFLINE=1` 합성 데이터로 찍은 것입니다. 실데이터에서는 DEMO 배지가 사라집니다.

## 구조

```
config/universe.yaml     패널별 티커 = 화면 구성 (YAML만 고치면 패널 항목이 바뀜)
config/events.yaml       이벤트 캘린더 (수동 + FRED 자동 + 실적 + 규칙)
src/mm/data/             store.py: Parquet 시계열 + SQLite 메타·수집로그·품질플래그
                         sources.py: Yahoo / FRED / Synthetic 어댑터
                         update.py: 증분 수집 (last_date − 10일부터 재수신해 수정치 반영)
src/mm/events/           calendar.py: 하이브리드 캘린더 + 이벤트별 변동성 배수(베이지안 축소추정)
src/mm/sim/              daily.py: FHS(EWMA σ × 경험적 표준화잔차) 다음 거래일 분포
                         portfolio.py: 정상 블록 부트스트랩 10년 시뮬(적립·리밸런싱·실질화)
src/mm/eval/             calibration.py: 커버리지·CRPS·PIT·Brier + split-conformal 폭 보정
src/mm/pipeline.py       수집 → 이벤트 → 어제 예측 채점 → 오늘 예측 → 10년 시뮬 → status.json
app/dashboard.py         Streamlit 한 화면 (읽기 전용, 외부 호출 0회)
jobs/run.py              CLI: daily | update | events | backfill | forecast | portfolio
jobs/scheduler.py        상주 스케줄러: 07:15 KST(화~토, 미국장 후) / 16:10 KST(월~금, 한국장 후)
```

## 로컬 실행

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env              # FRED_API_KEY 입력 (무료)
python -m jobs.run daily          # 최초: 2000년~ 전체 수집 + 500일 워크포워드 백필
streamlit run app/dashboard.py
```

`MM_OFFLINE=1 python -m jobs.run daily`를 쓰면 네트워크 없이 합성 데이터로 실행됩니다.

## 운영 배포 (Mac mini · Docker · Cloudflare Tunnel)

```bash
docker compose up -d --build
```

- **서브도메인** (`market.sielain.com`, 기존 air/parts와 같은 방식): `MM_BASE_PATH`를 비워 두고
  Zero Trust → Tunnels → sielain-tunnel → Published application routes에서
  Subdomain `market` / URL `market_monitor_app:8501`을 추가합니다.
- **서브디렉토리** (`sielain.com/market`): `.env`에 `MM_BASE_PATH=market`을 넣고,
  route의 Path를 `market`, URL을 `market_monitor_app:8501`로 지정합니다.
  터널은 경로를 그대로 전달하므로 앱이 `/market` 접두사로 서비스해야 합니다.

## 이벤트 캘린더 유지보수
- **연 1회** (12월): `config/events.yaml`의 `manual`에 다음 해 FOMC와 금통위 일정을 추가합니다. 미확인 일정은 `status: tentative`로 둡니다.
- CPI·고용·GDP·PCE·PPI·소매판매는 FRED가 자동으로 수집합니다. 첫 실행 시 `fred_releases`의 release_id 이름을 확인하세요.
- 옵션만기(셋째 금요일)는 코드가 자동으로 생성합니다.
