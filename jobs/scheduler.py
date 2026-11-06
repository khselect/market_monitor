"""경량 스케줄러 (worker 컨테이너 상주). 외부 의존성 없이 KST 기준 2회 실행.

- 07:15 KST (화~토) : 미국장 마감(16:00 ET = 05:00/06:00 KST) + Yahoo 일봉 확정 여유
- 16:10 KST (월~금) : 한국장 마감(15:30 KST) 이후
- 07:15 슬롯에서 LLM 일일 브리핑 생성, 토요일 07:15(금요일 미국장 마감 후)에는 주간 다이제스트도 생성
  → 기존 '주간 AI 트렌드 / 주간 증시 트렌드' 예약 작업을 대체
- 매주 월요일 실행 시 이벤트 캘린더 원격 갱신
실패해도 프로세스는 유지(다음 슬롯 재시도), 결과는 data/meta/status.json 에 기록.
"""
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm import pipeline  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
SLOTS = [((7, 15), {1, 2, 3, 4, 5}), ((16, 10), {0, 1, 2, 3, 4})]  # weekday: 월=0
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scheduler")


def next_run(now: datetime) -> datetime:
    cands = []
    for d in range(0, 8):
        day = (now + timedelta(days=d)).date()
        for (h, m), wds in SLOTS:
            t = datetime(day.year, day.month, day.day, h, m, tzinfo=KST)
            if t > now and t.weekday() in wds:
                cands.append(t)
    return min(cands)


def main() -> None:
    if "--now" in sys.argv:
        pipeline.run_daily(briefing=True)
    while True:
        nxt = next_run(datetime.now(KST))
        log.info("next run %s", nxt.isoformat())
        time.sleep(max((nxt - datetime.now(KST)).total_seconds(), 1))
        try:
            us_slot = nxt.hour == 7
            st = pipeline.run_daily(refresh_events=nxt.weekday() == 0 or not pipeline.FC_PATH.exists(),
                                    briefing=us_slot, weekly=us_slot and nxt.weekday() == 5)
            log.info("done ok=%s err=%s", st["series_ok"], st["series_err"])
        except Exception:  # noqa: BLE001
            log.exception("daily run failed — 다음 슬롯에서 재시도")


if __name__ == "__main__":
    main()
