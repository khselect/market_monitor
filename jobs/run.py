"""CLI: python -m jobs.run [daily|update|events|backfill|forecast|portfolio]"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm import pipeline  # noqa: E402
from mm.data.update import update_all  # noqa: E402
from mm.events import calendar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["daily", "update", "events", "backfill", "forecast", "portfolio", "briefing", "weekly", "health"])
p.add_argument("--days", type=int, default=500)
a = p.parse_args()
if a.cmd == "daily":
    st = pipeline.run_daily()
    print(f"완료: 시리즈 ok={st['series_ok']} err={st['series_err']} · 단계오류={len(st['stage_errors'])} · "
          f"건너뜀={st['skipped_targets']} → 상세: python -m jobs.run health")
elif a.cmd == "update":
    print(update_all().to_string())
elif a.cmd == "events":
    print(calendar.refresh().tail(20).to_string())
elif a.cmd == "backfill":
    print(len(pipeline.backfill(a.days)), "rows")
elif a.cmd == "forecast":
    print(pipeline.forecast_today().tail(3).drop(columns=["q", "q_raw"]).to_string())
elif a.cmd == "portfolio":
    r = pipeline.run_portfolio()
    print(r["verdict"].to_string())
elif a.cmd in ("briefing", "weekly"):
    from mm.llm import briefing
    b = briefing.generate(weekly=a.cmd == "weekly")
    print(b["_meta"], "\n", b["headline"])
elif a.cmd == "health":
    # 로컬/운영 점검: 데이터 신선도·수집 오류·예측 채점·브리핑 상태. 문제 있으면 exit 1
    import json
    import pandas as pd
    from mm.data import store
    from mm.llm import briefing as br
    problems = []
    cat = store.catalog()
    err = cat[cat.status != "ok"]
    for _, r in err.iterrows():
        problems.append(f"수집 오류 {r.id}: {r.message}")
    lag = (pd.Timestamp.today().normalize() - pd.to_datetime(cat.last_date)).dt.days
    for sid, d in zip(cat.id, lag):
        if d > 5:
            problems.append(f"데이터 지연 {sid}: {d}일")
    fc = pipeline.load_forecasts()
    pending = fc[fc.actual_pct.isna()] if len(fc) else fc
    if len(pending) > len(fc.target.unique()) if len(fc) else False:
        problems.append(f"채점 대기 예측 {len(pending)}건 (1건/대상 초과 → 실제값 수집 지연 의심)")
    b = br.load_latest()
    st = json.loads(pipeline.STATUS_PATH.read_text()) if pipeline.STATUS_PATH.exists() else {}
    print(f"시리즈 {len(cat)}개 · 오류 {len(err)} · 최근 갱신 {st.get('run_at')} · 오프라인={st.get('offline')}")
    print(f"예측 이력 {len(fc)}건 · 대기 {len(pending)}건")
    print(f"브리핑: {b and b['_meta'].get('generator')} · {b and b['_meta'].get('asof')} · 오류={b and b['_meta'].get('error')}")
    for e in st.get("stage_errors", []):
        problems.append(f"단계 실패 {e['stage']}: {e['error']}")
    for t in st.get("skipped_targets", []):
        problems.append(f"예측 대상 건너뜀 {t}: 데이터 없음/부족 → 다음 실행에서 수집되면 자동 백필")
    if b and b["_meta"].get("error"):
        problems.append("LLM 브리핑 실패 → 템플릿 사용 중: " + b["_meta"]["error"])
    print("\n".join(["⚠ " + p_ for p_ in problems]) or "✓ 이상 없음")
    sys.exit(1 if problems else 0)
