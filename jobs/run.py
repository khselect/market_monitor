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
p.add_argument("cmd", choices=["daily", "update", "events", "backfill", "forecast", "portfolio"])
p.add_argument("--days", type=int, default=500)
a = p.parse_args()
if a.cmd == "daily":
    print(pipeline.run_daily())
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
    print({k: r[k] for k in ("p_loss_real", "cvar5_real")})
