#!/usr/bin/env bash
# ── Capture watchdog (VM host) ───────────────────────────────────────────────────────
#  During a LIVE session, if the tick feed is STALE (no fresh ticks), restart the capture
#  container so it reloads the token from disk and re-establishes the Fyers WebSocket.
#  Belt for the morning failure modes the in-container WS-supervisor can't fix:
#    - a fresh token was uploaded but the running container never reloaded it, and
#    - a wedged WS that the supervisor's own restart didn't clear.
#  It does NOT fix an EXPIRED token (laptop never refreshed it) — a restart with no valid
#  token still captures nothing; that needs Phase-2 headless token refresh.
#
#  Cron (VM host, user ubuntu) — two shots to self-heal a dead open, CRON_TZ handles UTC box:
#    20 9 * * 1-5  bash /home/ubuntu/tradebot/capture_watchdog.sh
#    45 9 * * 1-5  bash /home/ubuntu/tradebot/capture_watchdog.sh
# ────────────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd /home/ubuntu/tradebot || exit 1
LOG=/home/ubuntu/tradebot/logs/capture_watchdog.log
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

# Seconds since the freshest tick in today's mirror. -1 = not a live session (skip);
# a large number = dead. Trading-day + session guard lives in the probe so a holiday no-ops.
AGE=$(docker compose exec -T tradebot python -c "
import datetime as dt, pandas as pd
from core.market_calendar import is_trading_day
from core.mirror_io import read_mirror
now = pd.Timestamp.now(tz='Asia/Kolkata')
if not is_trading_day(now.date()) or not (dt.time(9,15) <= now.time() <= dt.time(15,30)):
    print(-1); raise SystemExit
tk = read_mirror('ticks', now.date().isoformat(), None, None)
if tk is None or not len(tk):
    print(999999); raise SystemExit
last = tk['ts'].max()
last = last.tz_convert('Asia/Kolkata') if last.tzinfo else last.tz_localize('Asia/Kolkata')
print(int((now - last).total_seconds()))
" 2>/dev/null | tr -d '[:space:]')

[[ "$AGE" =~ ^-?[0-9]+$ ]] || AGE=999999
if [ "$AGE" = "-1" ]; then
  echo "[$(ts)] not a live session — skip" >> "$LOG"
elif [ "$AGE" -gt 180 ]; then
  echo "[$(ts)] STALE ${AGE}s since last tick — restarting capture container" >> "$LOG"
  docker compose restart tradebot >> "$LOG" 2>&1
  echo "[$(ts)] restart issued" >> "$LOG"
else
  echo "[$(ts)] alive (${AGE}s since last tick)" >> "$LOG"
fi
