#!/usr/bin/env bash
# ── Capture watchdog (VM host) ───────────────────────────────────────────────────────
#  Two failure modes look identical from the outside — "no fresh ticks" — and they need
#  OPPOSITE responses:
#
#    WEDGED WS   : token is valid, the socket died. A container restart fixes it, and did
#                  (2026-08-14 213s stale, 2026-08-19 450s stale — both healed).
#    DEAD TOKEN  : the Fyers token expired at its 06:00 cutoff and the laptop never pushed
#                  a new one. A restart CANNOT fix this. The container aborts on boot and
#                  re-enters a restart loop.
#
#  The first cut of this script treated them the same, and on 2026-08-20 it restarted a
#  token-dead container at 09:20 and again at 09:45, logged "restart issued" both times,
#  and stopped. The whole session captured nothing (1,579-byte tick mirror vs ~700KB) and
#  nobody was told until the next morning. A log line that says "restart issued" when the
#  restart is known-futile is worse than silence — it reads like a repair.
#
#  So: CHECK THE TOKEN FIRST, on the HOST. Host-side matters — the old tick probe ran via
#  `docker compose exec`, which returns nothing when the container is down, so a dead
#  container reported as 999999 and was indistinguishable from a data fault.
#  Restart ONLY for the case a restart can fix. Escalate the case it cannot.
#
#  Cron (VM host, user ubuntu):
#    20 9 * * 1-5  bash /home/ubuntu/tradebot/capture_watchdog.sh
#    45 9 * * 1-5  bash /home/ubuntu/tradebot/capture_watchdog.sh
# ────────────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd /home/ubuntu/tradebot || exit 1

LOG=/home/ubuntu/tradebot/logs/capture_watchdog.log
STATE=/home/ubuntu/tradebot/logs/.watchdog_last_restart
HEALTH=/home/ubuntu/tradebot/logs/capture_health.txt
NTFY_TOPIC="${NTFY_TOPIC:-tradebot-capture-6e5d23ee60cc}"
# Overridable ONLY so the alert branch can be exercised against a fake token without
# touching the live one mid-session. The alert path is the whole point of this script;
# shipping it untested is how the 2026-08-20 futile-restart logic survived so long.
TOKEN_FILE="${TOKEN_FILE:-/home/ubuntu/tradebot/access_token.txt}"

# The box runs UTC; cron's CRON_TZ controls only WHEN we fire. Stamp IST explicitly —
# the old script printed the literal string "UTC" next to an IST clock.
ts() { TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST'; }
log() { echo "[$(ts)] $*" >> "$LOG"; }

# Push. Best-effort and never fatal: an alert that fails must not also break the check.
notify() {  # notify <priority> <tags> <title> <body>
  curl -fsS --max-time 10 \
       -H "Priority: $1" -H "Tags: $2" -H "Title: $3" \
       -d "$4" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 \
    && log "  ntfy sent ($3)" || log "  ntfy FAILED (alert not delivered)"
}

# ── guards: trading day + session window, host-side so a dead container can't skew it ──
python3 - <<'PY' || exit 0
import datetime as dt, sys
sys.path.insert(0, "/home/ubuntu/tradebot")
from core.market_calendar import is_trading_day
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
ok = is_trading_day(now.date()) and dt.time(9, 15) <= now.time() <= dt.time(15, 30)
sys.exit(0 if ok else 1)
PY

# ── 1. TOKEN, on the host — the check that must come first ──────────────────────────
# Decodes the JWT `exp` straight out of access_token.txt. Stdlib only, no container, no
# broker call. Prints "<state> <minutes_left>".
TOK=$(TOKEN_FILE="$TOKEN_FILE" python3 - <<'PY'
import base64, json, datetime as dt, os
try:
    raw = open(os.environ["TOKEN_FILE"]).read().strip()
except Exception:
    print("MISSING 0"); raise SystemExit
tok = raw.split(":")[-1] if ":" in raw.split(".")[0] else raw
try:
    seg = tok.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))
    left = (claims["exp"] - dt.datetime.now().timestamp()) / 60.0
except Exception:
    print("UNREADABLE 0"); raise SystemExit
print(("VALID" if left > 0 else "EXPIRED"), int(left))
PY
)
STATE_TOK=$(echo "$TOK" | awk '{print $1}')
MINS=$(echo "$TOK" | awk '{print $2}')

if [ "$STATE_TOK" != "VALID" ]; then
  log "TOKEN $STATE_TOK — a restart cannot fix this. NOT restarting."
  echo "UNHEALTHY token=$STATE_TOK at $(ts)" > "$HEALTH"
  notify urgent "rotating_light" "Tradebot capture DOWN — token $STATE_TOK" \
    "The VM has no usable Fyers token, so it is capturing NOTHING today. A container restart will not help. On the laptop: run morning_token.bat (all 5 steps — the login alone is not enough, step 2 scp's it to the VM)."
  exit 2
fi

# Valid but dies before the bell — say so once, while there is still time to act.
if [ "$MINS" -lt 360 ]; then
  log "token VALID but only ${MINS}m left — expires before the session ends"
  notify high "hourglass" "Tradebot token expires mid-session" \
    "Only ${MINS} minutes of token left. Refresh it before it lapses or capture stops mid-day."
fi

# ── 2. tick freshness ───────────────────────────────────────────────────────────────
AGE=$(docker compose exec -T tradebot python -c "
import pandas as pd
from core.mirror_io import read_mirror
now = pd.Timestamp.now(tz='Asia/Kolkata')
tk = read_mirror('ticks', now.date().isoformat(), None, None)
if tk is None or not len(tk):
    print(999999); raise SystemExit
last = tk['ts'].max()
last = last.tz_convert('Asia/Kolkata') if last.tzinfo else last.tz_localize('Asia/Kolkata')
print(int((now - last).total_seconds()))
" 2>/dev/null | tr -d '[:space:]')
[[ "$AGE" =~ ^-?[0-9]+$ ]] || AGE=999999

if [ "$AGE" -le 180 ]; then
  log "alive (${AGE}s since last tick, token ${MINS}m left)"
  echo "HEALTHY age=${AGE}s token=${MINS}m at $(ts)" > "$HEALTH"
  exit 0
fi

# ── 3. stale WITH a valid token = the wedged-WS case a restart actually fixes ────────
# But only once. If the previous run already restarted and we are still stale, the restart
# is not working and repeating it is the 2026-08-20 mistake in a new costume.
NOW_EPOCH=$(date +%s)
LAST=$(cat "$STATE" 2>/dev/null || echo 0)
if [ $((NOW_EPOCH - LAST)) -lt 1500 ]; then
  log "STALE ${AGE}s and a restart was already issued $(( (NOW_EPOCH - LAST) / 60 ))m ago — NOT restarting again"
  echo "UNHEALTHY age=${AGE}s restart_ineffective at $(ts)" > "$HEALTH"
  notify urgent "rotating_light" "Tradebot capture DOWN — restart did not help" \
    "Token is valid (${MINS}m left) but ticks are ${AGE}s stale and a restart has already been tried. Needs a human: check 'docker compose logs tradebot' on the VM."
  exit 2
fi

log "STALE ${AGE}s since last tick, token valid — restarting capture container"
echo "$NOW_EPOCH" > "$STATE"
docker compose restart tradebot >> "$LOG" 2>&1
log "restart issued (will verify on the next run; no second restart before then)"
echo "RESTARTED age=${AGE}s at $(ts)" > "$HEALTH"
notify default "arrows_counterclockwise" "Tradebot capture restarted" \
  "Ticks were ${AGE}s stale with a valid token. Container restarted; next check confirms."
