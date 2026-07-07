#!/usr/bin/env bash
# ── BTST unattended runner for the VM (host-side orchestration) ──────────────────────
#  Runs the paper loop INSIDE the tradebot container (it has the deps AND reads the live
#  tick mirrors / DuckDB directly); the ledger persists in the mounted data/validation/,
#  so it survives container rebuilds. Idempotent + fully offline (btst_signal self-heals
#  exits from the tick mirror — no Fyers token, no same-day archive download needed).
#
#  MODES (why the split — a bug the old Windows btst_auto.bat had):
#    eod      15:28 IST → reconcile any due exit, EMIT tonight's strong-close candidates,
#                         scorecard. Fires just before the 15:30 close (near-final bar).
#    morning  09:35 IST → reconcile last night's exits + scorecard ONLY. NO emit — a
#                         morning emit reads a PARTIAL forming bar (09:15–09:35) whose clr
#                         is meaningless and can log a phantom candidate that never clears.
#
#  Cron (VM host, user ubuntu). CRON_TZ handles the UTC box — write IST times directly:
#     CRON_TZ=Asia/Kolkata
#     28 15 * * 1-5  bash /home/ubuntu/tradebot/btst_vm_cron.sh eod
#     35 9  * * 1-5  bash /home/ubuntu/tradebot/btst_vm_cron.sh morning
# ────────────────────────────────────────────────────────────────────────────────────
set -uo pipefail
MODE="${1:-eod}"
cd /home/ubuntu/tradebot || exit 1
LOG=/home/ubuntu/tradebot/logs/btst_cron.log
mkdir -p /home/ubuntu/tradebot/logs
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
DC=(docker compose exec -T tradebot python btst_signal.py)

{
  echo "==================== BTST cron [$MODE] $(ts) ===================="
  "${DC[@]}" --reconcile
  if [ "$MODE" = "eod" ]; then
    "${DC[@]}"                 # emit tonight's candidates (idempotent: updates the OPEN row)
  fi
  "${DC[@]}" --scorecard
  echo "==================== done [$MODE] $(ts) ===================="
} >> "$LOG" 2>&1
