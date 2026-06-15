#!/usr/bin/env bash
# cloud_bootstrap.sh — one-command stand-up of the Tradebot cloud stack on a fresh
# Ubuntu VM. Run it from inside the cloned repo:
#
#     git clone <your-repo> tradebot && cd tradebot && bash cloud_bootstrap.sh
#
# It installs Docker, collects your secrets into .env (password is hashed for you),
# VERIFIES the headless Fyers login before going live (fail fast), launches the
# stack behind HTTPS+password, and adds a morning re-auth cron. Re-runnable: if
# .env already exists it skips the prompts.
set -euo pipefail
cd "$(dirname "$0")"

[ -f docker-compose.yml ] || { echo "ERROR: run this from the tradebot repo directory."; exit 1; }

# ── 1. Docker ──────────────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo "▶ Installing Docker…"
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
  echo "  (you may need to log out/in for non-sudo docker; this script handles sudo for now)"
fi
if docker info >/dev/null 2>&1; then SUDO=""; else SUDO="sudo"; fi
dc() { $SUDO docker compose "$@"; }

# ── 2. Secrets → .env ──────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  echo "▶ Configuring .env (secrets stay on this server, gitignored)"
  read -rp   "  Fyers App ID [WVDZUTO6HL-100]: " v; APP_ID=${v:-WVDZUTO6HL-100}
  read -rsp  "  Fyers App SECRET: " SECRET; echo
  read -rp   "  Fyers login id (FY_ID, e.g. XS04195): " FYID
  read -rsp  "  Fyers PIN: " PIN; echo
  read -rsp  "  Fyers TOTP base32 secret: " TOTP; echo
  read -rp   "  Redirect URI [http://127.0.0.1:8085]: " v; RURI=${v:-http://127.0.0.1:8085}
  echo       "  Public address: a domain pointed at this VM, or ':443' to use the IP."
  read -rp   "  SITE_ADDRESS [:443]: " v; SITE=${v:-:443}
  read -rp   "  Dashboard username [admin]: " v; AUSER=${v:-admin}
  read -rsp  "  Dashboard password: " PW; echo
  echo "▶ Hashing password…"
  HASH=$($SUDO docker run --rm caddy caddy hash-password --plaintext "$PW")
  umask 077
  cat > .env <<EOF
FYERS_APP_ID=$APP_ID
FYERS_SECRET_KEY=$SECRET
FYERS_FY_ID=$FYID
FYERS_PIN=$PIN
FYERS_TOTP_SECRET=$TOTP
FYERS_REDIRECT_URI=$RURI
SITE_ADDRESS=$SITE
BASIC_AUTH_USER=$AUSER
BASIC_AUTH_HASH=$HASH
EOF
  echo "  wrote .env (chmod 600)"
else
  echo "▶ .env already present — skipping prompts."
  AUSER=$(grep -E '^BASIC_AUTH_USER=' .env | cut -d= -f2-)
fi

# ── 3. Verify the headless login BEFORE going live ─────────────────────────────
echo "▶ Verifying headless Fyers login (TOTP)…"
if ! dc run --rm tradebot python fyers_auth_headless.py; then
  echo "✗ Headless auth failed. Fix the Fyers values in .env and re-run."
  echo "  (If Fyers' endpoints changed, do one browser login and copy access_token.txt here.)"
  exit 1
fi

# ── 4. Launch ──────────────────────────────────────────────────────────────────
echo "▶ Building & starting the stack…"
dc up -d --build

# ── 5. Morning re-auth cron (deterministic fresh token each trading day) ───────
CRON="50 8 * * 1-5 cd $(pwd) && docker compose restart tradebot >/dev/null 2>&1"
( crontab -l 2>/dev/null | grep -v 'compose restart tradebot' ; echo "$CRON" ) | crontab - || \
  echo "  (could not set cron automatically — add this line via 'crontab -e': $CRON)"

echo
echo "✅ Live. Open  https://<this-VM-domain-or-IP>   user: ${AUSER:-admin}"
echo "   Logs:  $SUDO docker compose logs -f tradebot"
echo "   It now captures & stores on THIS server 9:15–15:30 daily — your laptop can be off."
