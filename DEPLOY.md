# Tradebot — cloud deploy (laptop-free, office-accessible)

Run the live dashboard on a small cloud VM so it captures all session data and is
reachable from any browser (incl. a locked-down office laptop) over **HTTPS + a
password** — no dependence on your personal laptop being on.

```
            ┌────────────────────────── Cloud VM (Mumbai) ──────────────────────────┐
  office ──HTTPS:443──▶  Caddy (TLS + basic-auth)  ──internal──▶  Tradebot (Dash)    │
  browser                                                         ├ Fyers WebSocket  │
  + password                                                      ├ DuckDB + parquet │
            │                          headless TOTP auth each morning (no browser)  │
            └───────────────────────────────────────────────────────────────────────┘
```

## Why this shape
- **App port is never public.** Only Caddy's 443 is exposed; Caddy proxies to the app
  over the internal Docker network and enforces a password. Office firewalls allow 443,
  so it gets through where a random port/tunnel would be blocked.
- **One process.** The app keeps WebSocket + candle state in-process, so it runs as a
  single process (no multi-worker gunicorn). Caddy handles TLS/auth separately.
- **Headless token.** `fyers_auth_headless.py` refreshes the daily token via TOTP — no
  browser, so it's fully unattended.

## Fast path (one command)
After you have a VM (step 1) and the repo cloned on it, just run:
```bash
bash cloud_bootstrap.sh
```
It installs Docker, prompts for your secrets (hashes the password for you), **verifies
the headless Fyers login before going live**, launches the stack, and sets the morning
re-auth cron. The manual steps below are the same thing, broken out.

## 1. Provision a VM (Mumbai region for low NSE latency)
Any of: AWS `ap-south-1` (t3.small), Oracle Cloud always-free (Ampere, Mumbai),
DigitalOcean BLR. **1–2 GB RAM is plenty.** Ubuntu 22.04/24.04.

Open inbound **80 and 443** in the cloud firewall/security-group. (80 is only for
Let's Encrypt's HTTP challenge; all real traffic is 443.)

## 2. Install Docker
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
```

## 3. Get the code + configure
```bash
git clone <your-repo-url> tradebot && cd tradebot
cp .env.example .env
# generate the dashboard password hash:
docker run --rm caddy caddy hash-password --plaintext 'choose-a-strong-password'
nano .env          # fill Fyers creds + SITE_ADDRESS + BASIC_AUTH_USER/HASH
```
`.env` is gitignored — your secrets never get committed.

**Fyers TOTP**: Fyers app/web → Profile → enable **TOTP**; when it shows the QR, use the
**base32 secret** (the "manual entry" key), not the 6-digit code → that's `FYERS_TOTP_SECRET`.

**SITE_ADDRESS**:
- Have a domain? Point an A-record at the VM IP and set `SITE_ADDRESS=trade.yourdomain.com`
  → Caddy auto-issues a real Let's Encrypt cert. Cleanest for office access.
- No domain? Set `SITE_ADDRESS=:443` → Caddy serves the VM IP with an internal cert
  (browser shows a one-time "not private" warning you click through). Works fine.

## 4. Verify the headless login once (before trusting the schedule)
```bash
docker compose run --rm tradebot python fyers_auth_headless.py
# → "OK — access token written to ..."   If it fails, fix creds before going on.
```
> `fyers_auth_headless.py` uses Fyers' internal TOTP endpoints — stable in practice but
> undocumented. If Fyers ever changes them, fall back to a one-time browser login and
> copy `access_token.txt` onto the VM.

## 5. Launch
```bash
docker compose up -d --build
docker compose logs -f tradebot        # watch it auth + connect the WebSocket
```
Open **https://trade.yourdomain.com** (or `https://<VM-IP>`), enter your password →
the live dashboard, captured on the VM, viewable from anywhere.

## 6. Fresh token every trading morning (cron)
A clean restart just before the open forces a fresh headless auth:
```bash
crontab -e
# 08:50 IST every weekday — restart so it re-auths for the new session
50 8 * * 1-5  cd ~/tradebot && /usr/bin/docker compose restart tradebot
```
(The supervisor also re-auths on its own when it finds the token stale; the cron just
makes the morning deterministic.)

## Notes / limits
- **Data persists** in `./data` on the VM (mounted volume) across restarts/redeploys.
- **EOD "Daily Context" (Layer 9)** reads the *Daily_Cash_Market* DuckDB, which is a
  separate local pipeline not on the VM — that layer degrades gracefully (guards exist);
  the live capture, conductor, playbook, and signals all work without it. Sync that data
  to the VM later if you want Layer 9 too.
- **Security**: keep the password strong; consider also restricting the security-group to
  your office IP range. Never expose port 8050 directly.
- **Cost**: ~₹400–800/mo, or free tier. The app idles off-market (mirrors `supervise.py`).
