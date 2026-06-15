# Tradebot live dashboard — cloud image.
# Single process: the app relies on in-process singletons (WebSocket capture,
# candle store, _latest map), so it MUST run as one process — do NOT put gunicorn
# with multiple workers in front of it. Caddy (separate container) does TLS + auth.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    TZ=Asia/Kolkata \
    DASH_HOST=0.0.0.0 \
    DASH_PORT=8050 \
    FYERS_HEADLESS=1 \
    TRADEBOT_NO_BROWSER=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata ca-certificates \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050
# supervise.py: ensures a token (headless TOTP here), launches the dashboard, and
# auto-restarts on crash / WebSocket stall during market hours.
CMD ["python", "supervise.py"]
