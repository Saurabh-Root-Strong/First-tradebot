"""L1 adapters — the ONLY layer allowed to touch the outside world.

`import fyers_apiv3`, `requests`, websocket clients, and DB drivers may appear
HERE and nowhere else. Everything upstream depends on an interface, never on the
vendor SDK. broker/ owns Fyers (auth, token, rate-limit, REST quota, websocket);
nse.py, news.py own their feeds. Target home for dashboard.py's LiveFeed.
"""
