"""Fyers broker adapter — the single boundary to the broker.

Owns: auth/token (access_token.txt), REST fetch (option chain, futures, quotes),
the REST option-chain quota + backoff (the thing that kills chain capture ~11am),
and the websocket. dashboard.py's _get_auth/_validate_token/fetch_*/_start_ws land here.
"""
