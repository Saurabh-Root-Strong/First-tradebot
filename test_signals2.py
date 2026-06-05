from signals import run_full_analysis

SYMS = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "NSE:FINNIFTY-INDEX",
    "NSE:MIDCPNIFTY-INDEX",
]
print("Running multi-timeframe analysis...")
results = run_full_analysis(SYMS)

for sym, r in results.items():
    if "error" in r:
        print(f"{r['label']}: ERROR {r['error']}")
        continue
    ov    = r.get("overall", ("?", ""))
    ws    = r.get("weighted_score", 0)
    tfs   = r.get("timeframes", {})
    print()
    print(f"{'='*60}")
    print(f"  {r['label']}   Overall: {ov[0]}   Weighted Score: {ws:+.2f}")
    print(f"{'='*60}")
    for tf, t in tfs.items():
        print(f"  {t['label']:<8}  {t['signal']:<14}  score={t['score']:>+5.1f}  "
              f"RSI={t.get('rsi',0):.1f}  conf={t['confidence']:.0f}%")
    print(f"  Bull TFs: {r['bull_timeframes']}/4   Bear TFs: {r['bear_timeframes']}/4")
