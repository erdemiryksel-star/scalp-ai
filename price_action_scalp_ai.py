# ============================================================
# PRICE ACTION SCALP AI — PAPER MODE — SINGLE FILE
# ============================================================
# - Closed bar only
# - Price-action based (no RSI / MACD)
# - Multi-timeframe scan
# - Paper trading + equity tracking
# - Simple, stable, mobile-friendly
# ============================================================

import time
import math
import ccxt
import pandas as pd
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================
EXCHANGE_ID = "mexc"       # change if needed
QUOTE = "USDT"
TIMEFRAMES = ["1m", "5m", "15m"]
LIMIT = 200

PAPER_START_EQUITY = 1000.0
RISK_PER_TRADE = 0.01     # %1 risk
RR = 2.0                  # Risk : Reward

LOOP_SECONDS = 60

# =========================
# STATE
# =========================
equity = PAPER_START_EQUITY
open_trade = None

# =========================
# EXCHANGE
# =========================
exchange = getattr(ccxt, EXCHANGE_ID)({
    "enableRateLimit": True
})

# =========================
# HELPERS
# =========================
def fetch_df(symbol, tf):
    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=LIMIT)
    df = pd.DataFrame(
        ohlcv, columns=["ts","open","high","low","close","volume"]
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def is_bullish_engulf(df):
    if len(df) < 2:
        return False
    c1 = df.iloc[-2]
    c2 = df.iloc[-1]
    return (
        c2.close > c2.open and
        c1.close < c1.open and
        c2.close > c1.open and
        c2.open < c1.close
    )

def is_bearish_engulf(df):
    if len(df) < 2:
        return False
    c1 = df.iloc[-2]
    c2 = df.iloc[-1]
    return (
        c2.close < c2.open and
        c1.close > c1.open and
        c2.open > c1.close and
        c2.close < c1.open
    )

def position_size(entry, stop):
    risk_amount = equity * RISK_PER_TRADE
    risk_per_unit = abs(entry - stop)
    if risk_per_unit == 0:
        return 0
    return risk_amount / risk_per_unit

# =========================
# MAIN LOOP
# =========================
def main():
    global equity, open_trade

    print("🚀 PRICE ACTION SCALP AI — PAPER MODE STARTED")
    print(f"💰 Starting equity: {equity:.2f} USDT")

    symbols = [
        s for s in exchange.load_markets().keys()
        if s.endswith("/" + QUOTE)
    ][:20]

    while True:
        try:
            for symbol in symbols:
                for tf in TIMEFRAMES:
                    df = fetch_df(symbol, tf)

                    if open_trade is None:
                        # BUY
                        if is_bullish_engulf(df):
                            entry = df.iloc[-1].close
                            stop = df.iloc[-2].low
                            tp = entry + (entry - stop) * RR
                            size = position_size(entry, stop)

                            if size > 0:
                                open_trade = {
                                    "side": "BUY",
                                    "symbol": symbol,
                                    "entry": entry,
                                    "stop": stop,
                                    "tp": tp,
                                    "size": size,
                                }
                                print(f"🟢 BUY {symbol} @ {entry:.4f} | SL {stop:.4f} | TP {tp:.4f}")

                        # SELL
                        elif is_bearish_engulf(df):
                            entry = df.iloc[-1].close
                            stop = df.iloc[-2].high
                            tp = entry - (stop - entry) * RR
                            size = position_size(entry, stop)

                            if size > 0:
                                open_trade = {
                                    "side": "SELL",
                                    "symbol": symbol,
                                    "entry": entry,
                                    "stop": stop,
                                    "tp": tp,
                                    "size": size,
                                }
                                print(f"🔴 SELL {symbol} @ {entry:.4f} | SL {stop:.4f} | TP {tp:.4f}")

                    else:
                        # Manage open trade
                        price = df.iloc[-1].close
                        t = open_trade

                        if t["side"] == "BUY":
                            if price <= t["stop"]:
                                loss = (t["entry"] - t["stop"]) * t["size"]
                                equity -= loss
                                print(f"❌ STOP LOSS | -{loss:.2f} | Equity {equity:.2f}")
                                open_trade = None

                            elif price >= t["tp"]:
                                profit = (t["tp"] - t["entry"]) * t["size"]
                                equity += profit
                                print(f"✅ TAKE PROFIT | +{profit:.2f} | Equity {equity:.2f}")
                                open_trade = None

                        elif t["side"] == "SELL":
                            if price >= t["stop"]:
                                loss = (t["stop"] - t["entry"]) * t["size"]
                                equity -= loss
                                print(f"❌ STOP LOSS | -{loss:.2f} | Equity {equity:.2f}")
                                open_trade = None

                            elif price <= t["tp"]:
                                profit = (t["entry"] - t["tp"]) * t["size"]
                                equity += profit
                                print(f"✅ TAKE PROFIT | +{profit:.2f} | Equity {equity:.2f}")
                                open_trade = None

            time.sleep(LOOP_SECONDS)

        except Exception as e:
            print("⚠️ Error:", e)
            time.sleep(5)

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()