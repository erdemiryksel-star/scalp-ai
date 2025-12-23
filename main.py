# ============================================================
# PRICE ACTION SCALP AI — PAPER MODE
# ============================================================

import time
import ccxt
import pandas as pd
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================
EXCHANGE_ID = "mexc"
QUOTE = "USDT"
TIMEFRAMES = ["1m", "5m", "15m"]
LIMIT = 200

PAPER_START_EQUITY = 1000.0
RISK_PER_TRADE = 0.01
RR = 2.0
LOOP_SECONDS = 60
SYMBOL_LIMIT = 20

# =========================
# STATE
# =========================
equity = PAPER_START_EQUITY
open_trade = None

# =========================
# EXCHANGE
# =========================
exchange = getattr(ccxt, EXCHANGE_ID)({
    "enableRateLimit": True,
})

# =========================
# HELPERS
# =========================
def fetch_df(symbol, tf):
    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=LIMIT)
    df = pd.DataFrame(
        ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def bullish_engulf(df):
    if len(df) < 2:
        return False
    a, b = df.iloc[-2], df.iloc[-1]
    return (
        b.close > b.open and
        a.close < a.open and
        b.close > a.open and
        b.open < a.close
    )

def bearish_engulf(df):
    if len(df) < 2:
        return False
    a, b = df.iloc[-2], df.iloc[-1]
    return (
        b.close < b.open and
        a.close > a.open and
        b.open > a.close and
        b.close < a.open
    )

def position_size(entry, stop):
    risk = equity * RISK_PER_TRADE
    per_unit = abs(entry - stop)
    return 0 if per_unit == 0 else risk / per_unit

# =========================
# MAIN
# =========================
def main():
    global equity, open_trade

    print("🚀 PRICE ACTION SCALP AI — PAPER MODE")
    print(f"💰 Equity: {equity:.2f} USDT")

    exchange.load_markets()
    symbols = [
        s for s in exchange.symbols if s.endswith("/" + QUOTE)
    ][:SYMBOL_LIMIT]

    while True:
        try:
            for symbol in symbols:
                for tf in TIMEFRAMES:
                    try:
                        df = fetch_df(symbol, tf)
                    except:
                        continue

                    price = df.iloc[-1].close

                    if open_trade is None:
                        if bullish_engulf(df):
                            entry = price
                            stop = df.iloc[-2].low
                            tp = entry + (entry - stop) * RR
                            size = position_size(entry, stop)
                            if size > 0:
                                open_trade = ("BUY", symbol, entry, stop, tp, size)
                                print(f"🟢 BUY {symbol} @ {entry:.4f}")

                        elif bearish_engulf(df):
                            entry = price
                            stop = df.iloc[-2].high
                            tp = entry - (stop - entry) * RR
                            size = position_size(entry, stop)
                            if size > 0:
                                open_trade = ("SELL", symbol, entry, stop, tp, size)
                                print(f"🔴 SELL {symbol} @ {entry:.4f}")

                    else:
                        side, sym, entry, stop, tp, size = open_trade
                        if symbol != sym:
                            continue

                        if side == "BUY":
                            if price <= stop:
                                equity -= (entry - stop) * size
                                print("❌ SL")
                                open_trade = None
                            elif price >= tp:
                                equity += (tp - entry) * size
                                print("✅ TP")
                                open_trade = None

                        else:
                            if price >= stop:
                                equity -= (stop - entry) * size
                                print("❌ SL")
                                open_trade = None
                            elif price <= tp:
                                equity += (entry - tp) * size
                                print("✅ TP")
                                open_trade = None

            time.sleep(LOOP_SECONDS)

        except Exception as e:
            print("⚠️", e)
            time.sleep(5)

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    main()