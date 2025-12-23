# ============================================================
# PRICE-ACTION SCALP AI — PAPER — LEARNING (RETRAIN + ROLLBACK)
# Multi-TF Scan: 1w, 1d, 4h, 1h, 15m, 5m, 1m
# Execution ALWAYS on 1m (closed bar only)
# %70+ confidence => ALERT + TRADE
# ============================================================

import os
import time
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import ccxt
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

pd.options.mode.chained_assignment = None

# =========================
# CONFIG
# =========================
EXCHANGE_ID = "mexc"
QUOTE = "USDT"
LOOP_SECONDS = 60
SYMBOL_LIMIT = 20

# Timeframes
EXEC_TF = "1m"
FEATURE_TFS = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]

# Risk / Stop (price-only)
VOL_LEN = 14
SL_VOL_MULT = 2.5

# Filters (signal quality)
OPEN_THR = 0.70      # %70 ve üzeri: bildir + işlem al
MIN_EDGE = 0.25

# RR
RR_BASE = 1.20
RR_MAX = 3.00

# Equity (paper)
START_EQUITY = 1.0

# Learning
LEARNING_LOG = "learning_log.csv"
MODELS_DIR = "models"
ACTIVE_LONG = os.path.join(MODELS_DIR, "model_long.joblib")
ACTIVE_SHORT = os.path.join(MODELS_DIR, "model_short.joblib")
MODEL_META = os.path.join(MODELS_DIR, "meta.json")

RETRAIN_MIN_TRADES = 80
RETRAIN_EVERY_TRADES = 50
HOLDOUT_FRAC = 0.20
AUC_DELTA = 0.01
LOSS_DELTA = 0.01

# Telegram (optional)
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# Data
FETCH_LIMIT = 600

# =========================
# UTILS
# =========================
def now_utc():
    return pd.Timestamp.utcnow()

def log(msg: str):
    print(f"[{now_utc().isoformat()}] {msg}")

def tg_send(msg: str):
    # If not configured, just print
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log(msg)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10,
        )
    except Exception:
        pass

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

# =========================
# EXCHANGE
# =========================
def make_exchange():
    return getattr(ccxt, EXCHANGE_ID)({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

def pick_symbols(ex):
    ex.load_markets()
    syms = []
    for s, m in ex.markets.items():
        try:
            if not m.get("active", True):
                continue
            if not m.get("swap", False):
                continue
            if m.get("quote") != QUOTE:
                continue
            syms.append(m["symbol"])
        except Exception:
            pass

    syms = sorted(list(dict.fromkeys(syms)))
    return syms[:SYMBOL_LIMIT]

def fetch_df(ex, sym, tf="1m", lim=400):
    try:
        d = ex.fetch_ohlcv(sym, tf, limit=lim)
    except Exception:
        return pd.DataFrame()

    if not d or len(d) < 10:
        return pd.DataFrame()

    df = pd.DataFrame(d, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# =========================
# PRICE ACTION FEATURES
# =========================
def true_range(df: pd.DataFrame):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    return pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

def vol_pct(df: pd.DataFrame, l=14):
    tr = true_range(df)
    v = tr.ewm(alpha=1 / l, adjust=False).mean()
    return v / df["close"]

def candle_features(df: pd.DataFrame):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    upper = (h - np.maximum(o, c)) / rng
    lower = (np.minimum(o, c) - l) / rng

    return pd.DataFrame({
        "close_pos": ((c - l) / rng).clip(0, 1),
        "body_pct": ((c - o).abs() / rng).clip(0, 1),
        "wick_up": upper.clip(0, 1),
        "wick_dn": lower.clip(0, 1),
        "ret1": c.pct_change(),
        "ret3": c.pct_change(3),
        "ret6": c.pct_change(6),
    }, index=df.index)

def tf_summary_features(df_tf: pd.DataFrame, tf_name: str, vol_len=14):
    if df_tf is None or df_tf.empty:
        return pd.DataFrame()

    # closed bars only
    d = df_tf.iloc[:-1].copy()
    if len(d) < max(30, vol_len + 5):
        return pd.DataFrame()

    X = candle_features(d)
    X["volp"] = vol_pct(d, vol_len)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        return pd.DataFrame()

    last = X.iloc[[-1]].copy()
    last.columns = [f"{tf_name}_{c}" for c in last.columns]
    return last

def build_multi_tf_x(ex, sym):
    parts = []
    for tf in FEATURE_TFS:
        lim = FETCH_LIMIT if tf == "1m" else 250
        df = fetch_df(ex, sym, tf, lim)
        if df.empty:
            return pd.DataFrame()

        part = tf_summary_features(df, tf, VOL_LEN)
        if part.empty:
            return pd.DataFrame()

        parts.append(part)

    X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    return X.iloc[[-1]].copy() if not X.empty else pd.DataFrame()

def get_exec_closed_bar(ex, sym):
    df = fetch_df(ex, sym, EXEC_TF, FETCH_LIMIT)
    if df.empty or len(df) < 50:
        return pd.DataFrame(), None
    # last closed 1m bar
    return df, df.iloc[-2]

# =========================
# MODEL
# =========================
def default_model():
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=250,
        random_state=42,
    )

def model_feature_list(m):
    return list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else None

def save_meta(meta: dict):
    ensure_dir(MODELS_DIR)
    with open(MODEL_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_models():
    ensure_dir(MODELS_DIR)

    if os.path.exists(ACTIVE_LONG) and os.path.exists(ACTIVE_SHORT):
        try:
            return joblib.load(ACTIVE_LONG), joblib.load(ACTIVE_SHORT)
        except Exception:
            pass

    # bootstrap
    mL, mS = default_model(), default_model()
    X = pd.DataFrame({
        "1m_close_pos": [.5, .7, .3, .6, .4, .8],
        "1m_body_pct":  [.2, .6, .1, .4, .3, .7],
        "1m_wick_up":   [.2, .1, .4, .2, .3, .1],
        "1m_wick_dn":   [.2, .3, .2, .2, .2, .4],
        "1m_ret1":      [.01, -.01, .02, -.02, .01, -.01],
        "1m_ret3":      [.02, -.02, .01, -.01, .03, -.03],
        "1m_ret6":      [.03, -.03, .02, -.02, .01, -.01],
        "1m_volp":      [.01, .006, .012, .008, .009, .007],
        "5m_ret1":      [.005, -.004, .006, -.006, .004, -.003],
        "15m_volp":     [.008, .007, .009, .008, .008, .007],
        "1h_ret3":      [.02, -.01, .015, -.02, .01, -.015],
        "4h_ret6":      [.05, -.03, .04, -.04, .03, -.02],
        "1d_ret3":      [.01, -.005, .008, -.01, .006, -.004],
        "1w_ret1":      [.02, -.01, .015, -.02, .01, -.015],
    })
    y = np.array([0, 1, 1, 0, 1, 0])

    mL.fit(X, y)
    mS.fit(X, y)

    joblib.dump(mL, ACTIVE_LONG)
    joblib.dump(mS, ACTIVE_SHORT)
    save_meta({"active_version": "bootstrap", "last_train_utc": None, "metrics": {}})

    return mL, mS

# =========================
# LEARNING LOG / RETRAIN
# =========================
CORE_COLS = [
    "ts_exit", "symbol", "side", "confidence", "edge",
    "entry", "exit_price_used", "pnl_pct", "result",
]

def append_learning_row(row: dict):
    ensure_dir(os.path.dirname(LEARNING_LOG) or ".")
    df = pd.DataFrame([row])
    header = not os.path.exists(LEARNING_LOG)
    df.to_csv(LEARNING_LOG, mode="a", header=header, index=False)

def load_learning_df():
    if not os.path.exists(LEARNING_LOG):
        return None
    try:
        df = pd.read_csv(LEARNING_LOG)
        return None if df.empty else df
    except Exception:
        return None

def time_split(df: pd.DataFrame, frac: float):
    n = len(df)
    cut = max(1, int(n * (1 - frac)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def evaluate_binary(m, X: pd.DataFrame, y: np.ndarray):
    p = m.predict_proba(X)[:, 1]
    out = {"auc": None, "logloss": None}

    try:
        if len(np.unique(y)) > 1:
            out["auc"] = float(roc_auc_score(y, p))
    except Exception:
        pass

    try:
        out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        pass

    return out

def retrain_if_needed(mL, mS, closed_since: int):
    df = load_learning_df()
    if df is None or len(df) < RETRAIN_MIN_TRADES:
        return mL, mS, 0

    if closed_since < RETRAIN_EVERY_TRADES:
        return mL, mS, closed_since

    feats = [c for c in df.columns if c not in CORE_COLS]
    if len(feats) < 8:
        return mL, mS, 0

    ver = now_utc().strftime("%Y%m%d_%H%M%S")
    metrics = {}
    accepted = False

    def train_side(side: str, old):
        d = df[df["side"] == side].copy()
        if len(d) < 50:
            return old, False, None, None

        tr, te = time_split(d, HOLDOUT_FRAC)

        Xtr = tr[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
        ytr = tr["result"].astype(int).values

        Xte = te[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
        yte = te["result"].astype(int).values

        nm = default_model()
        nm.fit(Xtr, ytr)

        mn = evaluate_binary(nm, Xte, yte)

        f_old = model_feature_list(old)
        Xte_old = Xte.reindex(columns=f_old, fill_value=0) if f_old else Xte
        mo = evaluate_binary(old, Xte_old, yte)

        ok = False
        if mn["auc"] and mo["auc"] and (mn["auc"] - mo["auc"]) >= AUC_DELTA:
            ok = True
        if mn["logloss"] and mo["logloss"] and (mo["logloss"] - mn["logloss"]) >= LOSS_DELTA:
            ok = True

        return (nm if ok else old), ok, mn, mo

    mL2, okL, mnL, moL = train_side("long", mL)
    mS2, okS, mnS, moS = train_side("short", mS)

    if mnL:
        metrics["long_new"] = mnL
    if moL:
        metrics["long_old"] = moL
    if mnS:
        metrics["short_new"] = mnS
    if moS:
        metrics["short_old"] = moS

    ensure_dir(MODELS_DIR)

    if okL:
        joblib.dump(mL2, os.path.join(MODELS_DIR, f"model_long_{ver}.joblib"))
        joblib.dump(mL2, ACTIVE_LONG)
        mL = mL2
        accepted = True

    if okS:
        joblib.dump(mS2, os.path.join(MODELS_DIR, f"model_short_{ver}.joblib"))
        joblib.dump(mS2, ACTIVE_SHORT)
        mS = mS2
        accepted = True

    save_meta({
        "active_version": ver if accepted else "kept",
        "last_train_utc": now_utc().isoformat(),
        "metrics": metrics,
    })

    tg_send(f"🧠 RETRAIN {'ACCEPTED' if accepted else 'REJECTED'}\n{json.dumps(metrics, ensure_ascii=False)}")
    return mL, mS, 0

# =========================
# PAPER POSITION
# =========================
@dataclass
class PaperPos:
    symbol: str
    side: str  # "long" | "short"
    entry: float
    sl: float
    tp: float
    opened_ts: str
    features: dict
    confidence: float
    edge: float

def pnl_pct(entry: float, exit_price: float, side: str):
    return (exit_price / entry - 1.0) if side == "long" else (entry / exit_price - 1.0)

def process_bar_for_exit(bar: pd.Series, pos: PaperPos):
    lo, hi = float(bar["low"]), float(bar["high"])

    if pos.side == "long":
        if lo <= pos.sl:
            return True, pos.sl, "SL"
        if hi >= pos.tp:
            return True, pos.tp, "TP"
    else:
        if hi >= pos.sl:
            return True, pos.sl, "SL"
        if lo <= pos.tp:
            return True, pos.tp, "TP"

    return False, None, None

# =========================
# MAIN LOOP
# =========================
def live():
    ex = make_exchange()
    mL, mS = load_models()
    symbols = pick_symbols(ex)

    equity = START_EQUITY
    open_pos = None
    closed_since_train = 0

    tg_send("🧪 PAPER MODE STARTED — MultiTF (W/D/4H/H1/M15/M5/M1) | THR=70%")

    while True:
        try:
            for sym in symbols:
                df1m, bar = get_exec_closed_bar(ex, sym)
                if df1m.empty or bar is None:
                    continue

                # manage open
                if open_pos and open_pos.symbol == sym:
                    closed, px, out = process_bar_for_exit(bar, open_pos)
                    if closed:
                        pnl = float(pnl_pct(open_pos.entry, px, open_pos.side))
                        equity *= (1.0 + pnl)

                        row = {
                            "ts_exit": now_utc().isoformat(),
                            "symbol": sym,
                            "side": open_pos.side,
                            "confidence": open_pos.confidence,
                            "edge": open_pos.edge,
                            "entry": open_pos.entry,
                            "exit_price_used": px,
                            "pnl_pct": pnl,
                            "result": 1 if pnl > 0 else 0,
                        }

                        for k, v in (open_pos.features or {}).items():
                            fv = safe_float(v)
                            if np.isfinite(fv):
                                row[k] = fv

                        append_learning_row(row)
                        closed_since_train += 1

                        tg_send(
                            f"{'🎯' if pnl > 0 else '🛑'} CLOSED {out} | {sym}\n"
                            f"PnL:%{pnl * 100:.3f} | Equity:{equity:.6f}"
                        )

                        open_pos = None
                        mL, mS, closed_since_train = retrain_if_needed(mL, mS, closed_since_train)

                    continue

                # only one open at a time
                if open_pos:
                    continue

                x = build_multi_tf_x(ex, sym)
                if x.empty:
                    continue

                fL, fS = model_feature_list(mL), model_feature_list(mS)
                xL = x.reindex(columns=fL, fill_value=0.0) if fL else x
                xS = x.reindex(columns=fS, fill_value=0.0) if fS else x

                pL = float(mL.predict_proba(xL)[0, 1])
                pS = float(mS.predict_proba(xS)[0, 1])

                conf = max(pL, pS)
                edge = abs(conf - 0.5)

                # filters
                if conf < OPEN_THR or edge < MIN_EDGE:
                    continue

                side = "long" if pL > pS else "short"
                price = float(bar["close"])

                vp = safe_float(
                    x.get("1m_volp", np.nan).iloc[0] if "1m_volp" in x.columns else np.nan
                )
                if not np.isfinite(vp) or vp <= 0:
                    continue

                stop_dist = price * vp * SL_VOL_MULT
                if stop_dist <= 0:
                    continue

                sl = price - stop_dist if side == "long" else price + stop_dist

                rr = RR_BASE + min(1.0, max(0.0, (conf - OPEN_THR) / (1 - OPEN_THR))) * (RR_MAX - RR_BASE)
                rr = max(RR_BASE, min(RR_MAX, rr))

                tp = price + rr * stop_dist if side == "long" else price - rr * stop_dist

                open_pos = PaperPos(
                    symbol=sym,
                    side=side,
                    entry=price,
                    sl=sl,
                    tp=tp,
                    opened_ts=str(bar["ts"]),
                    features=x.iloc[0].to_dict(),
                    confidence=conf,
                    edge=edge,
                )

                tg_send(
                    f"🟢 PAPER OPEN {side.upper()} | {sym}\n"
                    f"Entry:{price:.6f}\nSL:{sl:.6f}\nTP:{tp:.6f}\n"
                    f"Conf:{conf:.2f} Edge:{edge:.2f} RR:{rr:.2f}\n"
                    f"VOL%:{vp * 100:.3f} x{SL_VOL_MULT}\n"
                    f"Equity:{equity:.6f}"
                )

                break  # stop scanning after opening one trade

            time.sleep(LOOP_SECONDS)

        except Exception as e:
            log("ERROR: " + repr(e))
            time.sleep(5)

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    live()