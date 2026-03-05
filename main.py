import os
import logging
import asyncio
import time
import pytz
import joblib
import requests
import base64
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from threading import Thread
import ta

from flask import Flask
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("olasmos_fx")

WAT = pytz.timezone("Africa/Lagos")
UTC = pytz.utc

TOKEN             = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID           = os.environ.get("TELEGRAM_CHAT_ID", "")
TWELVE_DATA_API   = os.environ.get("TWELVE_DATA_API", "")
NEWS_API_KEY      = os.environ.get("NEWS_API_KEY", "")
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")
GITHUB_USERNAME   = "Olasmos1"
GITHUB_REPO       = "olasmos-fx-bot"
PORT              = int(os.environ.get("PORT", 10000))

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
YFINANCE_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F",
}
TWELVE_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "XAUUSD": "XAU/USD",
}
PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "XAUUSD": 0.10,
}
RISK_PERCENT = {
    "EURUSD": 2.0,
    "GBPUSD": 2.0,
    "USDJPY": 2.0,
    "XAUUSD": 1.0,
}
TP1_RR = 1.5
TP2_RR = 2.5
TP3_RR = 4.0
ADX_THRESHOLD = 20
AI_CONFIDENCE_THRESH = 0.55
MIN_CONFIDENCE_ACTIVE = 60
MIN_CONFIDENCE_PASSIVE = 75
SCAN_INTERVAL_ACTIVE = 60
SCAN_INTERVAL_PASSIVE = 300
AUTHORIZED_USERS = [int(CHAT_ID)] if CHAT_ID else []

bot_paused = False
active_trades = {}
signal_stats = {p: {"wins": 0, "losses": 0, "total": 0} for p in PAIRS}
models = {}
scalers = {}
feature_cols = {}
_last_weekend = None
daily_trades = []
setup_counter = 0

# FLASK
app_flask = Flask(__name__)

@app_flask.route("/")
def home():
    return "Olasmos FX Bot Running!"

@app_flask.route("/health")
def health():
    return {"status": "running"}, 200

def run_flask():
    app_flask.run(host="0.0.0.0", port=PORT, use_reloader=False, debug=False)

# DATA FETCHING
def fetch_twelve(symbol, interval, count=100):
    try:
        resp = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": TWELVE_SYMBOLS.get(symbol),
                "interval": interval,
                "outputsize": count,
                "apikey": TWELVE_DATA_API,
            },
            timeout=10,
        )
        data = resp.json()
        if "values" not in data:
            return None
        df = pd.DataFrame(data["values"])
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        for c in ["Open", "High", "Low", "Close"]:
            df[c] = pd.to_numeric(df[c])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.warning(f"TwelveData error {symbol}: {e}")
        return None

def fetch_yfinance(symbol, interval, count=100):
    try:
        yf_map = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "60m", "4h": "1h"}
        df = yf.download(
            YFINANCE_SYMBOLS.get(symbol, symbol),
            period="5d",
            interval=yf_map.get(interval, "5m"),
            progress=False,
        )
        if df.empty:
            return None
        if len(df.columns) == 5:
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df.tail(count)
    except Exception as e:
        logger.warning(f"yfinance error {symbol}: {e}")
        return None

def get_candles(symbol, interval, count=100):
    df = fetch_twelve(symbol, interval, count)
    if df is not None and len(df) >= 20:
        return df
    df = fetch_yfinance(symbol, interval, count)
    if df is not None and len(df) >= 20:
        return df
    return None

# AI MODELS
def download_model(filename):
    try:
        url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/models/{filename}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            content = base64.b64decode(resp.json()["content"])
            path = f"/tmp/{filename}"
            with open(path, "wb") as f:
                f.write(content)
            return path
        return None
    except Exception as e:
        logger.warning(f"Model download error: {e}")
        return None

def load_models():
    global models, scalers, feature_cols
    logger.info("Loading AI models from GitHub...")
    for symbol in PAIRS:
        try:
            mp = download_model(f"{symbol}_model.pkl")
            sp = download_model(f"{symbol}_scaler.pkl")
            fp = download_model(f"{symbol}_features.pkl")
            if mp and sp and fp:
                models[symbol] = joblib.load(mp)
                scalers[symbol] = joblib.load(sp)
                feature_cols[symbol] = joblib.load(fp)
                logger.info(f"✅ {symbol} model loaded")
        except Exception as e:
            logger.error(f"Model load error {symbol}: {e}")

# INDICATORS
def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    df["RSI"] = ta.momentum.RSIIndicator(close, 14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Mid"] = bb.bollinger_mavg()
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["EMA_21"] = ta.trend.EMAIndicator(close, 21).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(close, 50).ema_indicator()
    df["EMA_200"] = ta.trend.EMAIndicator(close, 200).ema_indicator()
    adx = ta.trend.ADXIndicator(high, low, close)
    df["ADX"] = adx.adx()
    df["ADX_pos"] = adx.adx_pos()
    df["ADX_neg"] = adx.adx_neg()
    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df["Candle_Body"] = abs(close - df["Open"])
    df["Upper_Wick"] = high - df[["Open", "Close"]].max(axis=1)
    df["Lower_Wick"] = df[["Open", "Close"]].min(axis=1) - low
    df["Is_Bullish"] = (close > df["Open"]).astype(int)
    df["Return_1"] = close.pct_change(1)
    df["Return_3"] = close.pct_change(3)
    df["Return_5"] = close.pct_change(5)
    return df

def add_smc(df):
    df = df.copy()
    bull_fvg, bear_fvg = [], []
    bull_ob, bear_ob = [0, 0], [0, 0]
    for i in range(2, len(df)):
        c1l = df["Low"].iloc[i - 2]
        c1h = df["High"].iloc[i - 2]
        c3h = df["High"].iloc[i]
        c3l = df["Low"].iloc[i]
        bull_fvg.append(c1l - c3h if c1l > c3h else 0)
        bear_fvg.append(c3l - c1h if c1h < c3l else 0)
        pb = abs(df["Close"].iloc[i - 1] - df["Open"].iloc[i - 1])
        cb = abs(df["Close"].iloc[i] - df["Open"].iloc[i])
        bull_ob.append(1 if df["Close"].iloc[i - 1] < df["Open"].iloc[i - 1] and df["Close"].iloc[i] > df["Open"].iloc[i] and cb > pb * 1.5 else 0)
        bear_ob.append(1 if df["Close"].iloc[i - 1] > df["Open"].iloc[i - 1] and df["Close"].iloc[i] < df["Open"].iloc[i] and cb > pb * 1.5 else 0)
    df["Bullish_FVG"] = [0, 0] + bull_fvg
    df["Bearish_FVG"] = [0, 0] + bear_fvg
    df["FVG_Present"] = ((df["Bullish_FVG"] > 0) | (df["Bearish_FVG"] > 0)).astype(int)
    df["Bullish_OB"] = bull_ob
    df["Bearish_OB"] = bear_ob
    df["Prev_High_20"] = df["High"].rolling(20).max().shift(1)
    df["Prev_Low_20"] = df["Low"].rolling(20).min().shift(1)
    df["Bullish_BOS"] = (df["Close"] > df["Prev_High_20"]).astype(int)
    df["Bearish_BOS"] = (df["Close"] < df["Prev_Low_20"]).astype(int)
    df["Swing_High_10"] = df["High"].rolling(10).max().shift(1)
    df["Swing_Low_10"] = df["Low"].rolling(10).min().shift(1)
    df["Liq_Grab_High"] = ((df["High"] > df["Swing_High_10"]) & (df["Close"] < df["Swing_High_10"])).astype(int)
    df["Liq_Grab_Low"] = ((df["Low"] < df["Swing_Low_10"]) & (df["Close"] > df["Swing_Low_10"])).astype(int)
    df["Dist_To_High_50"] = df["High"].rolling(50).max() - df["Close"]
    df["Dist_To_Low_50"] = df["Close"] - df["Low"].rolling(50).min()
    df["HH"] = (df["High"] > df["High"].shift(1)).astype(int)
    df["LL"] = (df["Low"] < df["Low"].shift(1)).astype(int)
    df["HL"] = (df["Low"] > df["Low"].shift(2)).astype(int)
    df["LH"] = (df["High"] < df["High"].shift(2)).astype(int)
    df.drop(columns=["Prev_High_20", "Prev_Low_20", "Swing_High_10", "Swing_Low_10"], inplace=True, errors="ignore")
    return df

# SESSION
def utc_now():
    return datetime.now(UTC)

def wat_now():
    return datetime.now(WAT)

def is_weekend():
    now = utc_now()
    wd = now.weekday()
    tm = now.hour * 60 + now.minute
    if wd == 4 and tm >= 21 * 60:
        return True
    if wd == 5:
        return True
    if wd == 6 and tm < 22 * 60:
        return True
    return False

def get_bot_mode():
    if is_weekend():
        return "sleep"
    now = utc_now()
    tm = now.hour * 60 + now.minute
    if tm >= 18 * 60 or tm < 2 * 60:
        return "sleep"
    if tm < 5 * 60 + 30:
        return "passive"
    return "active"

def get_sessions():
    h = utc_now().hour
    s = []
    if 0 <= h < 8:
        s.append("asian")
    if 7 <= h < 16:
        s.append("london")
    if 12 <= h < 21:
        s.append("new_york")
    if 12 <= h < 16:
        s.append("overlap")
    return s

# RISK
def calculate_risk(symbol, direction, entry, atr):
    pip = PIP_SIZE.get(symbol, 0.0001)
    risk_usd = min(10.0 * RISK_PERCENT.get(symbol, 2.0) / 100, 0.20)
    sl_dist = atr * (1.2 if symbol == "XAUUSD" else 1.5)
    sl_dist = max(sl_dist, pip * 5)
    if direction == "buy":
        sl = entry - sl_dist
        tp1 = entry + sl_dist * TP1_RR
        tp2 = entry + sl_dist * TP2_RR
        tp3 = entry + sl_dist * TP3_RR
    else:
        sl = entry + sl_dist
        tp1 = entry - sl_dist * TP1_RR
        tp2 = entry - sl_dist * TP2_RR
        tp3 = entry - sl_dist * TP3_RR
    sl_pips = sl_dist / pip
    lot_size = max(round(min(risk_usd / (sl_pips * 10 * 0.01), 0.10), 2), 0.01)
    digits = 2 if symbol == "XAUUSD" else 5
    return {
        "sl": round(sl, digits),
        "tp1": round(tp1, digits),
        "tp2": round(tp2, digits),
        "tp3": round(tp3, digits),
        "sl_pips": round(sl_pips, 1),
        "lot_size": lot_size,
        "risk_usd": round(risk_usd, 2),
    }

# ANALYSIS
def get_htf_bias(symbol):
    df = get_candles(symbol, "1h", 100)
    if df is None:
        return "neutral"
    df = add_indicators(df)
    df.dropna(inplace=True)
    if len(df) < 10:
        return "neutral"
    last = df.iloc[-1]
    close = float(last["Close"])
    e21 = float(last["EMA_21"])
    e50 = float(last["EMA_50"])
    if close > e21 > e50:
        return "bullish"
    if close < e21 < e50:
        return "bearish"
    return "neutral"

def detect_pattern(df):
    if len(df) < 3:
        return "none"
    c = df.iloc[-1]
    o, h, l, cl = c["Open"], c["High"], c["Low"], c["Close"]
    body = abs(cl - o)
    rng = h - l + 1e-10
    lower = min(o, cl) - l
    upper = h - max(o, cl)
    if body / rng < 0.1:
        return "doji"
    if lower > body * 2 and cl > o:
        return "hammer_bullish"
    if upper > body * 2 and cl < o:
        return "shooting_star_bearish"
    if upper > rng * 0.6 and cl < o:
        return "pin_bar_bearish"
    if lower > rng * 0.6 and cl > o:
        return "pin_bar_bullish"
    if cl > o and body / rng > 0.85:
        return "marubozu_bullish"
    if cl < o and body / rng > 0.85:
        return "marubozu_bearish"
    c1 = df.iloc[-2]
    if c1["Close"] < c1["Open"] and cl > o and o < c1["Close"] and cl > c1["Open"]:
        return "bullish_engulfing"
    if c1["Close"] > c1["Open"] and cl < o and o > c1["Close"] and cl < c1["Open"]:
        return "bearish_engulfing"
    return "none"

def get_ai_prediction(symbol, df):
    try:
        if symbol not in models:
            return None, 0
        feats = feature_cols[symbol]
        miss = [f for f in feats if f not in df.columns]
        if miss:
            return None, 0
        X = df[feats].iloc[-1:].values
        X_sc = scalers[symbol].transform(X)
        prob = models[symbol].predict_proba(X_sc)[0]
        conf = float(prob.max())
        pred = models[symbol].predict(X_sc)[0]
        return ("buy" if pred == 1 else "sell"), conf
    except Exception as e:
        logger.warning(f"AI error {symbol}: {e}")
        return None, 0

def get_news(symbol):
    try:
        kw = {
            "EURUSD": "euro dollar",
            "GBPUSD": "pound dollar",
            "USDJPY": "yen dollar",
            "XAUUSD": "gold price",
        }
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": kw.get(symbol, "forex"), "apiKey": NEWS_API_KEY, "language": "en", "pageSize": 5},
            timeout=8,
        )
        data = resp.json()
        if data.get("status") != "ok":
            return {"safe": True, "sentiment": "neutral"}
        headlines = [a["title"].lower() for a in data.get("articles", [])]
        high_impact = any(w in h for h in headlines for w in ["rate decision", "nfp", "cpi", "fomc"])
        bull = sum(w in h for h in headlines for w in ["rise", "rally", "surge", "bullish"])
        bear = sum(w in h for h in headlines for w in ["fall", "drop", "plunge", "bearish"])
        return {
            "safe": not high_impact,
            "sentiment": "bullish" if bull > bear + 1 else "bearish" if bear > bull + 1 else "neutral",
        }
    except:
        return {"safe": True, "sentiment": "neutral"}

async def generate_signal(symbol):
    global setup_counter
    mode = get_bot_mode()
    sessions = get_sessions()
    min_conf = MIN_CONFIDENCE_PASSIVE if mode == "passive" else MIN_CONFIDENCE_ACTIVE

    if symbol in active_trades:
        return None

    if symbol == "XAUUSD" and sessions == ["asian"]:
        df_c = get_candles(symbol, "1h", 20)
        if df_c is not None:
            df_c = add_indicators(df_c)
            df_c.dropna(inplace=True)
            if len(df_c) > 1:
                if float(df_c["ATR"].iloc[-1]) < float(df_c["ATR"].mean()) * 1.5:
                    return None

    df = get_candles(symbol, "5min", 150)
    if df is None or len(df) < 50:
        return None
    df = add_indicators(df)
    df = add_smc(df)
    df.dropna(inplace=True)
    if len(df) < 10:
        return None

    last = df.iloc[-1]
    atr = float(last.get("ATR", 0.001))
    adx = float(last.get("ADX", 0))
    if adx < 15:
        return None

    news = get_news(symbol)
    if not news["safe"] and mode != "passive":
        return None

    bias = get_htf_bias(symbol)
    ai_dir, ai_conf = get_ai_prediction(symbol, df)

    bull, bear = 0, 0
    if bias == "bullish":
        bull += 3
    elif bias == "bearish":
        bear += 3
    rsi = float(last.get("RSI", 50))
    macd_h = float(last.get("MACD_Hist", 0))
    if rsi < 40:
        bull += 1
    elif rsi > 60:
        bear += 1
    if macd_h > 0:
        bull += 1
    elif macd_h < 0:
        bear += 1
    if float(last.get("Bullish_BOS", 0)) > 0:
        bull += 2
    if float(last.get("Bearish_BOS", 0)) > 0:
        bear += 2
    if float(last.get("Bullish_FVG", 0)) > 0:
        bull += 1
    if float(last.get("Bearish_FVG", 0)) > 0:
        bear += 1
    if float(last.get("Liq_Grab_Low", 0)) > 0:
        bull += 2
    if float(last.get("Liq_Grab_High", 0)) > 0:
        bear += 2
    if float(last.get("Bullish_OB", 0)) > 0:
        bull += 1
    if float(last.get("Bearish_OB", 0)) > 0:
        bear += 1
    if news["sentiment"] == "bullish":
        bull += 1
    elif news["sentiment"] == "bearish":
        bear += 1

    direction = None
    if bull > bear + 1:
        direction = "buy"
    elif bear > bull + 1:
        direction = "sell"
    if direction is None:
        return None

    if ai_dir and ai_dir != direction:
        return None
    ai_boost = 15 if ai_dir == direction and ai_conf >= AI_CONFIDENCE_THRESH else 0
    base_score = min((max(bull, bear) / (bull + bear + 1e-10)) * 85, 85)
    confidence = min(base_score + ai_boost, 100)
    if confidence < min_conf:
        return None

    atr_rng = float(last["High"] - last["Low"])
    fast_mode = rsi >= 80 or rsi <= 20 or (atr > 0 and atr_rng > atr * 2)
    pattern = detect_pattern(df)

    bullish_p = ["hammer_bullish", "bullish_engulfing", "pin_bar_bullish", "marubozu_bullish"]
    bearish_p = ["shooting_star_bearish", "bearish_engulfing", "pin_bar_bearish", "marubozu_bearish"]
    if confidence < 75 and not fast_mode:
        if direction == "buy" and pattern not in bullish_p:
            confidence = confidence * 0.9
        if direction == "sell" and pattern not in bearish_p:
            confidence = confidence * 0.9

    entry = float(last["Close"])
    risk = calculate_risk(symbol, direction, entry, atr)
    sess_str = "+".join(s.title() for s in sessions) if sessions else "None"

    reasons = [f"HTF:{bias.upper()}"]
    if adx >= ADX_THRESHOLD:
        reasons.append(f"ADX:{adx:.0f}✅")
    if float(last.get("Bullish_BOS", 0)) > 0 or float(last.get("Bearish_BOS", 0)) > 0:
        reasons.append("BOS✅")
    if float(last.get("FVG_Present", 0)) > 0:
        reasons.append("FVG✅")
    if float(last.get("Liq_Grab_Low", 0)) > 0 or float(last.get("Liq_Grab_High", 0)) > 0:
        reasons.append("LiqGrab✅")
    if pattern != "none":
        reasons.append(pattern.replace("_", " ").title())
    if fast_mode:
        reasons.append("FastExec")
    if ai_dir:
        reasons.append(f"AI:{ai_conf*100:.0f}%")

    setup_counter += 1
    return {
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "sl": risk["sl"],
        "tp1": risk["tp1"],
        "tp2": risk["tp2"],
        "tp3": risk["tp3"],
        "sl_pips": risk["sl_pips"],
        "lot_size": risk["lot_size"],
        "risk_usd": risk["risk_usd"],
        "confidence": confidence,
        "pattern": pattern,
        "bias": bias,
        "fast_mode": fast_mode,
        "session": sess_str,
        "reason": " | ".join(reasons),
        "tp1_hit": False,
        "tp2_hit": False,
        "tp3_hit": False,
        "time": time.time(),
        "setup_num": setup_counter,
    }

def format_signal(sig):
    d = "BUY" if sig["direction"] == "buy" else "SELL"
    arrow = "📈" if sig["direction"] == "buy" else "📉"
    color = "🟢" if sig["direction"] == "buy" else "🔴"
    ft = "FAST EXECUTION\n" if sig["fast_mode"] else ""
    bar = "█" * int(sig["confidence"] / 10) + "░" * (10 - int(sig["confidence"] / 10))
    msg = "━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"NEW SIGNAL — {sig['symbol']}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += ft
    msg += f"{color} {d} {arrow}\n"
    msg += f"SCALPER | M5 | {sig['session']}\n\n"
    msg += f"Entry:  {sig['entry']}\n"
    msg += f"SL:     {sig['sl']} ({sig['sl_pips']} pips)\n"
    msg += f"TP1 (1:{TP1_RR}): {sig['tp1']}\n"
    msg += f"TP2 (1:{TP2_RR}): {sig['tp2']}\n"
    msg += f"TP3 (1:{TP3_RR}): {sig['tp3']}\n\n"
    msg += f"Lot: {sig['lot_size']} | Risk: ${sig['risk_usd']}\n"
    msg += f"Bias: {sig['bias'].upper()}\n"
    msg += f"Pattern: {sig['pattern'].replace('_', ' ').title()}\n\n"
    msg += f"Confidence: {sig['confidence']:.0f}%\n"
    msg += f"[{bar}]\n\n"
    msg += f"{sig['reason']}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "Always manage your risk!"
    return msg

# TRADE MONITOR
async def monitor_trades(bot):
    for symbol, sig in list(active_trades.items()):
        try:
            df = get_candles(symbol, "1min", 5)
            if df is None:
                continue
            price = float(df["Close"].iloc[-1])
            d = sig["direction"]

            if not sig["tp1_hit"]:
                if (d == "buy" and price >= sig["tp1"]) or (d == "sell" and price <= sig["tp1"]):
                    sig["tp1_hit"] = True
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text=f"TP1 HIT! — {symbol}\nTP1: {sig['tp1']}\nTP2: {sig['tp2']}\nMove SL to breakeven!",
                    )

            if sig["tp1_hit"] and not sig["tp2_hit"]:
                if (d == "buy" and price >= sig["tp2"]) or (d == "sell" and price <= sig["tp2"]):
                    sig["tp2_hit"] = True
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text=f"TP2 HIT! — {symbol}\nTP2: {sig['tp2']}\nTP3: {sig['tp3']}\nTrail your SL!",
                    )

            if sig["tp2_hit"] and not sig["tp3_hit"]:
                if (d == "buy" and price >= sig["tp3"]) or (d == "sell" and price <= sig["tp3"]):
                    sig["tp3_hit"] = True
                    signal_stats[symbol]["wins"] += 1
                    pips = round(sig["sl_pips"] * TP3_RR, 1)
                    daily_trades.append({
                        "setup": sig["setup_num"],
                        "symbol": symbol,
                        "direction": d,
                        "entry": sig["entry"],
                        "pips": pips,
                        "result": "tp",
                        "session": sig["session"],
                    })
                    del active_trades[symbol]
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text=f"FULL TARGET! — {symbol}\nALL TPs HIT! +{pips} pips\n{symbol} ready for next signal!",
                    )

            if (d == "buy" and price <= sig["sl"]) or (d == "sell" and price >= sig["sl"]):
                signal_stats[symbol]["losses"] += 1
                pips = round(sig["sl_pips"], 1)
                daily_trades.append({
                    "setup": sig["setup_num"],
                    "symbol": symbol,
                    "direction": d,
                    "entry": sig["entry"],
                    "pips": pips,
                    "result": "sl",
                    "session": sig["session"],
                })
                del active_trades[symbol]
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"SL HIT — {symbol}\n-{pips} pips\nLoss: -${sig['risk_usd']}\nAccount protected!\n{symbol} ready for next signal!",
                )
        except Exception as e:
            logger.error(f"Monitor error {symbol}: {e}")

# DAILY SUMMARY
async def send_daily_summary(bot):
    global daily_trades, setup_counter
    if not daily_trades:
        await bot.send_message(chat_id=CHAT_ID, text="No completed trades today yet!")
        return

    session_labels = {
        "asian": "🟨 Asian Session",
        "london": "🟩 European Session",
        "new_york": "🟥 US Session",
        "overlap": "🟦 Overlap Session",
        "none": "⬜ Other",
    }
    sessions_order = ["asian", "london", "new_york", "overlap", "none"]

    wins = sum(1 for t in daily_trades if t["result"] == "tp")
    losses = sum(1 for t in daily_trades if t["result"] == "sl")
    cancelled = sum(1 for t in daily_trades if t["result"] == "cancelled")
    total = len(daily_trades)
    total_pips = sum(t["pips"] if t["result"] == "tp" else -t["pips"] if t["result"] == "sl" else 0 for t in daily_trades)

    grouped = {}
    for t in daily_trades:
        raw = t["session"].lower().split("+")[0].strip()
        sess = raw if raw in session_labels else "none"
        grouped.setdefault(sess, []).append(t)

    today = wat_now().strftime("%d %b %Y")
    msg = f"EARNINGS OF DAY — {today}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    for sess_key in sessions_order:
        if sess_key not in grouped:
            continue
        msg += f"{session_labels[sess_key]}\n\n"
        for t in grouped[sess_key]:
            d = "BUY" if t["direction"] == "buy" else "SELL"
            pip = PIP_SIZE.get(t["symbol"], 0.0001)
            entry1 = t["entry"]
            entry2 = round(entry1 + pip * 2, 5)
            if t["result"] == "tp":
                status = f"TP +{t['pips']} pips (reached) 🟢"
            elif t["result"] == "sl":
                status = f"SL -{t['pips']} pips (hit) 🔴"
            else:
                status = "Cancelled 🟡"
            msg += f"• Setup {t['setup']}: {t['symbol']} {d} {entry1}-{entry2} | {status}\n"
        msg += "\n"

    sign = "+" if total_pips >= 0 else ""
    result_w = "PROFIT" if total_pips >= 0 else "LOSS"
    lot_gain = abs(total_pips) * 1.0

    msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"Total: {total} setups\n"
    msg += f"🟢 {wins} successful\n"
    msg += f"🔴 {losses} stop-loss\n"
    if cancelled > 0:
        msg += f"🟡 {cancelled} cancelled\n"
    msg += f"\nFINAL RESULT: {sign}{total_pips:.0f} PIPS ({result_w})\n\n"
    msg += f"With 0.1 lot per setup, you could {'gain' if total_pips >= 0 else 'lose'} around ${lot_gain:.0f}."

    await bot.send_message(chat_id=CHAT_ID, text=msg)
    daily_trades = []
    setup_counter = 0
    logger.info("Daily summary sent!")

# COMMANDS
def auth(func):
    async def wrapper(update, ctx):
        if update.effective_user.id not in AUTHORIZED_USERS:
            await update.message.reply_text("Unauthorized.")
            return
        return await func(update, ctx)
    return wrapper

@auth
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "OLASMOS FX BOT\n\n"
        "/status  — Bot status\n"
        "/pairs   — Monitored pairs\n"
        "/scan    — Force scan now\n"
        "/stats   — Win/Loss stats\n"
        "/risk    — Risk settings\n"
        "/session — Session info\n"
        "/summary — Daily summary\n"
        "/pause   — Pause signals\n"
        "/resume  — Resume signals\n"
        "/help    — Show commands"
    )

@auth
async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    mode = get_bot_mode()
    sess = ", ".join(s.title() for s in get_sessions()) or "None"
    await update.message.reply_text(
        f"BOT STATUS\n"
        f"Mode: {mode.upper()}\n"
        f"WAT: {wat_now().strftime('%H:%M %d/%m/%Y')}\n"
        f"Sessions: {sess}\n"
        f"{'PAUSED' if bot_paused else 'RUNNING'}\n"
        f"Models: {len(models)}/4\n"
        f"Active trades: {len(active_trades)}\n"
        f"Today trades: {len(daily_trades)}\n"
        f"Weekend: {'Yes' if is_weekend() else 'No'}"
    )

@auth
async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lines = ["Statistics"]
    tw, tl = 0, 0
    for p, s in signal_stats.items():
        t = s["wins"] + s["losses"]
        wr = (s["wins"] / t * 100) if t > 0 else 0
        lines.append(f"{p}: {s['wins']}W/{s['losses']}L ({wr:.0f}%)")
        tw += s["wins"]
        tl += s["losses"]
    tot = tw + tl
    lines.append(f"Total: {tw}W/{tl}L ({(tw/tot*100) if tot>0 else 0:.0f}%)")
    await update.message.reply_text("\n".join(lines))

@auth
async def cmd_risk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lines = ["Risk Settings"]
    for p in PAIRS:
        r = RISK_PERCENT[p]
        lines.append(f"{p}: {r}% = ${10*r/100:.2f}")
    lines.append(f"TP1:1:{TP1_RR} TP2:1:{TP2_RR} TP3:1:{TP3_RR}")
    await update.message.reply_text("\n".join(lines))

@auth
async def cmd_session(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sess = ", ".join(s.title() for s in get_sessions()) or "None"
    mode = get_bot_mode()
    await update.message.reply_text(
        f"Session Info\n"
        f"WAT: {wat_now().strftime('%H:%M')}\n"
        f"Sessions: {sess}\n"
        f"Mode: {mode.upper()}\n"
        f"Weekend: {'Yes' if is_weekend() else 'No'}"
    )

@auth
async def cmd_pairs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Pairs\n" + "".join(f"• {p}\n" for p in PAIRS) + "\nEntry: M1,M5,M15 | HTF: H1,H4")

@auth
async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating daily summary...")
    await send_daily_summary(update.get_bot())

@auth
async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global bot_paused
    bot_paused = True
    await update.message.reply_text("Paused. /resume to restart.")

@auth
async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global bot_paused
    bot_paused = False
    await update.message.reply_text("Resumed!")

@auth
async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Scanning all pairs...")
    found = 0
    for symbol in PAIRS:
        sig = await generate_signal(symbol)
        if sig:
            active_trades[symbol] = sig
            signal_stats[symbol]["total"] += 1
            await update.message.reply_text(format_signal(sig))
            found += 1
    if found == 0:
        await update.message.reply_text("Scan done — no signals right now.")

@auth
async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, ctx)
    
 @auth
async def cmd_debug(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Running debug scan...")
    for symbol in PAIRS:
        try:
            df = get_candles(symbol, "5min", 150)
            if df is None:
                await update.message.reply_text(f"{symbol}: ❌ No data")
                continue
            df = add_indicators(df)
            df = add_smc(df)
            df.dropna(inplace=True)
            if len(df) < 10:
                await update.message.reply_text(f"{symbol}: ❌ Not enough data")
                continue
            last = df.iloc[-1]
            adx  = float(last.get("ADX", 0))
            rsi  = float(last.get("RSI", 50))
            macd_h = float(last.get("MACD_Hist", 0))
            bias = get_htf_bias(symbol)
            bull, bear = 0, 0
            if bias == "bullish": bull += 3
            elif bias == "bearish": bear += 3
            if rsi < 40: bull += 1
            elif rsi > 60: bear += 1
            if macd_h > 0: bull += 1
            elif macd_h < 0: bear += 1
            if float(last.get("Bullish_BOS", 0)) > 0: bull += 2
            if float(last.get("Bearish_BOS", 0)) > 0: bear += 2
            if float(last.get("Bullish_FVG", 0)) > 0: bull += 1
            if float(last.get("Bearish_FVG", 0)) > 0: bear += 1
            if float(last.get("Liq_Grab_Low", 0)) > 0: bull += 2
            if float(last.get("Liq_Grab_High", 0)) > 0: bear += 2
            await update.message.reply_text(
                f"📊 {symbol}\n"
                f"ADX: {adx:.1f} (min 15)\n"
                f"RSI: {rsi:.1f}\n"
                f"MACD: {'▲' if macd_h > 0 else '▼'}\n"
                f"Bias: {bias}\n"
                f"Bull score: {bull}\n"
                f"Bear score: {bear}\n"
                f"Direction: {'BUY' if bull > bear+1 else 'SELL' if bear > bull+1 else 'NONE'}"
            )
        except Exception as e:
            await update.message.reply_text(f"{symbol}: ❌ Error: {str(e)[:50]}")   

# SCAN LOOP
async def scan_loop(bot):
    global _last_weekend
    load_models()

    await bot.send_message(
        chat_id=CHAT_ID,
        text=(
            "OLASMOS FX BOT STARTED!\n"
            f"Pairs: {', '.join(PAIRS)}\n"
            f"AI Models: {str(len(models))}/4 loaded\n"
            "Scalper mode active\n\n"
            "Schedule (Nigerian WAT):\n"
            "Active:  6:30AM-7:00PM\n"
            "Passive: 3:00AM-6:30AM\n"
            "Sleep:   7:00PM-3:00AM\n\n"
            "Type /summary for daily report!"
        ),
    )

    while True:
        try:
            wknd = is_weekend()
            if wknd != _last_weekend:
                if wknd:
                    await bot.send_message(chat_id=CHAT_ID, text="MARKET CLOSED\nWeekend sleep mode.\nSending daily summary...")
                    await send_daily_summary(bot)
                else:
                    await bot.send_message(chat_id=CHAT_ID, text="MARKET OPEN!\nBot resuming!")
                _last_weekend = wknd

            mode = get_bot_mode()
            if active_trades:
                await monitor_trades(bot)

            if mode == "sleep" or bot_paused:
                await asyncio.sleep(60)
                continue

            interval = SCAN_INTERVAL_PASSIVE if mode == "passive" else SCAN_INTERVAL_ACTIVE

            now_utc = utc_now()
            if now_utc.hour == 18 and now_utc.minute == 0 and daily_trades:
                await send_daily_summary(bot)

            for symbol in PAIRS:
                if symbol in active_trades:
                    continue
                try:
                    sig = await generate_signal(symbol)
                    if sig:
                        active_trades[symbol] = sig
                        signal_stats[symbol]["total"] += 1
                        await bot.send_message(chat_id=CHAT_ID, text=format_signal(sig))
                        logger.info(f"Signal: {symbol} {sig['direction']} {sig['confidence']:.0f}%")
                        await asyncio.sleep(3)
                except Exception as e:
                    logger.error(f"Scan error {symbol}: {e}")

            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(30)

# POST INIT
async def post_init(app: Application):
    asyncio.create_task(scan_loop(app.bot))
    logger.info("Scan loop started!")

# MAIN
def main():
    Thread(target=run_flask, daemon=True).start()
    logger.info(f"Flask on port {PORT}")

    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pairs", cmd_pairs))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("session", cmd_session))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("help", cmd_help))

    logger.info("Starting bot...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
