=========================================
# OLASMOS FX BOT — MAIN FILE
# Full SMC + AI Signal Bot
# Runs 24/7 on Render with webhook
# ============================================

import asyncio
import logging
import os
import time
import pytz
import joblib
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from threading import Thread
import ta
import base64

from telegram import Update, Bot
from telegram.ext import (Application, CommandHandler,
                           ContextTypes, MessageHandler, filters)
from flask import Flask, request as flask_request

import config
from keep_alive import keep_alive

# ─── LOGGING ─────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("olasmos_fx")

# ─── TIMEZONE ────────────────────────────────
WAT = pytz.timezone("Africa/Lagos")
UTC = pytz.utc

# ─── GLOBAL STATE ────────────────────────────
bot_paused        = False
active_trades     = {}   # symbol → signal details
signal_stats      = {p: {"wins":0,"losses":0,"total":0} for p in config.PAIRS}
models            = {}
scalers           = {}
feature_cols      = {}
_last_weekend     = None

# ============================================
# DATA FETCHING — Auto fallback chain
# Twelve Data → Alpha Vantage → yfinance
# ============================================

def fetch_twelve_data(symbol: str, interval: str, count: int = 100):
    try:
        td_sym = config.TWELVE_SYMBOLS.get(symbol)
        url    = "https://api.twelvedata.com/time_series"
        params = {
            "symbol":     td_sym,
            "interval":   interval,
            "outputsize": count,
            "apikey":     config.TWELVE_DATA_API,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if "values" not in data:
            return None
        df = pd.DataFrame(data["values"])
        df.rename(columns={"open":"Open","high":"High",
                            "low":"Low","close":"Close",
                            "volume":"Volume"}, inplace=True)
        for col in ["Open","High","Low","Close"]:
            df[col] = pd.to_numeric(df[col])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.warning(f"Twelve Data error {symbol}: {e}")
        return None


def fetch_alpha_vantage(symbol: str, interval: str, count: int = 100):
    try:
        av_map = {
            "1min":"1min","5min":"5min",
            "15min":"15min","30min":"30min",
            "1h":"60min","4h":"60min",
        }
        av_interval = av_map.get(interval, "5min")
        url    = "https://www.alphavantage.co/query"
        params = {
            "function":   "FX_INTRADAY",
            "from_symbol": symbol[:3],
            "to_symbol":   symbol[3:],
            "interval":   av_interval,
            "outputsize": "compact",
            "apikey":     config.ALPHA_VANTAGE_API,
        }
        if symbol == "XAUUSD":
            params["function"]    = "TIME_SERIES_INTRADAY"
            params["symbol"]      = "XAUUSD"
            params.pop("from_symbol", None)
            params.pop("to_symbol",   None)
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        key  = [k for k in data.keys() if "Time Series" in k]
        if not key:
            return None
        df = pd.DataFrame(data[key[0]]).T
        df.rename(columns={
            "1. open":"Open","2. high":"High",
            "3. low":"Low","4. close":"Close"
        }, inplace=True)
        for col in ["Open","High","Low","Close"]:
            df[col] = pd.to_numeric(df[col])
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df.tail(count)
    except Exception as e:
        logger.warning(f"Alpha Vantage error {symbol}: {e}")
        return None


def fetch_yfinance(symbol: str, interval: str, count: int = 100):
    try:
        yf_map = {
            "1min":"1m","5min":"5m","15min":"15m",
            "30min":"30m","1h":"60m","4h":"1h",
        }
        yf_interval = yf_map.get(interval, "5m")
        yf_symbol   = config.YFINANCE_SYMBOLS.get(symbol, symbol)
        df = yf.download(yf_symbol, period="5d",
                         interval=yf_interval, progress=False)
        if df.empty:
            return None
        df.columns = ["Open","High","Low","Close","Volume"] \
            if len(df.columns) == 5 else df.columns
        return df.tail(count)
    except Exception as e:
        logger.warning(f"yfinance error {symbol}: {e}")
        return None


def get_candles(symbol: str, interval: str, count: int = 100):
    """Auto-fallback: Twelve Data → Alpha Vantage → yfinance"""
    df = fetch_twelve_data(symbol, interval, count)
    if df is not None and len(df) >= 20:
        return df
    df = fetch_alpha_vantage(symbol, interval, count)
    if df is not None and len(df) >= 20:
        return df
    df = fetch_yfinance(symbol, interval, count)
    if df is not None and len(df) >= 20:
        return df
    logger.error(f"All data sources failed for {symbol} {interval}")
    return None


# ============================================
# LOAD AI MODELS FROM GITHUB
# ============================================

def download_model_from_github(filename: str):
    url     = f"https://api.github.com/repos/{config.GITHUB_USERNAME}/{config.GITHUB_REPO}/contents/models/{filename}"
    headers = {"Authorization": f"token {config.GITHUB_TOKEN}",
               "Accept": "application/vnd.github.v3+json"}
    resp    = requests.get(url, headers=headers)
    if resp.status_code == 200:
        content = base64.b64decode(resp.json()["content"])
        path    = f"/tmp/{filename}"
        with open(path, "wb") as f:
            f.write(content)
        return path
    return None


def load_models():
    global models, scalers, feature_cols
    logger.info("Loading AI models from GitHub...")
    for symbol in config.PAIRS:
        try:
            mp = download_model_from_github(f"{symbol}_model.pkl")
            sp = download_model_from_github(f"{symbol}_scaler.pkl")
            fp = download_model_from_github(f"{symbol}_features.pkl")
            if mp and sp and fp:
                models[symbol]      = joblib.load(mp)
                scalers[symbol]     = joblib.load(sp)
                feature_cols[symbol] = joblib.load(fp)
                logger.info(f"✅ {symbol} model loaded")
            else:
                logger.warning(f"⚠️ Could not load {symbol} model")
        except Exception as e:
            logger.error(f"Model load error {symbol}: {e}")


# ============================================
# INDICATORS
# ============================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    df["RSI"]         = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd              = ta.trend.MACD(close)
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"]   = macd.macd_diff()
    bb                = ta.volatility.BollingerBands(close)
    df["BB_Upper"]    = bb.bollinger_hband()
    df["BB_Lower"]    = bb.bollinger_lband()
    df["BB_Mid"]      = bb.bollinger_mavg()
    df["BB_Width"]    = df["BB_Upper"] - df["BB_Lower"]
    df["EMA_21"]      = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df["EMA_50"]      = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["EMA_200"]     = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    adx               = ta.trend.ADXIndicator(high, low, close)
    df["ADX"]         = adx.adx()
    df["ADX_pos"]     = adx.adx_pos()
    df["ADX_neg"]     = adx.adx_neg()
    df["ATR"]         = ta.volatility.AverageTrueRange(
                            high, low, close).average_true_range()
    df["Candle_Body"] = abs(close - df["Open"])
    df["Upper_Wick"]  = high - df[["Open","Close"]].max(axis=1)
    df["Lower_Wick"]  = df[["Open","Close"]].min(axis=1) - low
    df["Is_Bullish"]  = (close > df["Open"]).astype(int)
    df["Return_1"]    = close.pct_change(1)
    df["Return_3"]    = close.pct_change(3)
    df["Return_5"]    = close.pct_change(5)
    return df


def add_smc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bull_fvg, bear_fvg = [], []
    bull_ob,  bear_ob  = [0,0], [0,0]

    for i in range(2, len(df)):
        c1l = df["Low"].iloc[i-2]
        c1h = df["High"].iloc[i-2]
        c3h = df["High"].iloc[i]
        c3l = df["Low"].iloc[i]
        bull_fvg.append(c1l - c3h if c1l > c3h else 0)
        bear_fvg.append(c3l - c1h if c1h < c3l else 0)
        pb = abs(df["Close"].iloc[i-1] - df["Open"].iloc[i-1])
        cb = abs(df["Close"].iloc[i]   - df["Open"].iloc[i])
        bull_ob.append(1 if (df["Close"].iloc[i-1] < df["Open"].iloc[i-1]
                         and df["Close"].iloc[i]   > df["Open"].iloc[i]
                         and cb > pb * 1.5) else 0)
        bear_ob.append(1 if (df["Close"].iloc[i-1] > df["Open"].iloc[i-1]
                         and df["Close"].iloc[i]   < df["Open"].iloc[i]
                         and cb > pb * 1.5) else 0)

    df["Bullish_FVG"]     = [0,0] + bull_fvg
    df["Bearish_FVG"]     = [0,0] + bear_fvg
    df["FVG_Present"]     = ((df["Bullish_FVG"]>0)|(df["Bearish_FVG"]>0)).astype(int)
    df["Bullish_OB"]      = bull_ob
    df["Bearish_OB"]      = bear_ob
    df["Prev_High_20"]    = df["High"].rolling(20).max().shift(1)
    df["Prev_Low_20"]     = df["Low"].rolling(20).min().shift(1)
    df["Bullish_BOS"]     = (df["Close"] > df["Prev_High_20"]).astype(int)
    df["Bearish_BOS"]     = (df["Close"] < df["Prev_Low_20"]).astype(int)
    df["Swing_High_10"]   = df["High"].rolling(10).max().shift(1)
    df["Swing_Low_10"]    = df["Low"].rolling(10).min().shift(1)
    df["Liq_Grab_High"]   = ((df["High"] > df["Swing_High_10"]) &
                              (df["Close"] < df["Swing_High_10"])).astype(int)
    df["Liq_Grab_Low"]    = ((df["Low"]  < df["Swing_Low_10"])  &
                              (df["Close"] > df["Swing_Low_10"])).astype(int)
    df["Dist_To_High_50"] = df["High"].rolling(50).max() - df["Close"]
    df["Dist_To_Low_50"]  = df["Close"] - df["Low"].rolling(50).min()
    df["HH"] = (df["High"] > df["High"].shift(1)).astype(int)
    df["LL"] = (df["Low"]  < df["Low"].shift(1)).astype(int)
    df["HL"] = (df["Low"]  > df["Low"].shift(2)).astype(int)
    df["LH"] = (df["High"] < df["High"].shift(2)).astype(int)
    df.drop(columns=["Prev_High_20","Prev_Low_20",
                     "Swing_High_10","Swing_Low_10"],
            inplace=True, errors="ignore")
    return df


# ============================================
# SESSION MANAGER
# ============================================

def utc_now():
    return datetime.now(UTC)

def wat_now():
    return datetime.now(WAT)

def is_weekend():
    now = utc_now()
    wd  = now.weekday()
    tm  = now.hour * 60 + now.minute
    if wd == 4 and tm >= 21*60: return True
    if wd == 5:                  return True
    if wd == 6 and tm < 22*60:  return True
    return False

def get_bot_mode():
    if is_weekend(): return "sleep"
    now = utc_now()
    tm  = now.hour * 60 + now.minute
    if tm >= config.SLEEP_START_UTC*60 or tm < config.SLEEP_END_UTC*60:
        return "sleep"
    if tm < config.PASSIVE_END_UTC*60 + config.PASSIVE_END_MIN:
        return "passive"
    return "active"

def get_sessions():
    now = utc_now()
    h   = now.hour
    active = []
    if 0  <= h < 8:  active.append("asian")
    if 7  <= h < 16: active.append("london")
    if 12 <= h < 21: active.append("new_york")
    if 12 <= h < 16: active.append("overlap")
    return active

def is_high_liquidity():
    s = get_sessions()
    return any(x in s for x in ["london","new_york","overlap"])


# ============================================
# RISK MANAGER
# ============================================

def calculate_risk(symbol: str, direction: str,
                   entry: float, atr: float) -> dict:
    pip      = config.PIP_SIZE.get(symbol, 0.0001)
    balance  = 10.0   # default — no MT5 on Render
    risk_pct = config.RISK_PERCENT.get(symbol, 2.0) / 100
    risk_usd = min(balance * risk_pct, config.MAX_LOSS_PER_TRADE)

    # Gold tighter SL
    sl_atr_mult = 1.2 if symbol == "XAUUSD" else 1.5
    sl_dist     = atr * sl_atr_mult
    sl_dist     = max(sl_dist, pip * 5)

    if direction == "buy":
        sl  = entry - sl_dist
        tp1 = entry + sl_dist * config.TP1_RR
        tp2 = entry + sl_dist * config.TP2_RR
        tp3 = entry + sl_dist * config.TP3_RR
    else:
        sl  = entry + sl_dist
        tp1 = entry - sl_dist * config.TP1_RR
        tp2 = entry - sl_dist * config.TP2_RR
        tp3 = entry - sl_dist * config.TP3_RR

    sl_pips  = sl_dist / pip
    pip_val  = 10.0 if symbol != "XAUUSD" else 1.0
    lot_size = max(risk_usd / (sl_pips * pip_val * 0.01), config.MIN_LOT)
    lot_size = round(min(lot_size, 0.10), 2)

    digits = 2 if symbol == "XAUUSD" else 5
    return {
        "sl":       round(sl,  digits),
        "tp1":      round(tp1, digits),
        "tp2":      round(tp2, digits),
        "tp3":      round(tp3, digits),
        "sl_pips":  round(sl_pips, 1),
        "lot_size": lot_size,
        "risk_usd": round(risk_usd, 2),
    }


# ============================================
# SIGNAL ANALYSIS
# ============================================

def get_htf_bias(symbol: str) -> str:
    df = get_candles(symbol, "1h", 100)
    if df is None: return "neutral"
    df = add_indicators(df)
    df.dropna(inplace=True)
    if len(df) < 10: return "neutral"
    last  = df.iloc[-1]
    close = float(last["Close"])
    e21   = float(last["EMA_21"])
    e50   = float(last["EMA_50"])
    if close > e21 > e50: return "bullish"
    if close < e21 < e50: return "bearish"
    return "neutral"


def detect_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3: return "none"
    c = df.iloc[-1]
    o, h, l, cl = c["Open"], c["High"], c["Low"], c["Close"]
    body  = abs(cl - o)
    rng   = h - l + 1e-10
    lower = min(o,cl) - l
    upper = h - max(o,cl)
    if body/rng < 0.1:                          return "doji"
    if lower > body*2 and cl > o:               return "hammer_bullish"
    if upper > body*2 and cl < o:               return "shooting_star_bearish"
    if upper > rng*0.6 and cl < o:              return "pin_bar_bearish"
    if lower > rng*0.6 and cl > o:              return "pin_bar_bullish"
    if cl > o and body/rng > 0.85:              return "marubozu_bullish"
    if cl < o and body/rng > 0.85:              return "marubozu_bearish"
    c1 = df.iloc[-2]
    if (c1["Close"] < c1["Open"] and cl > o
            and o < c1["Close"] and cl > c1["Open"]):
        return "bullish_engulfing"
    if (c1["Close"] > c1["Open"] and cl < o
            and o > c1["Close"] and cl < c1["Open"]):
        return "bearish_engulfing"
    return "none"


def get_ai_prediction(symbol: str, df: pd.DataFrame):
    """Returns (direction, confidence) from AI model"""
    try:
        if symbol not in models: return None, 0
        feats   = feature_cols[symbol]
        missing = [f for f in feats if f not in df.columns]
        if missing: return None, 0
        X       = df[feats].iloc[-1:].values
        X_sc    = scalers[symbol].transform(X)
        proba   = models[symbol].predict_proba(X_sc)[0]
        conf    = float(proba.max())
        pred    = models[symbol].predict(X_sc)[0]
        direction = "buy" if pred == 1 else "sell"
        return direction, conf
    except Exception as e:
        logger.warning(f"AI prediction error {symbol}: {e}")
        return None, 0


def check_fast_execution(df: pd.DataFrame, atr: float) -> tuple:
    last = df.iloc[-1]
    rsi  = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50
    rng  = float(last["High"] - last["Low"])
    if rsi >= config.FAST_EXEC_RSI or rsi <= (100-config.FAST_EXEC_RSI):
        return True, "momentum_burst"
    if atr > 0 and rng > atr * config.FAST_EXEC_ATR_MULT:
        return True, "volatility_spike"
    return False, ""


def get_news_sentiment(symbol: str) -> dict:
    try:
        kw   = {"EURUSD":"euro dollar","GBPUSD":"pound dollar",
                "USDJPY":"yen dollar","XAUUSD":"gold price"}
        url  = "https://newsapi.org/v2/everything"
        resp = requests.get(url, params={
            "q": kw.get(symbol,"forex"),
            "apiKey": config.NEWS_API_KEY,
            "language":"en","sortBy":"publishedAt","pageSize":5
        }, timeout=8)
        data = resp.json()
        if data.get("status") != "ok":
            return {"safe":True,"sentiment":"neutral"}
        headlines   = [a["title"].lower() for a in data.get("articles",[])]
        high_impact = any(w in h for h in headlines
                         for w in ["rate decision","nfp","cpi","fomc","crisis"])
        bull = sum(w in h for h in headlines
                   for w in ["rise","rally","surge","gain","bullish"])
        bear = sum(w in h for h in headlines
                   for w in ["fall","drop","plunge","decline","bearish"])
        sentiment = "bullish" if bull > bear+1 else \
                    "bearish" if bear > bull+1 else "neutral"
        return {"safe": not high_impact, "sentiment": sentiment,
                "high_impact": high_impact}
    except:
        return {"safe":True,"sentiment":"neutral","high_impact":False}


async def generate_signal(symbol: str):
    """Full signal pipeline"""
    mode     = get_bot_mode()
    sessions = get_sessions()
    min_conf = (config.MIN_CONFIDENCE_PASSIVE
                if mode == "passive"
                else config.MIN_CONFIDENCE_ACTIVE)

    # ── Block if pair already has active trade ──
    if symbol in active_trades:
        return None

    # ── Gold Asian session check ──
    if symbol == "XAUUSD" and sessions == ["asian"]:
        df_check = get_candles(symbol, "1h", 20)
        if df_check is not None:
            df_check = add_indicators(df_check)
            df_check.dropna(inplace=True)
            if len(df_check) > 1:
                atr_now = float(df_check["ATR"].iloc[-1])
                atr_avg = float(df_check["ATR"].mean())
                if atr_now < atr_avg * config.GOLD_ASIAN_MIN_ATR:
                    logger.info("XAUUSD: Low Asian volatility — skipping")
                    return None

    # ── Fetch entry candles ──
    df = get_candles(symbol, "5min", 150)
    if df is None or len(df) < 50: return None

    # ── Add features ──
    df = add_indicators(df)
    df = add_smc(df)
    df.dropna(inplace=True)
    if len(df) < 10: return None

    last = df.iloc[-1]
    atr  = float(last.get("ATR", 0.001))

    # ── ADX filter ──
    adx = float(last.get("ADX", 0))
    if adx < 18: return None

    # ── News filter ──
    news = get_news_sentiment(symbol)
    if not news["safe"] and mode != "passive":
        return None

    # ── HTF bias ──
    bias = get_htf_bias(symbol)

    # ── AI prediction ──
    ai_dir, ai_conf = get_ai_prediction(symbol, df)

    # ── Score signal ──
    bull = 0
    bear = 0
    if bias == "bullish":   bull += 3
    elif bias == "bearish": bear += 3
    rsi  = float(last.get("RSI", 50))
    macd_h = float(last.get("MACD_Hist", 0))
    if rsi < 40:    bull += 1
    elif rsi > 60:  bear += 1
    if macd_h > 0:  bull += 1
    elif macd_h < 0: bear += 1
    if float(last.get("Bullish_BOS",0)) > 0: bull += 2
    if float(last.get("Bearish_BOS",0)) > 0: bear += 2
    if float(last.get("Bullish_FVG",0)) > 0: bull += 1
    if float(last.get("Bearish_FVG",0)) > 0: bear += 1
    if float(last.get("Liq_Grab_Low",0))  > 0: bull += 2
    if float(last.get("Liq_Grab_High",0)) > 0: bear         return True, "volatility_spike"
    return False, ""


def get_news_sentiment(symbol: str) -> dict:
    try:
        kw   = {"EURUSD":"euro dollar","GBPUSD":"pound dollar",
                "USDJPY":"yen dollar","XAUUSD":"gold price"}
        url  = "https://newsapi.org/v2/everything"
        resp = requests.get(url, params={
            "q": kw.get(symbol,"forex"),
            "apiKey": config.NEWS_API_KEY,
            "language":"en","sortBy":"publishedAt","pageSize":5
        }, timeout=8)
        data = resp.json()
        if data.get("status") != "ok":
            return {"safe":True,"sentiment":"neutral"}
        headlines   = [a["title"].lower() for a in data.get("articles",[])]
        high_impact = any(w in h for h in headlines
                         for w in ["rate decision","nfp","cpi","fomc","crisis"])
        bull = sum(w in h for h in headlines
                   for w in ["rise","rally","surge","gain","bullish"])
        bear = sum(w in h for h in headlines
                   for w in ["fall","drop","plunge","decline","bearish"])
        sentiment = "bullish" if bull > bear+1 else \
                    "bearish" if bear > bull+1 else "neutral"
        return {"safe": not high_impact, "sentiment": sentiment,
                "high_impact": high_impact}
    except:
        return {"safe":True,"sentiment":"neutral","high_impact":False}


async def generate_signal(symbol: str):
    """Full signal pipeline"""
    mode     = get_bot_mode()
    sessions = get_sessions()
    min_conf = (config.MIN_CONFIDENCE_PASSIVE
                if mode == "passive"
                else config.MIN_CONFIDENCE_ACTIVE)

    # ── Block if pair already has active trade ──
    if symbol in active_trades:
        return None

    # ── Gold Asian session check ──
    if symbol == "XAUUSD" and sessions == ["asian"]:
        df_check = get_candles(symbol, "1h", 20)
        if df_check is not None:
            df_check = add_indicators(df_check)
            df_check.dropna(inplace=True)
            if len(df_check) > 1:
                atr_now = float(df_check["ATR"].iloc[-1])
                atr_avg = float(df_check["ATR"].mean())
                if atr_now < atr_avg * config.GOLD_ASIAN_MIN_ATR:
                    logger.info("XAUUSD: Low Asian volatility — skipping")
                    return None

    # ── Fetch entry candles ──
    df = get_candles(symbol, "5min", 150)
    if df is None or len(df) < 50: return None

    # ── Add features ──
    df = add_indicators(df)
    df = add_smc(df)
    df.dropna(inplace=True)
    if len(df) < 10: return None

    last = df.iloc[-1]
    atr  = float(last.get("ATR", 0.001))

    # ── ADX filter ──
    adx = float(last.get("ADX", 0))
    if adx < 18: return None

    # ── News filter ──
    news = get_news_sentiment(symbol)
    if not news["safe"] and mode != "passive":
        return None

    # ── HTF bias ──
    bias = get_htf_bias(symbol)

    # ── AI prediction ──
    ai_dir, ai_conf = get_ai_prediction(symbol, df)

    # ── Score signal ──
    bull = 0
    bear = 0
    if bias == "bullish":   bull += 3
    elif bias == "bearish": bear += 3
    rsi  = float(last.get("RSI", 50))
    macd_h = float(last.get("MACD_Hist", 0))
    if rsi < 40:    bull += 1
    elif rsi > 60:  bear += 1
    if macd_h > 0:  bull += 1
    elif macd_h < 0: bear += 1
    if float(last.get("Bullish_BOS",0)) > 0: bull += 2
    if float(last.get("Bearish_BOS",0)) > 0: bear += 2
    if float(last.get("Bullish_FVG",0)) > 0: bull += 1
    if float(last.get("Bearish_FVG",0)) > 0: bear += 1
    if float(last.get("Liq_Grab_Low",0))  > 0: bull += 2
    if float(last.get("Liq_Grab_High",0)) > 0: bear += 2
    if float(last.get("Bullish_OB",0)) > 0: bull += 1
    if float(last.get("Bearish_OB",0)) > 0: bear += 1
    if news["sentiment"] == "bullish": bull += 1
    elif news["sentiment"] == "bearish": bear += 1

    direction = None
    if bull > bear + 1:   direction = "buy"
    elif bear > bull + 1: direction = "sell"
    if direction is None: return None

    # ── AI confirmation ──
    ai_boost = 0
    if ai_dir == direction and ai_conf >= config.AI_CONFIDENCE_THRESH:
        ai_boost = 15
    elif ai_dir and ai_dir != direction:
        return None   # AI disagrees — skip

    # ── Calculate confidence ──
    base_score = min((max(bull,bear) / (bull+bear+1e-10)) * 85, 85)
    confidence = min(base_score + ai_boost, 100)

    if confidence < min_conf: return None

    # ── Fast execution ──
    fast_mode, fast_reason = check_fast_execution(df, atr)

    # ── Candlestick confirmation ──
    pattern = detect_pattern(df)
    bullish_p = ["hammer_bullish","bullish_engulfing",
                 "pin_bar_bullish","marubozu_bullish"]
    bearish_p = ["shooting_star_bearish","bearish_engulfing",
                 "pin_bar_bearish","marubozu_bearish"]

    needs_conf = confidence < 85 and not fast_mode
    if needs_conf:
        if direction == "buy"  and pattern not in bullish_p: return None
        if direction == "sell" and pattern not in bearish_p: return None

    # ── Get entry price ──
    entry = float(last["Close"])

    # ── Calculate SL/TP ──
    risk = calculate_risk(symbol, direction, entry, atr)

    # ── Build reason ──
    reasons = []
    reasons.append(f"HTF: {bias.upper()}")
    if adx >= config.ADX_THRESHOLD:
        reasons.append(f"ADX: {adx:.0f}✅")
    if float(last.get("Bullish_BOS",0)) > 0 or float(last.get("Bearish_BOS",0)) > 0:
        reasons.append("BOS confirmed")
    if float(last.get("FVG_Present",0)) > 0:
        reasons.append("FVG present")
    if float(last.get("Liq_Grab_Low",0)) > 0 or float(last.get("Liq_Grab_High",0)) > 0:
        reasons.append("Liq grab detected")
    if pattern != "none":
        reasons.append(pattern.replace("_"," ").title())
    if fast_mode:
        reasons.append(f"⚡ Fast exec ({fast_reason})")
    if ai_dir:
        reasons.append(f"🤖 AI: {ai_conf*100:.0f}%")

    sess_str = "+".join(s.title() for s in sessions) if sessions else "None"

    return {
        "symbol":     symbol,
        "direction":  direction,
        "entry":      entry,
        "sl":         risk["sl"],
        "tp1":        risk["tp1"],
        "tp2":        risk["tp2"],
        "tp3":        risk["tp3"],
        "sl_pips":    risk["sl_pips"],
        "lot_size":   risk["lot_size"],
        "risk_usd":   risk["risk_usd"],
        "confidence": confidence,
        "pattern":    pattern,
        "bias":       bias,
        "fast_mode":  fast_mode,
        "session":    sess_str,
        "reason":     " | ".join(reasons),
        "tp1_hit":    False,
        "tp2_hit":    False,
        "tp3_hit":    False,
        "time":       time.time(),
    }


def format_signal(sig: dict) -> str:
    d    = "🟢 BUY  📈" if sig["direction"] == "buy" else "🔴 SELL 📉"
    fast = "⚡ FAST EXECUTION\n" if sig["fast_mode"] else ""
    bar  = "█" * int(sig["confidence"]/10) + "░" * (10-int(sig["confidence"]/10))
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🚀 NEW SIGNAL — {sig['symbol']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{fast}"
        f"{d}\n"
        f"🎯 SCALPER  |  M5  |  📡 {sig['session']}\n\n"
        f"🎯 Entry:  {sig['entry']}\n"
        f"🛑 SL:     {sig['sl']} ({sig['sl_pips']} pips)\n"
        f"✅ TP1 (1:{config.TP1_RR}): {sig['tp1']}\n"
        f"✅ TP2 (1:{config.TP2_RR}): {sig['tp2']}\n"
        f"✅ TP3 (1:{config.TP3_RR}): {sig['tp3']}\n\n"
        f"📊 Lot: {sig['lot_size']}  |  Risk: ${sig['risk_usd']}\n"
        f"📈 Bias: {sig['bias'].upper()}\n"
        f"🕯 Pattern: {sig['pattern'].replace('_',' ').title()}\n\n"
        f"🔥 Confidence: {sig['confidence']:.0f}%\n"
        f"[{bar}]\n\n"
        f"📝 {sig['reason']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ Always manage your risk!"
    )


# ============================================
# TRADE MONITOR — TP/SL HIT DETECTION
# ============================================

async def monitor_trades(bot: Bot):
    """Check if any active trades hit TP or SL"""
    for symbol, sig in list(active_trades.items()):
        try:
            df = get_candles(symbol, "1min", 5)
            if df is None: continue
            price = float(df["Close"].iloc[-1])
            d     = sig["direction"]

            # ── TP1 ──
            if not sig["tp1_hit"]:
                if (d == "buy"  and price >= sig["tp1"]) or \
                   (d == "sell" and price <= sig["tp1"]):
                    sig["tp1_hit"] = True
                    await bot.send_message(
                        chat_id=config.TELEGRAM_CHAT_ID,
                        text=(
                            f"🎯 TP1 HIT! — {symbol}\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"✅ Take Profit 1 Reached!\n\n"
                            f"Direction: {'🟢 BUY' if d=='buy' else '🔴 SELL'}\n"
                            f"Entry:  {sig['entry']}\n"
                            f"TP1:    {sig['tp1']} ✅\n"
                            f"TP2:    {sig['tp2']} ⏳\n"
                            f"TP3:    {sig['tp3']} ⏳\n\n"
                            f"💡 Move SL to breakeven now!"
                        )
                    )

            # ── TP2 ──
            if sig["tp1_hit"] and not sig["tp2_hit"]:
                if (d == "buy"  and price >= sig["tp2"]) or \
                   (d == "sell" and price <= sig["tp2"]):
                    sig["tp2_hit"] = True
                    await bot.send_message(
                        chat_id=config.TELEGRAM_CHAT_ID,
                        text=(
                            f"🎯 TP2 HIT! — {symbol}\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"✅ Take Profit 2 Reached!\n\n"
                            f"TP1: {sig['tp1']} ✅\n"
                            f"TP2: {sig['tp2']} ✅\n"
                            f"TP3: {sig['tp3']} ⏳\n\n"
                            f"💡 Trail your SL toward TP3!"
                        )
                    )

            # ── TP3 ──
            if sig["tp2_hit"] and not sig["tp3_hit"]:
                if (d == "buy"  and price >= sig["tp3"]) or \
                   (d == "sell" and price <= sig["tp3"]):
                    sig["tp3_hit"] = True
                    signal_stats[symbol]["wins"] += 1
                    del active_trades[symbol]
                    await bot.send_message(
                        chat_id=config.TELEGRAM_CHAT_ID,
                        text=(
                            f"🏆 FULL TARGET HIT! — {symbol}\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"🎉 ALL 3 TPs REACHED!\n\n"
                            f"TP1: {sig['tp1']} ✅\n"
                            f"TP2: {sig['tp2']} ✅\n"
                            f"TP3: {sig['tp3']} ✅\n\n"
                            f"🔥 Perfect trade!\n"
                            f"🔓 {symbol} ready for next signal!"
                        )
                    )

            # ── SL ──
            if (d == "buy"  and price <= sig["sl"]) or \
               (d == "sell" and price >= sig["sl"]):
                signal_stats[symbol]["losses"] += 1
                del active_trades[symbol]
                await bot.send_message(
                    chat_id=config.TELEGRAM_CHAT_ID,
                    text=(
                        f"🛑 STOP LOSS HIT — {symbol}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"❌ Trade Closed at SL\n\n"
                        f"Direction: {'🟢 BUY' if d=='buy' else '🔴 SELL'}\n"
                        f"Entry: {sig['entry']}\n"
                        f"SL:    {sig['sl']} ❌\n\n"
                        f"💸 Loss: -${sig['risk_usd']}\n"
                        f"✅ Account protected!\n"
                        f"🔓 {symbol} ready for next signal!"
                    )
                )

        except Exception as e:
            logger.error(f"Monitor error {symbol}: {e}")


# ============================================
# TELEGRAM COMMANDS
# ============================================

def auth(func):
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id not in config.AUTHORIZED_USERS:
            await update.message.reply_text("⛔ Unauthorized.")
            return
        return await func(update, ctx)
    return wrapper


@auth
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *OLASMOS FX BOT* — Online!\n\n"
        "SMC + AI Powered Scalper Bot\n\n"
        "Commands:\n"
        "/status — Bot status\n"
        "/pairs — Monitored pairs\n"
        "/scan — Force scan now\n"
        "/stats — Win/Loss stats\n"
        "/risk — Risk settings\n"
        "/session — Session info\n"
        "/pause — Pause signals\n"
        "/resume — Resume signals\n"
        "/help — Show commands",
        parse_mode="Markdown"
    )


@auth
async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    mode     = get_bot_mode()
    sessions = get_sessions()
    paused   = "⏸ PAUSED" if bot_paused else "▶️ RUNNING"
    models_ok = f"✅ {len(models)}/4 loaded" if models else "⚠️ Loading..."
    sess_str  = ", ".join(s.title() for s in sessions) if sessions else "None"
    mode_e    = {"sleep":"💤","passive":"😴","active":"✅"}.get(mode,"❓")
    wknd      = " 🔴 WEEKEND" if is_weekend() else ""
    await update.message.reply_text(
        f"📊 *BOT STATUS*\n"
        f"━━━━━━━━━━━━━━\n"
        f"{mode_e} Mode: {mode.upper()}{wknd}\n"
        f"🕐 WAT: {wat_now().strftime('%H:%M %d/%m/%Y')}\n"
        f"📡 Sessions: {sess_str}\n\n"
        f"{paused}\n"
        f"🤖 AI Models: {models_ok}\n"
        f"📈 Active trades: {len(active_trades)}",
        parse_mode="Markdown"
    )


@auth
async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lines = ["📈 *Signal Statistics*\n━━━━━━━━━━━━━━"]
    total_w, total_l = 0, 0
    for p, s in signal_stats.items():
        t  = s["wins"] + s["losses"]
        wr = (s["wins"]/t*100) if t > 0 else 0
        lines.append(f"{p}: {s['wins']}W/{s['losses']}L ({wr:.0f}% WR)")
        total_w += s["wins"]
        total_l += s["losses"]
    tot = total_w + total_l
    owr = (total_w/tot*100) if tot > 0 else 0
    lines.append(f"━━━━━━━━━━━━━━\nTotal: {total_w}W/{total_l}L ({owr:.0f}% WR)")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


@auth
async def cmd_risk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lines = ["💰 *Risk Settings*\n━━━━━━━━━━━━━━"]
    for p in config.PAIRS:
        r = config.RISK_PERCENT[p]
        lines.append(f"{p}: {r}% = ${10*r/100:.2f}")
    lines.append(f"━━━━━━━━━━━━━━\n"
                 f"TP1: 1:{config.TP1_RR}  "
                 f"TP2: 1:{config.TP2_RR}  "
                 f"TP3: 1:{config.TP3_RR}\n"
                 f"🛡️ Max loss/trade: $0.20")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


@auth
async def cmd_session(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sessions = get_sessions()
    mode     = get_bot_mode()
    sess_str = ", ".join(s.title() for s in sessions) if sessions else "None"
    mode_e   = {"sleep":"💤","passive":"😴","active":"✅"}.get(mode,"❓")
    await update.message.reply_text(
        f"🕐 *Session Info*\n"
        f"━━━━━━━━━━━━━━\n"
        f"WAT: {wat_now().strftime('%H:%M')}\n"
        f"UTC: {utc_now().strftime('%H:%M')}\n"
        f"Active: {sess_str}\n"
        f"{mode_e} Bot: {mode.upper()}\n"
        f"Weekend: {'Yes 🔴' if is_weekend() else 'No ✅'}",
        parse_mode="Markdown"
    )


@auth
async def cmd_pairs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pairs = "\n".join([f"• {p}" for p in config.PAIRS])
    await update.message.reply_text(
        f"📋 *Monitored Pairs*\n{pairs}\n\n"
        f"Entry: M1, M5, M15\n"
        f"HTF Bias: M30, H1, H4",
        parse_mode="Markdown"
    )


@auth
async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global bot_paused
    bot_paused = True
    await update.message.reply_text("⏸ Bot paused. /resume to restart.")


@auth
async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global bot_paused
    bot_paused = False
    await update.message.reply_text("▶️ Bot resumed!")


@auth
async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Scanning all pairs...")
    found = 0
    for symbol in config.PAIRS:
        sig = await generate_signal(symbol)
        if sig:
            active_trades[symbol] = sig
            signal_stats[symbol]["total"] += 1
            await update.message.reply_text(format_signal(sig))
            found += 1
    if found == 0:
        await update.message.reply_text("✅ Scan done — no signals right now.")


@auth
async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, ctx)


# ============================================
# MAIN SCANNING LOOP
# ============================================

async def scan_loop(bot: Bot):
    global _last_weekend

    # Load AI models
    load_models()

    await bot.send_message(
        chat_id=config.TELEGRAM_CHAT_ID,
        text=(
            "🤖 *OLASMOS FX BOT STARTED!*\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Monitoring: {', '.join(config.PAIRS)}\n"
            f"AI Models: {'✅ Loaded' if models else '⚠️ Loading...'}\n"
            "Scalper mode active 🎯\n\n"
            "Schedule (Nigerian Time):\n"
            "✅ Active:  6:30AM – 7:00PM\n"
            "😴 Passive: 3:00AM – 6:30AM\n"
            "💤 Sleep:   7:00PM – 3:00AM"
        ),
        parse_mode="Markdown"
    )

    while True:
        try:
            # ── Weekend notification ──
            wknd = is_weekend()
            if wknd != _last_weekend:
                if wknd:
                    await bot.send_message(
                        chat_id=config.TELEGRAM_CHAT_ID,
                        text=(
                            "🔴 *FOREX MARKET CLOSED*\n"
                            "Weekend closure detected!\n"
                            "Bot sleeping until Sunday 11PM WAT 🌙"
                        ),
                        parse_mode="Markdown"
                    )
                else:
                    await bot.send_message(
                        chat_id=config.TELEGRAM_CHAT_ID,
                        text=(
                            "🌅 *FOREX MARKET OPEN!*\n"
                            "Markets reopened! Bot resuming... 🚀"
                        ),
                        parse_mode="Markdown"
                    )
                _last_weekend = wknd

            mode = get_bot_mode()

            # ── Monitor active trades (even during sleep) ──
            if active_trades:
                await monitor_trades(bot)

            if mode == "sleep" or bot_paused:
                await asyncio.sleep(60)
                continue

            interval = (config.SCAN_INTERVAL_PASSIVE
                        if mode == "passive"
                        else config.SCAN_INTERVAL_ACTIVE)

            # ── Scan each pair ──
            for symbol in config.PAIRS:
                if symbol in active_trades:
                    continue
                try:
                    sig = await generate_signal(symbol)
                    if sig:
                        active_trades[symbol] = sig
                        signal_stats[symbol]["total"] += 1
                        await bot.send_message(
                            chat_id=config.TELEGRAM_CHAT_ID,
                            text=format_signal(sig)
                        )
                        logger.info(f"Signal: {symbol} {sig['direction']} "
                                    f"{sig['confidence']:.0f}%")
                        await asyncio.sleep(3)
                except Exception as e:
                    logger.error(f"Scan error {symbol}: {e}")

            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(30)


# ============================================
# MAIN — Start bot with webhook
# ============================================

async def main():
    keep_alive()

    app = Application.builder().token(config.TELEGRAM_TOKEN).build()

    # Register commands
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("pairs",   cmd_pairs))
    app.add_handler(CommandHandler("stats",   cmd_stats))
    app.add_handler(CommandHandler("risk",    cmd_risk))
    app.add_handler(CommandHandler("session", cmd_session))
    app.add_handler(CommandHandler("pause",   cmd_pause))
    app.add_handler(CommandHandler("resume",  cmd_resume))
    app.add_handler(CommandHandler("scan",    cmd_scan))
    app.add_handler(CommandHandler("help",    cmd_help))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    bot       = app.bot
    scan_task = asyncio.create_task(scan_loop(bot))

    logger.info("✅ Olasmos FX Bot is LIVE!")

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        scan_task.cancel()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
