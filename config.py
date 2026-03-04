# ============================================
# OLASMOS FX BOT — CONFIGURATION
# All settings in one place
# ============================================

import os

# ─── TELEGRAM ────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ─── API KEYS ────────────────────────────────
TWELVE_DATA_API   = os.environ.get("TWELVE_DATA_API", "")
ALPHA_VANTAGE_API = os.environ.get("ALPHA_VANTAGE_API", "")
NEWS_API_KEY      = os.environ.get("NEWS_API_KEY", "")
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")

# ─── GITHUB MODELS ───────────────────────────
GITHUB_USERNAME   = "Olasmos1"
GITHUB_REPO       = "olasmos-fx-bot"
GITHUB_BRANCH     = "main"

# ─── PAIRS ───────────────────────────────────
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

# yfinance symbols
YFINANCE_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F",
}

# Twelve Data symbols
TWELVE_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "XAUUSD": "XAU/USD",
}

# ─── TIMEFRAMES ──────────────────────────────
ENTRY_TIMEFRAMES = ["1min", "5min", "15min"]
HTF_TIMEFRAMES   = ["30min", "1h", "4h"]

# ─── RISK MANAGEMENT ─────────────────────────
RISK_PERCENT = {
    "XAUUSD": 1.0,
    "EURUSD": 2.0,
    "GBPUSD": 2.0,
    "USDJPY": 2.0,
}

# Take Profit R:R
TP1_RR = 1.5
TP2_RR = 2.5
TP3_RR = 4.0

# Small account protection
MIN_ACCOUNT_BALANCE  = 10.0
MAX_LOSS_PER_TRADE   = 0.20
MIN_LOT              = 0.01

# ─── PIP SIZES ───────────────────────────────
PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "XAUUSD": 0.10,
}

# ─── SESSION TIMES (UTC) ─────────────────────
SESSIONS = {
    "asian":    {"start": 0,  "end": 8},
    "london":   {"start": 7,  "end": 16},
    "new_york": {"start": 12, "end": 21},
    "overlap":  {"start": 12, "end": 16},
}

# ─── BOT SCHEDULE (UTC) ──────────────────────
# Nigerian WAT = UTC+1
# Sleep:   19:00-03:00 WAT = 18:00-02:00 UTC
# Passive: 03:00-06:30 WAT = 02:00-05:30 UTC
# Active:  06:30-19:00 WAT = 05:30-18:00 UTC
SLEEP_START_UTC   = 18
SLEEP_END_UTC     = 2
PASSIVE_END_UTC   = 5
PASSIVE_END_MIN   = 30

# ─── INDICATOR SETTINGS ──────────────────────
RSI_PERIOD        = 14
RSI_OVERBOUGHT    = 70
RSI_OVERSOLD      = 30
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL_SPAN  = 9
BB_PERIOD         = 20
BB_STD            = 2
ADX_PERIOD        = 14
ADX_THRESHOLD     = 25
EMA_FAST          = 21
EMA_SLOW          = 50
EMA_TREND         = 200
ATR_PERIOD        = 14

# ─── SMC SETTINGS ────────────────────────────
FVG_MIN_PIPS = {
    "EURUSD": 5,
    "GBPUSD": 5,
    "USDJPY": 5,
    "XAUUSD": 10,
}
OB_LOOKBACK  = 20
BOS_LOOKBACK = 30
LIQ_LOOKBACK = 40

# ─── SIGNAL SETTINGS ─────────────────────────
MIN_CONFIDENCE_ACTIVE  = 70
MIN_CONFIDENCE_PASSIVE = 88
SIGNAL_COOLDOWN        = 0      # 0 = wait for TP/SL hit
AI_CONFIDENCE_THRESH   = 0.65

# ─── GOLD SETTINGS ───────────────────────────
GOLD_SPREAD_MAX       = 30
GOLD_ASIAN_MIN_ATR    = 1.5
GOLD_ASIAN_ALLOWED    = True    # allowed if high volatility

# ─── FAST EXECUTION ──────────────────────────
FAST_EXEC_RSI         = 80
FAST_EXEC_SPREAD_MULT = 2.0
FAST_EXEC_ATR_MULT    = 2.0

# ─── SCAN INTERVALS ──────────────────────────
SCAN_INTERVAL_ACTIVE  = 60     # seconds
SCAN_INTERVAL_PASSIVE = 300    # seconds

# ─── AUTHORIZED USERS ────────────────────────
AUTHORIZED_USERS = [int(TELEGRAM_CHAT_ID)] if TELEGRAM_CHAT_ID else []

# ─── RENDER WEBHOOK ──────────────────────────
WEBHOOK_URL  = os.environ.get("RENDER_EXTERNAL_URL", "")
PORT         = int(os.environ.get("PORT", 8080))
