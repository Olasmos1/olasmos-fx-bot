# ============================================
# scripts/retrain.py
# Weekly retraining script
# Runs automatically via GitHub Actions
# ============================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import ta, joblib, os, base64, requests
import warnings
warnings.filterwarnings('ignore')

print("🤖 Starting weekly retraining...")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ── Config ──
GITHUB_TOKEN    = os.environ.get('GITHUB_TOKEN', '')
GITHUB_USERNAME = "Olasmos1"
GITHUB_REPO     = "olasmos-fx-bot"

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F"
}

EXCLUDE_COLS = ['Label','Open','High','Low','Close','Volume']

# ── Download data ──
def download_data(pairs):
    data = {}
    for name, ticker in pairs.items():
        print(f"📥 Downloading {name}...")
        df = yf.download(ticker, period="2y", interval="1h", progress=False)
        if not df.empty:
            df.columns = ['Open','High','Low','Close','Volume'] if len(df.columns)==5 else df.columns
            data[name] = df
            print(f"✅ {name}: {len(df)} candles")
    return data

# ── Add indicators ──
def add_indicators(df):
    df = df.copy()
    close, high, low = df['Close'], df['High'], df['Low']
    df['RSI']         = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd              = ta.trend.MACD(close)
    df['MACD']        = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist']   = macd.macd_diff()
    bb                = ta.volatility.BollingerBands(close)
    df['BB_Upper']    = bb.bollinger_hband()
    df['BB_Lower']    = bb.bollinger_lband()
    df['BB_Mid']      = bb.bollinger_mavg()
    df['BB_Width']    = df['BB_Upper'] - df['BB_Lower']
    df['EMA_21']      = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['EMA_50']      = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df['EMA_200']     = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    adx               = ta.trend.ADXIndicator(high, low, close)
    df['ADX']         = adx.adx()
    df['ADX_pos']     = adx.adx_pos()
    df['ADX_neg']     = adx.adx_neg()
    df['ATR']         = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df['Candle_Body'] = abs(close - df['Open'])
    df['Upper_Wick']  = high - df[['Open','Close']].max(axis=1)
    df['Lower_Wick']  = df[['Open','Close']].min(axis=1) - low
    df['Is_Bullish']  = (close > df['Open']).astype(int)
    df['Return_1']    = close.pct_change(1)
    df['Return_3']    = close.pct_change(3)
    df['Return_5']    = close.pct_change(5)
    return df

# ── Add SMC features ──
def add_smc(df):
    df = df.copy()
    bull_fvg, bear_fvg = [], []
    bull_ob,  bear_ob  = [0,0], [0,0]

    for i in range(2, len(df)):
        c1l = df['Low'].iloc[i-2]
        c1h = df['High'].iloc[i-2]
        c3h = df['High'].iloc[i]
        c3l = df['Low'].iloc[i]
        bull_fvg.append(c1l - c3h if c1l > c3h else 0)
        bear_fvg.append(c3l - c1h if c1h < c3l else 0)
        pb = abs(df['Close'].iloc[i-1] - df['Open'].iloc[i-1])
        cb = abs(df['Close'].iloc[i]   - df['Open'].iloc[i])
        bull_ob.append(1 if df['Close'].iloc[i-1] < df['Open'].iloc[i-1]
                          and df['Close'].iloc[i] > df['Open'].iloc[i]
                          and cb > pb * 1.5 else 0)
        bear_ob.append(1 if df['Close'].iloc[i-1] > df['Open'].iloc[i-1]
                          and df['Close'].iloc[i] < df['Open'].iloc[i]
                          and cb > pb * 1.5 else 0)

    df['Bullish_FVG']     = [0,0] + bull_fvg
    df['Bearish_FVG']     = [0,0] + bear_fvg
    df['FVG_Present']     = ((df['Bullish_FVG']>0)|(df['Bearish_FVG']>0)).astype(int)
    df['Bullish_OB']      = bull_ob
    df['Bearish_OB']      = bear_ob
    df['Prev_High_20']    = df['High'].rolling(20).max().shift(1)
    df['Prev_Low_20']     = df['Low'].rolling(20).min().shift(1)
    df['Bullish_BOS']     = (df['Close'] > df['Prev_High_20']).astype(int)
    df['Bearish_BOS']     = (df['Close'] < df['Prev_Low_20']).astype(int)
    df['Swing_High_10']   = df['High'].rolling(10).max().shift(1)
    df['Swing_Low_10']    = df['Low'].rolling(10).min().shift(1)
    df['Liq_Grab_High']   = ((df['High']>df['Swing_High_10'])&(df['Close']<df['Swing_High_10'])).astype(int)
    df['Liq_Grab_Low']    = ((df['Low']<df['Swing_Low_10'])&(df['Close']>df['Swing_Low_10'])).astype(int)
    df['Dist_To_High_50'] = df['High'].rolling(50).max() - df['Close']
    df['Dist_To_Low_50']  = df['Close'] - df['Low'].rolling(50).min()
    df['HH'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['LL'] = (df['Low']  < df['Low'].shift(1)).astype(int)
    df['HL'] = (df['Low']  > df['Low'].shift(2)).astype(int)
    df['LH'] = (df['High'] < df['High'].shift(2)).astype(int)
    df.drop(columns=['Prev_High_20','Prev_Low_20',
                     'Swing_High_10','Swing_Low_10'], inplace=True)
    return df

# ── Create labels ──
def create_labels(df, symbol):
    df    = df.copy()
    pip   = {"EURUSD":0.0001,"GBPUSD":0.0001,"USDJPY":0.01,"XAUUSD":0.10}.get(symbol,0.0001)
    min_m = {"EURUSD":8,"GBPUSD":8,"USDJPY":8,"XAUUSD":15}.get(symbol,8) * pip
    fh    = df['High'].rolling(5).max().shift(-5)
    fl    = df['Low'].rolling(5).min().shift(-5)
    up    = fh - df['Close']
    down  = df['Close'] - fl
    labels = []
    for i in range(len(df)):
        u = up.iloc[i] if i < len(up) else 0
        d = down.iloc[i] if i < len(down) else 0
        if pd.isna(u) or pd.isna(d): labels.append(0)
        elif u > min_m and u > d*1.2: labels.append(1)
        elif d > min_m and d > u*1.2: labels.append(2)
        else: labels.append(0)
    df['Label'] = labels
    return df

# ── Upload to GitHub ──
def upload_to_github(local_path, github_path, token):
    with open(local_path, 'rb') as f:
        content = base64.b64encode(f.read()).decode('utf-8')
    url     = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{github_path}"
    headers = {"Authorization": f"token {token}",
               "Accept": "application/vnd.github.v3+json"}
    resp    = requests.get(url, headers=headers)
    sha     = resp.json().get('sha') if resp.status_code == 200 else None
    data    = {"message": f"🤖 Retrained: {github_path}",
               "content": content, "branch": "main"}
    if sha: data["sha"] = sha
    r = requests.put(url, json=data, headers=headers)
    return r.status_code in [200, 201]

# ── MAIN ──
os.makedirs("models", exist_ok=True)

all_data = download_data(PAIRS)
scores   = {}

for name, df in all_data.items():
    print(f"\n🔧 Retraining {name}...")
    try:
        df = add_indicators(df)
        df = add_smc(df)
        df = create_labels(df, name)
        df.dropna(inplace=True)

        df_clean         = df[df['Label'] != 0].copy()
        df_clean['Label_Binary'] = (df_clean['Label'] == 1).astype(int)
        feature_cols     = [c for c in df_clean.columns
                            if c not in EXCLUDE_COLS+['Label','Label_Binary']]
        X = df_clean[feature_cols].values
        y = df_clean['Label_Binary'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False)

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        n_trees = 300 if name == "XAUUSD" else 200
        model   = RandomForestClassifier(
            n_estimators=n_trees, max_depth=10,
            min_samples_leaf=20, class_weight='balanced',
            random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        scores[name] = accuracy
        print(f"✅ {name}: {accuracy*100:.1f}% accuracy")

        # Save locally
        joblib.dump(model,        f"models/{name}_model.pkl")
        joblib.dump(scaler,       f"models/{name}_scaler.pkl")
        joblib.dump(feature_cols, f"models/{name}_features.pkl")

        # Upload to GitHub
        for ftype in ['model','scaler','features']:
            fname = f"{name}_{ftype}.pkl"
            if upload_to_github(f"models/{fname}", f"models/{fname}", GITHUB_TOKEN):
                print(f"📤 Uploaded {fname}")

    except Exception as e:
        print(f"❌ Error on {name}: {e}")

print("\n" + "="*40)
print("🎉 WEEKLY RETRAINING COMPLETE!")
for name, acc in scores.items():
    print(f"   {name}: {acc*100:.1f}%")
print("="*40)
