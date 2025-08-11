#!/usr/bin/env python3
"""
Telegram bildirimli, saatlik kripto sinyal botu (Render arka plan worker için uygun).
- Sadece sinyal üretir ve Telegram'a gönderir.
- Otomatik alım/satım kısmı COMMENTED OUT (güvenlik için).
"""

import os
import time
import json
import logging
from datetime import datetime, timezone
import requests
import pandas as pd
from binance.client import Client
from apscheduler.schedulers.blocking import BlockingScheduler
from ta.trend import MACD
from ta.momentum import RSIIndicator

# ---------- CONFIG (env vars) ----------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")  # must be string
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
SYMBOLS = os.environ.get("COIN_LIST", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")  # comma separated
INTERVAL = os.environ.get("INTERVAL", "1h")  # binance interval (1h)
LIMIT = int(os.environ.get("LIMIT", "500"))
AGREE_THRESHOLD = float(os.environ.get("AGREE_THRESHOLD", "0.6"))  # yüzde 0-1
STATE_FILE = "state.json"

# Weights for indicators
WEIGHTS = {
    "ema": float(os.environ.get("W_EMA", "1.0")),
    "rsi": float(os.environ.get("W_RSI", "0.8")),
    "macd": float(os.environ.get("W_MACD", "0.9")),
    "bb": float(os.environ.get("W_BB", "0.6")),
    "volume": float(os.environ.get("W_VOL", "0.5")),
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kripto-bot")

# Binance client (public if keys empty)
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)


# ---------- Helper: state persistence ----------
def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("State load error: %s", e)
        return {}

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.warning("State save error: %s", e)


# ---------- Telegram ----------
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram bilgileri ayarlı değil. Mesaj atılmadı.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram mesajı gönderildi.")
    except Exception as e:
        logger.error("Telegram gönderim hatası: %s", e)


# ---------- Data fetch ----------
def fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=500):
    """
    Binance kline -> pandas.DataFrame: index = open_time, columns: open, high, low, close, volume
    """
    try:
        bars = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        logger.error("Binance veri çekme hatası: %s", e)
        raise
    df = pd.DataFrame(bars, columns=[
        'open_time','open','high','low','close','volume','close_time',
        'qav','num_trades','taker_base_vol','taker_quote_vol','ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open','high','low','close','volume']].astype(float)
    return df


# ---------- Indicators & signals ----------
def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    # EMA
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()

    # RSI
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

    # MACD (ta library)
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # Bollinger Bands (20,2)
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    # Volume avg
    df['vol_avg20'] = df['volume'].rolling(20).mean()

    return df


def compute_signals(df: pd.DataFrame):
    """
    Returns dict of indicator signals: 1 buy, -1 sell, 0 hold
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signals = {}
    # EMA crossover
    if latest['ema12'] > latest['ema26'] and prev['ema12'] <= prev['ema26']:
        signals['ema'] = 1
    elif latest['ema12'] < latest['ema26'] and prev['ema12'] >= prev['ema26']:
        signals['ema'] = -1
    else:
        signals['ema'] = 0

    # RSI
    if latest['rsi'] < 30:
        signals['rsi'] = 1
    elif latest['rsi'] > 70:
        signals['rsi'] = -1
    else:
        signals['rsi'] = 0

    # MACD histogram
    if latest['macd_hist'] > 0 and prev['macd_hist'] <= 0:
        signals['macd'] = 1
    elif latest['macd_hist'] < 0 and prev['macd_hist'] >= 0:
        signals['macd'] = -1
    else:
        signals['macd'] = 0

    # Bollinger breakout - conservative
    if latest['close'] < latest['bb_lower']:
        signals['bb'] = 1  # oversold bounce possible
    elif latest['close'] > latest['bb_upper']:
        signals['bb'] = -1
    else:
        signals['bb'] = 0

    # Volume spike
    try:
        vol_avg = latest['vol_avg20']
        signals['volume'] = 1 if latest['volume'] > vol_avg * 1.8 else 0
    except Exception:
        signals['volume'] = 0

    return signals


def aggregate(signals: dict, weights: dict = WEIGHTS):
    score = sum(weights.get(k, 0) * signals.get(k, 0) for k in signals.keys())
    if score >= 1.0:
        final = "BUY"
    elif score <= -1.0:
        final = "SELL"
    else:
        final = "HOLD"

    pos = sum(1 for v in signals.values() if v == 1)
    neg = sum(1 for v in signals.values() if v == -1)
    total = len(signals)
    agree_pct = max(pos, neg) / total if total else 0.0
    return final, score, agree_pct


# ---------- Main cycle ----------
def run_cycle(symbol="BTCUSDT"):
    logger.info("Çalışıyor: %s", symbol)
    try:
        df = fetch_ohlcv(symbol=symbol, interval=INTERVAL, limit=LIMIT)
    except Exception:
        logger.exception("Veri çekilemedi, atlandı: %s", symbol)
        return

    df = compute_indicators(df)

    # require non-na last row
    if df.isnull().iloc[-1].any():
        logger.warning("Eksik göstergeler, atlandı: %s", symbol)
        return

    signals = compute_signals(df)
    final, score, agree = aggregate(signals)

    last_price = df['close'].iloc[-1]
    ts_utc = df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc).isoformat()

    message = (
        f"<b>{symbol}</b>\n"
        f"Zaman (UTC): {ts_utc}\n"
        f"Son fiyat: {last_price:.8f}\n"
        f"Karar: <b>{final}</b>\n"
        f"Score: {score:.2f}\n"
        f"Agree: {agree:.0%}\n"
        f"Sinyaller: {signals}\n"
    )

    # load last sent state to avoid duplicate spam
    state = load_state()
    last = state.get(symbol, {})

    # send only if agreement büyükse ve sinyal değişmiş
    send_flag = False
    if agree >= AGREE_THRESHOLD and final in ("BUY", "SELL"):
        # check previous (avoid duplicate notifications for same decision)
        prev_decision = last.get("decision")
        if prev_decision != final:
            send_flag = True
    # additional: always log
    logger.info("Sinyaller %s | final=%s score=%.2f agree=%.2f", signals, final, score, agree)

    if send_flag:
        send_telegram(message)
        state[symbol] = {"decision": final, "timestamp": ts_utc, "score": score, "agree": agree}
        save_state(state)
    else:
        logger.info("Bildirim gönderilmedi (send_flag=False). final=%s agree=%.2f", final, agree)


# Optional: place order (COMMENTED for safety)
"""
def place_order(symbol, side, quantity, order_type='MARKET'):
    # WARNING: enable only AFTER thorough testing and risk rules
    try:
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        logger.info("Order placed: %s", order)
        return order
    except Exception as e:
        logger.error("Order error: %s", e)
        return None
"""

# ---------- Scheduler ----------
def job_all():
    for s in SYMBOLS:
        s = s.strip().upper()
        try:
            run_cycle(s)
        except Exception:
            logger.exception("Job error for %s", s)

def main():
    logger.info("Bot başlatılıyor. İzlenecek semboller: %s", SYMBOLS)
    # Run once at start
    job_all()

    scheduler = BlockingScheduler()
    # Run every hour at minute 0
    scheduler.add_job(job_all, 'cron', minute=0)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler durduruluyor.")
    except Exception:
        logger.exception("Scheduler hatası.")


if __name__ == "__main__":
    main()
    
