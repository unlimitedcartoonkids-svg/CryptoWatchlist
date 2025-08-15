import asyncio
import re
import numpy as np
from typing import Any, List, Sequence, Tuple, cast

from telegram import Update as TGUpdate
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

# IMPORT PUBLIC ASYNC BINANCE API FUNCTIONS
from data_api import get_exchange_info, get_klines

# --- OHLCV PARSING ---
def parse_ohlcv(candles: Sequence[Sequence[Any]]) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[int]]:
    opens = [float(c[1]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    volumes = [float(c[5]) for c in candles]
    times = [int(c[0]) for c in candles]
    return opens, highs, lows, closes, volumes, times

# --- FVG DETECTION ---
def find_bullish_fvg_indices(highs: List[float], lows: List[float]) -> List[int]:
    out = []
    for n in range(2, len(highs)):
        if lows[n] > highs[n-2]:
            out.append(n)
    return out

def last_fvg_respected(highs: List[float], lows: List[float], closes: List[float]) -> bool:
    idxs = find_bullish_fvg_indices(highs, lows)
    if not idxs:
        return False
    n = idxs[-1]
    gap_top = lows[n]
    gap_bottom = highs[n-2]
    # Check if after FVG, price had at least one close inside zone, then closed above
    for i in range(n+1, len(closes)):
        body_low = min(closes[i], closes[i-1])
        body_high = max(closes[i], closes[i-1])
        if gap_bottom <= body_low and body_high <= gap_top:
            # Now see if after that price closed above gap_top
            for j in range(i+1, len(closes)):
                if closes[j] > gap_top and closes[j] > closes[j-1]:
                    return True
    return False

# --- VOLATILITY & VOLUME METRIC ---
def volatility_metric(closes: List[float], win=30) -> float:
    arr = np.array(closes, dtype=float)
    if arr.size < win:
        return 0.0
    seg = arr[-win:]
    mu = float(np.mean(seg))
    return float(np.std(seg) / mu) if mu else 0.0

def mean_volume(volumes: List[float], win=30) -> float:
    arr = np.array(volumes, dtype=float)
    if arr.size < win:
        return float(np.mean(arr)) if arr.size else 0.0
    return float(np.mean(arr[-win:]))

# --- MAIN SCAN FUNCTION (ASYNC PUBLIC API) ---
async def get_fvg_respect_coins_async(tf: str, top_n: int = 20) -> List[Tuple[str, float, float]]:
    # Get all USDT spot coins from public API
    exinfo = await get_exchange_info()
    symbols = [
        s["symbol"] for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and not any(x in s["symbol"] for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])
    ]
    out: List[Tuple[str, float, float]] = []

    async def check_symbol(sym: str):
        try:
            candles = await get_klines(sym, tf, 80)
            _, highs, lows, closes, volumes, _ = parse_ohlcv(candles)
            if len(closes) < 35:
                return None
            if last_fvg_respected(highs, lows, closes):
                vol = volatility_metric(closes, win=30)
                mv = mean_volume(volumes, win=30)
                return (sym, vol, mv)
        except Exception:
            return None

    # Async gather with some throttling to avoid rate limit
    tasks = [check_symbol(sym) for sym in symbols]
    results = []
    # To avoid hitting rate limits hard, process in batches
    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_result = await asyncio.gather(*batch)
        results.extend(batch_result)
        await asyncio.sleep(0.2)  # tweak as needed

    # Filter None and sort
    out = [r for r in results if r is not None]
    out.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return out[:top_n]

# --- TELEGRAM COMMAND HANDLER ---
TIMEFRAME_MAP = {
    "1H": "1h",
    "4H": "4h",
    "1D": "1d"
}

async def fvg_coinlist_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip().upper()
    m = re.match(r'(1H|4H|1D)\s*FVG\s*COIN\s*LIST', text)
    if not m:
        await update.message.reply_text("❌ Invalid command format! উদাহরণ: 4H FVG Coin list", parse_mode=ParseMode.MARKDOWN)
        return
    tf_label = m.group(1)
    tf = TIMEFRAME_MAP.get(tf_label, None)
    if not tf:
        await update.message.reply_text("❌ Supported timeframes: 1H, 4H, 1D", parse_mode=ParseMode.MARKDOWN)
        return

    await update.message.reply_text(f"⏳ Scanning Binance ({tf_label}) FVG respected coins...", parse_mode=ParseMode.MARKDOWN)
    try:
        coins = await get_fvg_respect_coins_async(tf, 20)
        if not coins:
            await update.message.reply_text("😔 No FVG respected coins found.", parse_mode=ParseMode.MARKDOWN)
            return
        lines = [f"🎯 Top FVG Respected Coins ({tf_label}) [Sorted: Volatility+Volume]:", ""]
        for i, (sym, vol, mv) in enumerate(coins, 1):
            lines.append(f"{i:2d}. `{sym}` | Volatility: {vol:.4f} | AvgVol: {mv:,.0f}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

# Example integration to main.py:
# from fvg_coinlist import fvg_coinlist_handler
# app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"(?i)^(1H|4H|1D)\s*FVG\s*COIN\s*LIST\s*$"), fvg_coinlist_handler))