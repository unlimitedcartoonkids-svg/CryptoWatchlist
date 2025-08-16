import asyncio
import re
import numpy as np
from datetime import datetime, UTC
from typing import Any, List, Sequence, Tuple, Optional, Dict

from telegram import Update as TGUpdate
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from data_api import get_exchange_info, get_klines, get_column, get_removed_map

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
    return [n for n in range(2, len(highs)) if lows[n] > highs[n-2]]

def last_fvg_status(highs: List[float], lows: List[float], closes: List[float]) -> Optional[Tuple[str, float, float]]:
    idxs = find_bullish_fvg_indices(highs, lows)
    if not idxs:
        return None
    n = idxs[-1]
    gap_top = lows[n]
    gap_bottom = highs[n-2]
    current_price = closes[-1]

    inside_index = None
    for i in range(n+1, len(closes)):
        body_low = min(closes[i], closes[i-1])
        body_high = max(closes[i], closes[i-1])
        if gap_bottom <= body_low and body_high <= gap_top:
            inside_index = i
            break

    if inside_index is not None:
        for j in range(inside_index+1, len(closes)):
            if closes[j] > gap_top and closes[j] > closes[j-1]:
                return "RESPECTED", gap_bottom, gap_top
        if gap_bottom <= current_price <= gap_top:
            if inside_index >= len(closes) - 2:
                return "JUST_ENTERED", gap_bottom, gap_top
            else:
                return "INSIDE", gap_bottom, gap_top
    return None

# --- MSS / BOS DETECTION ---
def bullish_mss_or_bos(highs: List[float], closes: List[float]) -> bool:
    if len(closes) < 10:
        return False
    for i in range(5, len(closes)):
        if closes[i] > max(highs[:i-2]):
            return True
    return False

# --- RSI Divergence ---
def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return []
    deltas = np.diff(values)
    seed = deltas[:period]
    up = float(np.sum(seed[seed >= 0])) / period
    down = float(-np.sum(seed[seed < 0])) / period
    rs = up / down if down != 0 else 0.0
    rsi_values = [100 - (100 / (1 + rs))]

    up_avg, down_avg = up, down
    for delta in deltas[period:]:
        up_val = float(max(delta, 0))
        down_val = float(-min(delta, 0))
        up_avg = (up_avg * (period - 1) + up_val) / period
        down_avg = (down_avg * (period - 1) + down_val) / period
        rs = up_avg / down_avg if down_avg != 0 else 0.0
        rsi_values.append(100 - (100 / (1 + rs)))
    return [np.nan] * period + rsi_values

def bullish_rsi_divergence(closes: List[float], lows: List[float]) -> bool:
    rsi_vals = rsi(closes)
    if len(rsi_vals) < 5:
        return False
    price_low1, price_low2 = lows[-5], lows[-1]
    rsi_low1, rsi_low2 = rsi_vals[-5], rsi_vals[-1]
    return price_low2 < price_low1 and rsi_low2 > rsi_low1

# --- RS (relative strength) ---
def relative_strength(closes: List[float]) -> float:
    if len(closes) >= 20:
        return float(closes[-1]) / float(np.mean(closes[-20:]))
    else:
        return 0.0

# --- CVD Proxy ---
def cvd_proxy(closes: List[float], volumes: List[float]) -> float:
    delta = [float(v) if closes[i] > closes[i-1] else -float(v) for i, v in enumerate(volumes) if i > 0]
    return float(np.sum(delta))

# --- VOLATILITY & VOLUME ---
def volatility_metric(closes: List[float], win: int = 30) -> float:
    arr = np.array(closes, dtype=float)
    if arr.size < win:
        return 0.0
    mu = float(np.mean(arr[-win:]))
    return float(np.std(arr[-win:]) / mu) if mu else 0.0

def mean_volume(volumes: List[float], win: int = 30) -> float:
    arr = np.array(volumes, dtype=float)
    if arr.size < win:
        return float(np.mean(arr)) if arr.size else 0.0
    return float(np.mean(arr[-win:]))

# --- MAIN SCAN ---
async def get_fvg_coins_async(tf: str) -> List[Dict[str, Any]]:
    exinfo = await get_exchange_info()
    symbols = [
        s["symbol"] for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and not any(x in s["symbol"] for x in ["UPUSDT","DOWNUSDT","BULLUSDT","BEARUSDT"])
    ]

    watchlist = set([s.upper() for s in await get_column("watchlist")])
    haram = set([s.upper() for s in await get_column("haram")])
    removed = await get_removed_map()

    sem = asyncio.Semaphore(8)

    async def check_symbol(sym: str):
        async with sem:
            try:
                candles = await get_klines(sym, tf, 150)  # <- 150 candles
                opens, highs, lows, closes, volumes, _ = parse_ohlcv(candles)
                if len(closes) < 35:
                    return None

                fvg_info = last_fvg_status(highs, lows, closes)
                if not fvg_info:
                    return None
                status_fvg, gap_bottom, gap_top = fvg_info
                if not bullish_mss_or_bos(highs, closes):
                    return None
                if closes[-1] < gap_bottom:
                    return None

                vol_metric = volatility_metric(closes)
                avg_vol = mean_volume(volumes)
                status = ("HARAM" if sym in haram else
                          "REMOVED" if sym in removed else
                          "WATCHLIST" if sym in watchlist else
                          "NEW")
                rsi_div = bullish_rsi_divergence(closes, lows)
                rs_val = relative_strength(closes)
                cvd_val = cvd_proxy(closes, volumes)

                return {
                    "symbol": sym,
                    "fvg_status": status_fvg,
                    "volatility": vol_metric,
                    "avg_volume": avg_vol,
                    "status": status,
                    "inside_fvg": gap_bottom <= closes[-1] <= gap_top,
                    "rsi_div": rsi_div,
                    "rs_val": rs_val,
                    "cvd_val": cvd_val
                }
            except Exception:
                return None

    tasks = [check_symbol(s) for s in symbols]
    results = []
    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        results.extend(await asyncio.gather(*tasks[i:i+batch_size]))
        await asyncio.sleep(0.3)

    return [r for r in results if r]

# --- Stylish Telegram Message Formatter ---
def format_stylish_coinlist(coins: List[Dict[str, Any]], tf_label: str) -> str:
    now = datetime.now(UTC)
    dt_str = now.strftime('%Y-%m-%d %H:%M UTC')
    header = (
        f"🚀 *Binance FVG Coin Scanner — {tf_label} Timeframe*\n\n"
        f"🔍 *Scan Status:* Completed\n"
        f"🕒 *Timeframe:* {tf_label}\n"
        f"📅 *Last Updated:* {dt_str}\n"
        f"━━━━━━━━━━━━━━\n"
        f"🪙 *Selected Coins:*\n"
    )
    if not coins:
        return (
            header +
            "\n😔 No coins matched the criteria for this timeframe.\n"
            "🔁 Try again later or change your scan parameters."
        )

    coins.sort(key=lambda x: (x["volatility"], x["avg_volume"]), reverse=True)
    body = ""
    for i, c in enumerate(coins, 1):
        signals = []
        if c["rsi_div"]:
            signals.append("📈 *RSI Divergence*")
        if c["rs_val"] > 1:
            signals.append("💪 *RS↑*")
        if c["cvd_val"] > 0:
            signals.append("📊 *CVD↑*")
        extra_str = " | ".join(signals) if signals else "—"
        body += (
            f"\n{i}️⃣ *{c['symbol']}*\n"
            f" • FVG: _{c['fvg_status']}_\n"
            f" • Volatility: `{c['volatility']:.4f}`\n"
            f" • Avg Volume: `{int(c['avg_volume']):,}`\n"
            f" • Status: _{c['status']}_\n"
            f" • Signals: {extra_str}\n"
        )
    footer = (
        "\n━━━━━━━━━━━━━━\n"
        "✨ *Scan coins for technical signals. Make smart decisions!*"
    )
    return header + body + footer

# --- TELEGRAM COMMAND HANDLER ---
TIMEFRAME_MAP = {"1H": "1h", "4H": "4h", "1D": "1d"}

async def fvg_coinlist_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip().upper()
    m = re.match(r'(1H|4H|1D)\s*FVG\s*COIN\s*LIST', text)
    if not m:
        await update.message.reply_text(
            "⚠️ Invalid command format!\n"
            "📝 Example: `4H FVG Coin list`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    tf_label = m.group(1)
    tf = TIMEFRAME_MAP.get(tf_label)
    if tf is None:
        await update.message.reply_text(
            "❌ Invalid timeframe! Only 1H, 4H, 1D allowed.",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    await update.message.reply_text(
        f"⏳ Scanning Binance ({tf_label}) FVG coins...",
        parse_mode=ParseMode.MARKDOWN
    )

    coins = await get_fvg_coins_async(tf)
    message = format_stylish_coinlist(coins, tf_label)
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)