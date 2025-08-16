import asyncio
import re
import numpy as np
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
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi_values = [100 - (100 / (1 + rs))]

    up_avg, down_avg = up, down
    for delta in deltas[period:]:
        up_val = max(delta, 0)
        down_val = -min(delta, 0)
        up_avg = (up_avg * (period - 1) + up_val) / period
        down_avg = (down_avg * (period - 1) + down_val) / period
        rs = up_avg / down_avg if down_avg != 0 else 0
        rsi_values.append(100 - (100 / (1 + rs)))
    return [np.nan] * (period) + rsi_values

def bullish_rsi_divergence(closes: List[float], lows: List[float]) -> bool:
    rsi_vals = rsi(closes)
    if len(rsi_vals) < 5:
        return False
    price_low1, price_low2 = lows[-5], lows[-1]
    rsi_low1, rsi_low2 = rsi_vals[-5], rsi_vals[-1]
    return price_low2 < price_low1 and rsi_low2 > rsi_low1

# --- RS (relative strength) ---
def relative_strength(closes: List[float]) -> float:
    return closes[-1] / np.mean(closes[-20:]) if len(closes) >= 20 else 0

# --- CVD Proxy ---
def cvd_proxy(closes: List[float], volumes: List[float]) -> float:
    delta = [v if closes[i] > closes[i-1] else -v for i, v in enumerate(volumes) if i > 0]
    return sum(delta)

# --- VOLATILITY & VOLUME ---
def volatility_metric(closes: List[float], win=30) -> float:
    arr = np.array(closes, dtype=float)
    if arr.size < win:
        return 0.0
    mu = float(np.mean(arr[-win:]))
    return float(np.std(arr[-win:]) / mu) if mu else 0.0

def mean_volume(volumes: List[float], win=30) -> float:
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
    out: List[Dict[str, Any]] = []

    async def check_symbol(sym: str):
        async with sem:
            try:
                candles = await get_klines(sym, tf, 80)
                o, h, l, c, v, _ = parse_ohlcv(candles)
                if len(c) < 35:
                    return None

                fvg_info = last_fvg_status(h, l, c)
                if not fvg_info:
                    return None
                status_fvg, gap_bottom, gap_top = fvg_info
                if not bullish_mss_or_bos(h, c):
                    return None
                if c[-1] < gap_bottom:
                    return None

                vol_metric = volatility_metric(c)
                avg_vol = mean_volume(v)
                status = ("HARAM" if sym in haram else
                          "REMOVED" if sym in removed else
                          "WATCHLIST" if sym in watchlist else
                          "NEW")
                rsi_div = bullish_rsi_divergence(c, l)
                rs_val = relative_strength(c)
                cvd_val = cvd_proxy(c, v)

                return {
                    "symbol": sym,
                    "fvg_status": status_fvg,
                    "volatility": vol_metric,
                    "avg_volume": avg_vol,
                    "status": status,
                    "inside_fvg": gap_bottom <= c[-1] <= gap_top,
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

# --- TELEGRAM COMMAND HANDLER ---
TIMEFRAME_MAP = {"1H": "1h", "4H": "4h", "1D": "1d"}

async def fvg_coinlist_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip().upper()
    m = re.match(r'(1H|4H|1D)\s*FVG\s*COIN\s*LIST', text)
    if not m:
        await update.message.reply_text("❌ Invalid command format! উদাহরণ: 4H FVG Coin list", parse_mode=ParseMode.MARKDOWN)
        return
    tf_label = m.group(1)
    tf = TIMEFRAME_MAP.get(tf_label)
    await update.message.reply_text(f"⏳ Scanning Binance ({tf_label}) FVG coins...", parse_mode=ParseMode.MARKDOWN)

    coins = await get_fvg_coins_async(tf)
    if not coins:
        await update.message.reply_text("😔 No coins found.", parse_mode=ParseMode.MARKDOWN)
        return

    coins.sort(key=lambda x: (x["volatility"], x["avg_volume"]), reverse=True)
    lines = [f"🎯 FVG Coins ({tf_label}):", ""]
    for i, c in enumerate(coins, 1):
        extra = []
        if c["rsi_div"]:
            extra.append("📈RSI Div")
        if c["rs_val"] > 1:
            extra.append("💪RS↑")
        if c["cvd_val"] > 0:
            extra.append("📊CVD↑")
        extra_str = " | ".join(extra) if extra else "—"
        lines.append(f"{i:2d}. `{c['symbol']}` [{c['fvg_status']}] | Vol: {c['volatility']:.4f} | AvgVol: {c['avg_volume']:.0f} | {c['status']} | {extra_str}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
