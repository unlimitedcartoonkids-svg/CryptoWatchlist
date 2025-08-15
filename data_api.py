# data_api.py

import asyncio
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone

import aiohttp
from supabase import create_client, Client

from config_and_utils import (
    SUPABASE_URL, SUPABASE_KEY, BINANCE_BASE, BINANCE_API_KEY, BINANCE_API_SECRET,
    logger, MAX_CONCURRENCY, REQUEST_TIMEOUT
)

from binance.client import Client as BinanceClient
from analysis import volatility_metric

# Create Supabase and Binance clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
binance_connector_client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)

aiohttp_session: Optional[aiohttp.ClientSession] = None
binance_sem: Optional[asyncio.Semaphore] = None

# ===================== Supabase async wrappers =====================

async def _to_thread(fn, *a, **kw):
    return await asyncio.to_thread(fn, *a, **kw)

async def get_column(table: str, column: str = "coin") -> List[str]:
    try:
        def query():
            res = supabase.table(table).select(column).execute()
            return res.data or []
        rows = await _to_thread(query)
        return [r[column].upper() for r in rows if r.get(column)]
    except Exception:
        logger.exception("Supabase get_column error")
        return []

async def add_coin(table: str, coin: str) -> None:
    coin_up = coin.upper()
    existing = await get_column(table)
    if coin_up in existing:
        return
    try:
        def insert():
            return supabase.table(table).insert({"coin": coin_up}).execute()
        await _to_thread(insert)
        logger.info("Added %s to %s", coin_up, table)
    except Exception:
        logger.exception("Supabase add_coin error")

async def add_coin_with_date(table: str, coin: str) -> None:
    coin_up = coin.upper()
    existing = await get_column(table)
    if coin_up in existing:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        def insert():
            return supabase.table(table).insert({"coin": coin_up, "timestamp": timestamp}).execute()
        await _to_thread(insert)
        logger.info("Added %s to %s with timestamp", coin_up, table)
    except Exception:
        logger.exception("Supabase add_coin_with_date error")

async def remove_coin_from_table(table: str, coin: str) -> bool:
    coin_up = coin.upper()
    try:
        def delete_fn():
            resp = supabase.table(table).select("id, coin").eq("coin", coin_up).execute()
            rows = resp.data or []
            for r in rows:
                supabase.table(table).delete().eq("id", r["id"]).execute()
            return bool(rows)
        return await _to_thread(delete_fn)
    except Exception:
        logger.exception("Supabase remove_coin_from_table error")
        return False

async def get_removed_map() -> Dict[str, str]:
    try:
        def query():
            res = supabase.table("removed").select("coin, timestamp").execute()
            return res.data or []
        rows = await _to_thread(query)
        return {r["coin"].upper(): r.get("timestamp") for r in rows if r.get("coin")}
    except Exception:
        logger.exception("Supabase get_removed_map error")
        return {}

async def log_to_supabase(symbol: str, reasons_text: str, opinion_text: Optional[str]) -> None:
    coin = symbol.upper()
    ts_iso = datetime.now(timezone.utc).isoformat()
    payload = {"coin": coin, "timestamp": ts_iso, "reasons": reasons_text}
    if opinion_text is not None:
        payload["opinion"] = opinion_text
    try:
        def insert():
            return supabase.table("signals").insert(payload).execute()
        await _to_thread(insert)
        logger.info("Logged to supabase: %s | %s", coin, reasons_text)
    except Exception:
        logger.exception("Supabase log error for %s", coin)

async def fetch_signals_since(since_iso: str) -> List[Dict[str, Any]]:
    try:
        def query():
            return supabase.table("signals").select("*").gte("timestamp", since_iso).execute().data or []
        return await _to_thread(query)
    except Exception:
        logger.exception("Supabase fetch_signals_since error")
        return []

# ===================== Binance REST =====================

class RateLimitError(Exception):
    def __init__(self, status: int, retry_after: Optional[int] = None, msg: Optional[str] = None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(msg or f"Rate limit: {status}")

async def binance_request(path: str, params: Optional[dict] = None) -> Tuple[Any, Optional[str]]:
    global aiohttp_session
    if aiohttp_session is None:
        raise RuntimeError("HTTP session is not initialized")
    url = f"{BINANCE_BASE}{path}"
    params = params or {}
    async with aiohttp_session.get(url, params=params) as resp:
        used_weight = resp.headers.get("X-MBX-USED-WEIGHT-1M") or resp.headers.get("X-MBX-USED-WEIGHT")
        if resp.status == 429:
            retry = resp.headers.get("Retry-After")
            body = await resp.text()
            raise RateLimitError(429, retry_after=int(retry) if retry and retry.isdigit() else None, msg=f"429 {body}")
        if resp.status != 200:
            body = await resp.text()
            raise Exception(f"HTTP {resp.status}: {body}")
        data = await resp.json(content_type=None)
        return data, used_weight

async def get_exchange_info(filtering_on=True):
    data, used = await binance_request("/api/v3/exchangeInfo")
    if used:
        logger.debug("ExchangeInfo weight: %s", used)
    if not filtering_on:
        return data
    symbols = [
        s for s in data.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and not any(x in s["symbol"] for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])
    ]
    data["symbols"] = symbols
    return data

async def get_klines(symbol: str, interval: str, limit: int = 150):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data, used = await binance_request("/api/v3/klines", params=params)
    return data

async def get_ticker_24h(symbol: str) -> Optional[float]:
    try:
        data, _ = await binance_request("/api/v3/ticker/24hr", {"symbol": symbol})
        pct = float(data.get("priceChangePercent", 0.0))
        return pct
    except Exception:
        logger.exception("24h ticker error for %s", symbol)
        return None

async def get_top_gainers(n=10, filtering_on=True) -> List[Tuple[str, float]]:
    exinfo = await get_exchange_info(filtering_on=filtering_on)
    symbols = [
        s["symbol"] for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and (filtering_on is False or not any(x in s["symbol"] for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"]))
    ]
    global binance_sem
    if binance_sem is None:
        binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def get_pct(symbol):
        try:
            if binance_sem is None:
                raise RuntimeError("binance_sem not initialized")
            sem = binance_sem
            async with sem:
                pct = await get_ticker_24h(symbol)
                await asyncio.sleep(0.25)
            return (symbol, pct)
        except Exception:
            return (symbol, None)

    tasks = [get_pct(sym) for sym in symbols]
    all_results = await asyncio.gather(*tasks)
    filtered = [(sym, pct) for sym, pct in all_results if pct is not None]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:n]

def parse_ohlcv(candles: list):
    if not candles:
        return [], [], [], [], [], []
    opens = [float(c[1]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows  = [float(c[3]) for c in candles]
    closes= [float(c[4]) for c in candles]
    volumes=[float(c[5]) for c in candles]
    times = [int(c[0]) for c in candles]
    return opens, highs, lows, closes, volumes, times

# ===================== Binance Official Client For Historical Win Rate =====================

def get_binance_klines_official(symbol: str, interval: str, start_time: int, end_time: int, max_per_req=1000) -> List[list]:
    all_klines = []
    cur_start = start_time
    while cur_start < end_time:
        klines = binance_connector_client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=cur_start,
            endTime=end_time,
            limit=max_per_req,
        )
        if not klines:
            break
        all_klines.extend(klines)
        last_time = klines[-1][0]
        if last_time == cur_start:
            break
        cur_start = last_time + 1
        if len(klines) < max_per_req:
            break
    return all_klines

# ===================== Session Management =====================

async def init_sessions():
    global aiohttp_session, binance_sem
    if aiohttp_session is None:
        aiohttp_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=100, ssl=False),
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            headers={"User-Agent": "CryptoWatchlistBot/2.0 (+contact)"}
        )
    if binance_sem is None:
        binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

# ===================== Volatility Coin List =====================


async def volatility_coin_list(top_n: int = 20) -> List[Tuple[str, float]]:
    exinfo = await get_exchange_info()
    symbols = [
        s["symbol"] for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and not any(x in s["symbol"] for x in ["UPUSDT","DOWNUSDT","BULLUSDT","BEARUSDT"])
    ]
    global binance_sem
    if binance_sem is None:
        binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def vol_for(sym: str) -> Tuple[str,float]:
        try:
            if binance_sem is None:
                raise RuntimeError("Semaphore not initialized")
            sem = binance_sem
            async with sem:
                k = await get_klines(sym, "1h", 180)
                _,_,_,c,_,_ = parse_ohlcv(k)
                v = volatility_metric(c, win=60)  # 60 bars (~2.5 days 1H)
                await asyncio.sleep(0.07)
                return sym, v
        except Exception:
            return sym, 0.0

    tasks = [vol_for(s) for s in symbols]
    res = await asyncio.gather(*tasks)
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:max(1, top_n)]