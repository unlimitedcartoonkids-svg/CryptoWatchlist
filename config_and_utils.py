# config_and_utils.py

import os
import logging
import asyncio
import re
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple, Dict, Any
from zoneinfo import ZoneInfo
from flask import Flask

from dotenv import load_dotenv

# ===================== env & logging =====================
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
CHAT_ID_RAW = os.getenv("CHAT_ID", "").strip()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()

if not BOT_TOKEN or not SUPABASE_URL or not SUPABASE_KEY or not CHAT_ID_RAW:
    raise EnvironmentError("Missing env: BOT_TOKEN/SUPABASE_URL/SUPABASE_KEY/CHAT_ID")

try:
    CHAT_ID: Any = int(CHAT_ID_RAW)
except Exception:
    CHAT_ID = CHAT_ID_RAW

BINANCE_BASE = "https://api.binance.com"
BINANCE_BATCH_SIZE = int(os.getenv("BINANCE_BATCH_SIZE", "15"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))
SLEEP_TIME = float(os.getenv("SLEEP_TIME", "6"))
ALERT_COOLDOWN_MIN = int(os.getenv("ALERT_COOLDOWN_MIN", "30"))
USER_TZ = ZoneInfo(os.getenv("USER_TZ", "Asia/Dhaka"))  # UTC+6
AUTO_ADD_NEW_COINS = os.getenv("AUTO_ADD_NEW_COINS", "false").lower() in ("1","true","yes")

aiohttp_session: Optional["aiohttp.ClientSession"] = None
binance_sem: Optional[asyncio.Semaphore] = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("crypto-scanner")

# ===================== Flask keep alive =====================
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "✅ Bot is alive!"

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

def keep_alive():
    asyncio.get_event_loop().run_in_executor(None, run_flask)

# ===================== Misc & Command Helpers =====================
def parse_command(text: str) -> Tuple[List[str], str]:
    if not text:
        return [], ""
    clean = text.strip()
    tokens = re.split(r'[\s,;]+', clean)
    actions_map = {
        "check": "Check",
        "remove": "Remove",
        "haram": "Haram",
        "add": "Add again",
        "addagain": "Add again",
        "add_again": "Add again",
        "halal": "Halal",
    }
    found_action = ""
    coins = []
    for t in tokens:
        tl = t.lower()
        if tl in actions_map:
            found_action = actions_map[tl]
            continue
        m = re.match(r'^([A-Za-z0-9]{2,10})(?:[/\-]?USDT)?$', t.upper())
        if m:
            base = m.group(1)
            coin = base if base.endswith("USDT") else f"{base}USDT"
            coins.append(coin.upper())
    return coins, found_action

def get_username(update) -> str:
    user = update.effective_user
    return user.username or str(user.id) if user else "Unknown"

def is_admin(username: Optional[str]) -> bool:
    return bool(username) and username.lower() in ["redwanict",]

def now_local_str() -> str:
    return datetime.now(USER_TZ).strftime("%d-%m-%Y %H:%M %Z")