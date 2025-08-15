import re

from telegram import Update as TGUpdate
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)

from config_and_utils import (
    BOT_TOKEN, logger, keep_alive, parse_command, get_username, is_admin
)
from data_api import (
    get_column, add_coin, add_coin_with_date, remove_coin_from_table,
    get_removed_map, get_top_gainers
)
from fvg_coinlist import fvg_coinlist_handler

# =============== Telegram Bot Handlers ===============

async def start(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text(
            f"⛔ Sorry @{username},\nyou do not have permission to use the Crypto Watchlist Tracking Bot."
        )
        return
    await update.message.reply_text(
        "👋 Hello! Commands:\n"
        "- `BTCUSDT Check`, `ETH Remove`, `BNB Haram`, `BTC Add`, `BNB Halal`\n"
        "- `Top Gainer List`\n"
        "- `Volatility Coin List` or `Volatility Coin List 30`\n"
        "- `1H FVG Coin List`, `4H FVG Coin List`, `1D FVG Coin List`",
        parse_mode=ParseMode.MARKDOWN
    )

async def status(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    try:
        watchlist = await get_column("watchlist")
        haram = await get_column("haram")
    except Exception as e:
        await update.message.reply_text(f"❌ Supabase error: {e}")
        return

    wl_total = len(watchlist)
    hr_total = len(haram)

    parts = []
    parts.append(f"📊 Watchlist ({wl_total}):")
    if watchlist:
        parts.append(" ".join(f"`{w}`" for w in watchlist))
    else:
        parts.append("—")
    parts.append("")
    parts.append(f"⚠️ Haram ({hr_total}):")
    if haram:
        parts.append(" ".join(f"`{h}`" for h in haram))
    else:
        parts.append("—")

    reply = "\n".join(parts)
    await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)

async def top_gainer_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text("⛔ You are not authorized to use commands.")
        return
    await update.message.reply_text("⏳ Fetching top gainers, please wait...")
    try:
        gainers = await get_top_gainers(10, filtering_on=False)
        lines = ["🔥 Top 10 Binance USDT Spot Gainers (24h) [Filter OFF]:\n"]
        for i, (coin, pct) in enumerate(gainers, 1):
            lines.append(f"{i}. `{coin}` : {pct:+.2f}%")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to fetch: {e}")

async def volatility_list_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text("⛔ You are not authorized to use commands.")
        return
    text = update.message.text.strip()
    m = re.search(r'volatility\s+coin\s+list\s*(\d+)?', text, flags=re.IGNORECASE)
    top_n = 20
    if m and m.group(1):
        try:
            top_n = max(5, min(100, int(m.group(1))))
        except Exception:
            top_n = 20
    await update.message.reply_text(f"⏳ Computing top {top_n} high-volatility coins (1H)...")
    try:
        from data_api import volatility_coin_list
        items = await volatility_coin_list(top_n)
        lines = [f"🌪️ Top {top_n} High Volatility (1H):", ""]
        for i, (sym, vol) in enumerate(items, 1):
            lines.append(f"{i:>2}. `{sym}` — Vol={vol:.4f}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Failed: {e}")

async def handle_commands(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()

    # Special commands
    if re.search(r'^\s*top\s+gainer\s+list\s*$', text, flags=re.IGNORECASE):
        await top_gainer_handler(update, context)
        return
    if re.search(r'^\s*volatility\s+coin\s+list(?:\s+\d+)?\s*$', text, flags=re.IGNORECASE):
        await volatility_list_handler(update, context)
        return

    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text("⛔ You are not authorized to use commands.")
        return
    logger.info("Command from %s: %s", username, text)
    coins, action = parse_command(text)
    valid_actions = {"Check", "Remove", "Haram", "Add again", "Halal"}
    if not coins or action not in valid_actions:
        await update.message.reply_text(
            "❌ Invalid format. Use:\n"
            "Check, Remove, Haram, Add Again, Halal\n"
            "Example: BTC ETH Check\n"
            "Or send: Top Gainer List / Volatility Coin List",
        )
        return
    try:
        watchlist = await get_column("watchlist")
        haram = await get_column("haram")
        removed_map = await get_removed_map()
    except Exception as e:
        await update.message.reply_text(f"❌ Supabase error: {e}")
        return
    already, added, removed, marked_haram, unharamed = [], [], [], [], []
    for coin in coins:
        if action == "Check":
            if coin in haram:
                marked_haram.append(coin)
            elif coin in removed_map:
                removed.append(f"{coin} - {removed_map[coin]}")
            elif coin in watchlist:
                already.append(coin)
            else:
                await add_coin_with_date("watchlist", coin)
                added.append(coin)
        elif action == "Remove":
            if await remove_coin_from_table("watchlist", coin):
                await add_coin_with_date("removed", coin)
                removed.append(coin)
        elif action == "Haram":
            await add_coin("haram", coin)
            marked_haram.append(coin)
            if coin in watchlist and await remove_coin_from_table("watchlist", coin):
                await add_coin_with_date("removed", coin)
                removed.append(f"{coin} (removed from watchlist due to haram)")
        elif action == "Add again":
            if coin in removed_map:
                await add_coin_with_date("watchlist", coin)
                await remove_coin_from_table("removed", coin)
                added.append(coin)
        elif action == "Halal":
            if await remove_coin_from_table("haram", coin):
                unharamed.append(coin)

    reply_parts = []
    if already:
        reply_parts.append("🟢 Already in Watchlist:\n" + " ".join(f"`{x}`" for x in already))
    if added:
        reply_parts.append("✅ New Added:\n" + " ".join(f"`{x}`" for x in added))
    if marked_haram:
        reply_parts.append("⚠️ Marked as Haram:\n" + " ".join(f"`{x}`" for x in marked_haram))
    if removed:
        reply_parts.append("🗑️ Removed:\n" + " ".join(f"`{x}`" for x in removed))
    if unharamed:
        reply_parts.append("✅ Removed from Haram:\n" + " ".join(f"`{x}`" for x in unharamed))
    reply = "\n\n".join(reply_parts) or "✅ No changes made."
    await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)

# =============== App Bootstrap and Loop ===============

async def post_init(application):
    from data_api import init_sessions
    await init_sessions()
    logger.info("post_init completed: aiohttp session created and background tasks started.")

def main() -> None:
    keep_alive()
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"(?i)^status$"), status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_commands))
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.Regex(r"(?i)^(1H|4H|1D)\s*FVG\s*COIN\s*LIST\s*$"),
            fvg_coinlist_handler
        )
    )
    logger.info("Bot starting (press Ctrl+C to stop)...")
    app.run_polling()

if __name__ == "__main__":
    main()