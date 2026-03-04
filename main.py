import os
import logging
import asyncio
from threading import Thread
from flask import Flask
from telegram import Update, Bot
from telegram.ext import (Application, CommandHandler,
                           ContextTypes)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PORT    = int(os.environ.get("PORT", 10000))

# ─── FLASK ───────────────────────────────────
app_flask = Flask(__name__)

@app_flask.route('/')
def home():
    return "🤖 Olasmos FX Bot Running! ✅"

@app_flask.route('/health')
def health():
    return {"status": "ok"}, 200

def run_flask():
    app_flask.run(host='0.0.0.0', port=PORT,
                  use_reloader=False, debug=False)

# ─── BOT ─────────────────────────────────────
async def cmd_start(update: Update,
                    ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Olasmos FX Bot is LIVE! 🚀")

async def cmd_status(update: Update,
                     ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Bot running on Render 24/7!")

async def send_startup(token, chat_id):
    try:
        bot = Bot(token)
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "🤖 *OLASMOS FX BOT LIVE!*\n"
                "━━━━━━━━━━━━━━━━━━━\n"
                "✅ Bot connected!\n"
                "✅ Running 24/7 on Render\n"
                "📡 Monitoring markets...\n"
                "Type /start to begin!"
            ),
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Startup error: {e}")

async def run_bot():
    # Send startup message
    await send_startup(TOKEN, CHAT_ID)

    # Build app
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))

    # Start polling
    await app.initialize()
    await app.start()
    await app.updater.start_polling(
        drop_pending_updates=True)
    logger.info("✅ Bot polling started!")

    # Keep running
    await asyncio.Event().wait()

def main():
    # Start Flask
    Thread(target=run_flask, daemon=True).start()
    logger.info(f"✅ Flask on port {PORT}")

    # Start bot
    asyncio.run(run_bot())

if __name__ == "__main__":
    main()
