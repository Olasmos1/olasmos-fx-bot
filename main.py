import os
import logging
import pytz
from datetime import datetime
from threading import Thread
from flask import Flask
from telegram import Update
from telegram.ext import (Application, CommandHandler,
                           ContextTypes)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PORT    = int(os.environ.get("PORT", 10000))
WAT     = pytz.timezone("Africa/Lagos")

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

# ─── COMMANDS ────────────────────────────────
async def cmd_start(update: Update,
                    ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *OLASMOS FX BOT*\n\n"
        "✅ Bot is LIVE and running!\n\n"
        "Commands:\n"
        "/start — Welcome\n"
        "/status — Bot status\n"
        "/help — Show commands",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update,
                     ctx: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(WAT).strftime('%H:%M %d/%m/%Y')
    await update.message.reply_text(
        f"📊 *BOT STATUS*\n"
        f"━━━━━━━━━━━━━━\n"
        f"✅ Running on Render 24/7\n"
        f"🕐 WAT: {now}\n"
        f"📡 Monitoring: EURUSD, GBPUSD,\n"
        f"   USDJPY, XAUUSD\n"
        f"🤖 Full version loading...",
        parse_mode="Markdown"
    )

async def cmd_help(update: Update,
                   ctx: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, ctx)

async def post_init(app: Application):
    try:
        await app.bot.send_message(
            chat_id=CHAT_ID,
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
        logger.info("✅ Startup message sent!")
    except Exception as e:
        logger.error(f"Startup error: {e}")

# ─── MAIN ────────────────────────────────────
def main():
    Thread(target=run_flask, daemon=True).start()
    logger.info(f"✅ Flask started on port {PORT}")

    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help",   cmd_help))

    logger.info("🚀 Bot starting polling...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

if __name__ == "__main__":
    main()
