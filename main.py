import os
import logging
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
                  use_reloader=False,
                  debug=False)

# ─── COMMANDS ────────────────────────────────
async def cmd_start(update: Update,
                    ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Olasmos FX Bot is LIVE! 🚀")

async def cmd_status(update: Update,
                     ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Bot is running on Render 24/7!")

# ─── POST INIT ───────────────────────────────
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
        logger.error(f"Startup message error: {e}")

# ─── MAIN ────────────────────────────────────
def main():
    # Start Flask in background thread
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info(f"✅ Flask started on port {PORT}")

    # Build Telegram app
    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    # Add handlers
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))

    # Start polling
    logger.info("🚀 Starting Telegram polling...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False
    )

if __name__ == "__main__":
    main()
