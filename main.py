import os
import asyncio
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
    app_flask.run(host='0.0.0.0', port=PORT)

# ─── COMMANDS ────────────────────────────────
async def cmd_start(update: Update,
                    ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Olasmos FX Bot is LIVE! 🚀")

# ─── MAIN ────────────────────────────────────
async def post_init(app: Application):
    await app.bot.send_message(
        chat_id=CHAT_ID,
        text=(
            "🤖 *OLASMOS FX BOT LIVE!*\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "✅ Connected to Telegram!\n"
            "✅ Running on Render 24/7\n"
            "🚀 Full bot loading soon..."
        ),
        parse_mode="Markdown"
    )
    logger.info("✅ Bot started and message sent!")

def main():
    Thread(target=run_flask, daemon=True).start()
    logger.info(f"Flask running on port {PORT}")

    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
