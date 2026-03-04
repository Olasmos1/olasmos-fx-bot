import os
import asyncio
import logging
from threading import Thread
from flask import Flask
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PORT    = int(os.environ.get("PORT", 8080))

app_flask = Flask(__name__)

@app_flask.route('/')
def home():
    return "Olasmos FX Bot Running! ✅"

@app_flask.route('/health')
def health():
    return {"status": "ok"}, 200

def run_flask():
    app_flask.run(host='0.0.0.0', port=PORT)

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Olasmos FX Bot is LIVE!")

async def main():
    Thread(target=run_flask, daemon=True).start()
    bot = Bot(TOKEN)
    await bot.send_message(chat_id=CHAT_ID,
        text="🤖 OLASMOS FX BOT LIVE!\n✅ Full version loading...")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
