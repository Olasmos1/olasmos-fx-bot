# Test version — checks if bot connects properly
import os
import asyncio
import logging
from telegram import Bot
from flask import Flask
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask keep alive
app = Flask(__name__)

@app.route('/')
def home():
    return "Olasmos FX Bot is Running! ✅"

@app.route('/health')
def health():
    return {"status": "running"}, 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

# Telegram test
async def main():
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token:
        logger.error("No token found!")
        return

    bot = Bot(token=token)
    await bot.send_message(
        chat_id=chat_id,
        text="🤖 Olasmos FX Bot — Test successful!\n✅ Bot is connected and running on Render!"
    )
    logger.info("✅ Bot started successfully!")

    # Keep running
    while True:
        await asyncio.sleep(60)

if __name__ == "__main__":
    # Start Flask in background
    t = Thread(target=run_flask)
    t.daemon = True
    t.start()

    # Start bot
    asyncio.run(main())
