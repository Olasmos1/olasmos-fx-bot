# ============================================
# KEEP ALIVE — Prevents Render from sleeping
# Runs a tiny web server on port 8080
# Render pings it to keep bot alive 24/7
# ============================================

from flask import Flask
from threading import Thread
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)   # silence Flask logs

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <html>
    <head><title>Olasmos FX Bot</title></head>
    <body style='background:#0a0f1a; color:#F5C842; 
                 font-family:monospace; text-align:center; 
                 padding-top:100px;'>
        <h1>🤖 Olasmos FX Bot</h1>
        <p style='color:#00FF9C'>✅ Bot is running live!</p>
        <p style='color:#4DAFFF'>Monitoring: EURUSD | GBPUSD | USDJPY | XAUUSD</p>
        <p style='color:#666'>Powered by SMC + AI</p>
    </body>
    </html>
    """

@app.route('/health')
def health():
    return {"status": "running", "bot": "Olasmos FX Bot"}, 200

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()
