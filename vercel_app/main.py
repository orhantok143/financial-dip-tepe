# vercel_app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
from production.professional_trader import ProfessionalLiveTrader

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trader instance
trader = None
trader_thread = None

@app.on_event("startup")
async def startup_event():
    global trader
    trader = ProfessionalLiveTrader(initial_balance=10000.0)

@app.get("/")
async def root():
    return {
        "message": "🤖 Financial Dip-Tope Trading API",
        "status": "running",
        "deployed_on": "Vercel"
    }

@app.get("/api/start")
async def start_trading():
    def run_trader():
        trader.start_professional_trading()
    
    global trader_thread
    if trader_thread is None or not trader_thread.is_alive():
        trader_thread = threading.Thread(target=run_trader)
        trader_thread.daemon = True
        trader_thread.start()
        return {"status": "started", "message": "Trading started"}
    return {"status": "already_running"}

@app.get("/api/stop")
async def stop_trading():
    if trader:
        trader.is_running = False
        return {"status": "stopped", "message": "Trading stopped"}
    return {"status": "not_running"}

@app.get("/api/status")
async def get_status():
    if trader and hasattr(trader, 'signal_history') and trader.signal_history:
        last_signal = trader.signal_history[-1]
        return {
            "status": "active",
            "last_signal": last_signal,
            "total_signals": len(trader.signal_history)
        }
    return {"status": "inactive"}

# Vercel için gereklidir
app = app