from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.agents.orchestrator import SymbolAnalysisOrchestrator
import yfinance as yf
from fastapi import Query
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = FastAPI(title="StockScope AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CSV_PATH = os.path.join(BASE_DIR, "stocks_master.csv")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- Load stock universe ----------
if not os.path.exists(CSV_PATH):
    raise RuntimeError("stocks_master.csv not found")

stocks_df = pd.read_csv(CSV_PATH)

stocks_df["symbol"] = stocks_df["symbol"].str.upper()
stocks_df["name"] = stocks_df.get("name", "").fillna("")

# ---- sector is OPTIONAL ----
if "sector" not in stocks_df.columns:
    stocks_df["sector"] = "Unknown"
else:
    stocks_df["sector"] = stocks_df["sector"].fillna("Unknown")


STOCKS = stocks_df.to_dict(orient="records")
VALID_SYMBOLS = {s["symbol"] for s in STOCKS}

# ---------- Helpers ----------
def get_sector_peers(symbol: str, sector: str, limit: int = 6):
    if not sector or sector == "Unknown":
        return []

    peers = (
        stocks_df[
            (stocks_df["sector"] == sector)
            & (stocks_df["symbol"] != symbol)
        ]
        .head(limit)
        .to_dict(orient="records")
    )
    return peers


# ---------- UI ----------
@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# ---------- Autocomplete ----------
@app.get("/api/search")
def search_stocks(q: str):
    q = q.strip().lower()
    if not q:
        return []

    matches = [
        s for s in STOCKS
        if q in s["symbol"].lower() or q in s["name"].lower()
    ]
    return matches[:10]

# ---------- Stock fundamentals ----------
@app.get("/api/stock/{symbol}")
def get_stock(symbol: str):
    symbol = symbol.upper()

    if symbol not in VALID_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid stock symbol")

    try:
        ticker = yf.Ticker(symbol + ".NS")

        try:
            info = ticker.get_info() or {}
        except Exception:
            info = {}

        hist = ticker.history(period="2d")
        if hist.empty:
            raise HTTPException(status_code=404, detail="No price data")

        last_price = round(float(hist["Close"].iloc[-1]), 2)
        prev_price = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else last_price
        change_pct = round(((last_price - prev_price) / prev_price) * 100, 2) if prev_price else 0

        sector = info.get("sector") or "Unknown"
        industry = info.get("industry") or "Unknown"
        peers = get_sector_peers(symbol, sector)

        return {
            "company": {
                "name": info.get("longName"),
                "symbol": symbol,
                "price": last_price,
                "change_pct": change_pct,
                "website": info.get("website"),
                "sector": sector,
                "industry": industry,
            },
            "metrics": {
                "market_cap": info.get("marketCap"),
                "pe": info.get("trailingPE"),
                "roe": info.get("returnOnEquity"),
                "roce": info.get("returnOnAssets"),
                "book_value": info.get("bookValue"),
                "dividend_yield": info.get("dividendYield"),
                "high_low": f"{info.get('fiftyTwoWeekHigh')} / {info.get('fiftyTwoWeekLow')}",
            },
            "sector_overview": {
                "sector": sector,
                "industry": industry,
                "peers": peers,
            },
            "about": info.get("longBusinessSummary"),
        }

    except HTTPException:
        raise
    except Exception as e:
        print("Stock API error:", e)
        raise HTTPException(status_code=500, detail="Failed to fetch stock data")

# ---------- AI Insights ----------
@app.get("/api/ai/insights/{symbol}")
async def ai_insights(symbol: str):
    symbol = symbol.upper()
    if symbol not in VALID_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid stock symbol")

    orchestrator = SymbolAnalysisOrchestrator()
    return await orchestrator.run(symbol)

from math import isnan

@app.get("/api/chart/{symbol}")
def chart_data(symbol: str, period: str = "1y"):
    try:
        ticker = yf.Ticker(symbol.upper() + ".NS")

        hist = ticker.history(period=period)

        if hist.empty:
            raise HTTPException(status_code=404, detail="No chart data")

        close = hist["Close"]

        dma50 = close.rolling(50).mean()
        dma200 = close.rolling(200).mean()

        def safe(v):
            if v is None:
                return None
            v = float(v)
            return None if isnan(v) else round(v, 2)

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in close.index],
            "price": [safe(v) for v in close],
            "dma50": [safe(v) for v in dma50],
            "dma200": [safe(v) for v in dma200],
        }

    except HTTPException:
        raise
    except Exception as e:
        print("Chart API error:", e)
        raise HTTPException(status_code=500, detail="Failed to load chart")

