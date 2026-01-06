from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from agents.orchestrator import SymbolAnalysisOrchestrator
import yfinance as yf
from fastapi import Query
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from math import isnan
import json
import threading
from typing import List, Dict, Optional
from functools import lru_cache
import math


def clean_nans(obj):
    """
    Recursively replace NaN, inf, and -inf values with None for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'item'):  # numpy types
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj


app = FastAPI(title="StockScope AI")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CSV_PATH = os.path.join(BASE_DIR, "stocks_master.csv")
CACHE_PATH = os.path.join(BASE_DIR, "peer_cache.json")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- Load stock universe ----------
if not os.path.exists(CSV_PATH):
    raise RuntimeError("stocks_master.csv not found")


stocks_df = pd.read_csv(CSV_PATH)
stocks_df["symbol"] = stocks_df["symbol"].str.upper()
stocks_df["name"] = stocks_df.get("name", "").fillna("")

if "sector" not in stocks_df.columns:
    stocks_df["sector"] = "Unknown"
else:
    stocks_df["sector"] = stocks_df["sector"].fillna("Unknown")

STOCKS = stocks_df.to_dict(orient="records")
VALID_SYMBOLS = {s["symbol"] for s in STOCKS}


# ============================================================
# DYNAMIC PEER MANAGER
# ============================================================

class DynamicPeerManager:
    """
    Automatically builds and maintains peer groups for all stocks
    without manual sector definitions.
    """

    def __init__(self, csv_path: str, cache_path: str):
        self.csv_path = csv_path
        self.cache_path = cache_path
        self.peer_cache = {}
        self.industry_groups = {}
        self.sector_groups = {}
        self.stock_metadata = {}
        self.last_refresh = None
        self.cache_duration = timedelta(days=7)
        self._lock = threading.Lock()

        if self._load_from_cache():
            print("✅ Peer database loaded from cache")
        else:
            print("⏳ Building peer database from scratch (this may take 5-10 minutes)...")
            self._build_peer_database()

    def _load_from_cache(self) -> bool:
        """Load peer database from cache file if recent enough"""
        try:
            if not os.path.exists(self.cache_path):
                return False

            with open(self.cache_path, "r") as f:
                cache_data = json.load(f)

            last_refresh = datetime.fromisoformat(cache_data["last_refresh"])

            if datetime.now() - last_refresh > self.cache_duration:
                print("⚠️  Cache expired, rebuilding...")
                return False

            self.industry_groups = cache_data["industry_groups"]
            self.sector_groups = cache_data["sector_groups"]
            self.stock_metadata = cache_data["stock_metadata"]
            self.last_refresh = last_refresh

            return True

        except Exception as e:
            print(f"Error loading cache: {e}")
            return False

    def _build_peer_database(self):
        """Scans all stocks from CSV and builds sector/industry mappings."""
        stocks_df = pd.read_csv(self.csv_path)
        total_stocks = len(stocks_df)

        industry_groups = {}
        sector_groups = {}
        stock_metadata = {}

        processed = 0
        errors = 0

        for idx, row in stocks_df.iterrows():
            symbol = row.get("symbol", "").upper()
            if not symbol:
                continue

            try:
                ticker = yf.Ticker(symbol + ".NS")
                info = ticker.get_info() or {}

                industry = (info.get("industry") or "").strip()
                sector = (info.get("sector") or "").strip()

                if not industry and not sector:
                    errors += 1
                    continue

                stock_metadata[symbol] = {
                    "name": info.get("longName") or symbol,
                    "industry": industry,
                    "sector": sector,
                    "market_cap": info.get("marketCap"),
                    "last_updated": datetime.now().isoformat()
                }

                if industry:
                    if industry not in industry_groups:
                        industry_groups[industry] = []
                    industry_groups[industry].append({
                        "symbol": symbol,
                        "name": info.get("longName") or symbol,
                        "sector": sector,
                        "market_cap": info.get("marketCap"),
                    })

                if sector:
                    if sector not in sector_groups:
                        sector_groups[sector] = []
                    sector_groups[sector].append({
                        "symbol": symbol,
                        "name": info.get("longName") or symbol,
                        "industry": industry,
                        "market_cap": info.get("marketCap"),
                    })

                processed += 1

                if (idx + 1) % 50 == 0:
                    print(f"Progress: {idx + 1}/{total_stocks} stocks ({(idx+1)/total_stocks*100:.1f}%)")

            except Exception as e:
                errors += 1
                if errors % 20 == 0:
                    print(f"Errors encountered: {errors} (continuing...)")
                continue

        self.industry_groups = industry_groups
        self.sector_groups = sector_groups
        self.stock_metadata = stock_metadata
        self.last_refresh = datetime.now()

        self._save_to_cache()

        print(f"\n✅ Peer database built successfully!")
        print(f"   - Total stocks processed: {processed}/{total_stocks}")
        print(f"   - Industries discovered: {len(industry_groups)}")
        print(f"   - Sectors discovered: {len(sector_groups)}")
        print(f"   - Errors: {errors}")

    def _save_to_cache(self):
        """Save peer database to disk for faster subsequent loads"""
        try:
            cache_data = {
                "industry_groups": self.industry_groups,
                "sector_groups": self.sector_groups,
                "stock_metadata": self.stock_metadata,
                "last_refresh": self.last_refresh.isoformat()
            }

            with open(self.cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            print(f"💾 Cache saved to {self.cache_path}")

        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def get_peers(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        Enhanced peer matching with comprehensive financial metrics including quarterly data
        """
        symbol = symbol.upper()

        if symbol in self.stock_metadata:
            stock_data = self.stock_metadata[symbol]
            industry = stock_data["industry"]
            sector = stock_data["sector"]
        else:
            try:
                ticker = yf.Ticker(symbol + ".NS")
                info = ticker.get_info() or {}
                industry = info.get("industry", "")
                sector = info.get("sector", "")
            except:
                return []

        peers = []

        if industry and industry in self.industry_groups:
            industry_peers = [p for p in self.industry_groups[industry] if p["symbol"] != symbol]
            industry_peers.sort(key=lambda x: x.get("market_cap") or 0, reverse=True)
            peers.extend(industry_peers[:limit])

        if len(peers) < limit and sector and sector in self.sector_groups:
            sector_peers = [
                p for p in self.sector_groups[sector] 
                if p["symbol"] != symbol and p["symbol"] not in [x["symbol"] for x in peers]
            ]
            sector_peers.sort(key=lambda x: x.get("market_cap") or 0, reverse=True)
            peers.extend(sector_peers[:limit - len(peers)])

        enriched_peers = []
        for peer in peers[:limit]:
            try:
                t = yf.Ticker(peer["symbol"] + ".NS")
                info = t.get_info() or {}

                quarterly_financials = t.quarterly_financials

                quarterly_revenue = None
                quarterly_revenue_growth = None
                quarterly_profit = None
                quarterly_profit_growth = None

                if quarterly_financials is not None and not quarterly_financials.empty:
                    try:
                        if "Total Revenue" in quarterly_financials.index:
                            revenues = quarterly_financials.loc["Total Revenue"]
                            if len(revenues) >= 2:
                                quarterly_revenue = revenues.iloc[0] / 10000000
                                prev_revenue = revenues.iloc[1] / 10000000
                                quarterly_revenue_growth = ((quarterly_revenue - prev_revenue) / prev_revenue) * 100

                        if "Net Income" in quarterly_financials.index:
                            profits = quarterly_financials.loc["Net Income"]
                            if len(profits) >= 2:
                                quarterly_profit = profits.iloc[0] / 10000000
                                prev_profit = profits.iloc[1] / 10000000
                                if prev_profit != 0:
                                    quarterly_profit_growth = ((quarterly_profit - prev_profit) / prev_profit) * 100
                    except:
                        pass

                current_price = info.get("currentPrice") or info.get("regularMarketPrice")
                pe_ratio = info.get("trailingPE")
                market_cap = info.get("marketCap")
                dividend_yield = info.get("dividendYield")
                roe = info.get("returnOnEquity")
                roce = info.get("returnOnAssets")

                enriched_peers.append({
                    "symbol": peer["symbol"],
                    "name": info.get("longName") or peer.get("name") or peer["symbol"],
                    "current_price": round(current_price, 2) if current_price else None,
                    "pe": round(pe_ratio, 2) if pe_ratio else None,
                    "market_cap": market_cap,
                    "market_cap_cr": round(market_cap / 10000000, 2) if market_cap else None,
                    "dividend_yield": round(dividend_yield * 100, 2) if dividend_yield else None,
                    "roe": round(roe * 100, 2) if roe else None,
                    "roce": round(roce * 100, 2) if roce else None,
                    "quarterly_revenue": round(quarterly_revenue, 2) if quarterly_revenue else None,
                    "quarterly_revenue_growth": round(quarterly_revenue_growth, 2) if quarterly_revenue_growth else None,
                    "quarterly_profit": round(quarterly_profit, 2) if quarterly_profit else None,
                    "quarterly_profit_growth": round(quarterly_profit_growth, 2) if quarterly_profit_growth else None,
                })
            except Exception as e:
                enriched_peers.append({
                    "symbol": peer["symbol"],
                    "name": peer.get("name") or peer["symbol"],
                    "current_price": None,
                    "pe": None,
                    "market_cap": peer.get("market_cap"),
                    "market_cap_cr": round(peer.get("market_cap") / 10000000, 2) if peer.get("market_cap") else None,
                    "dividend_yield": None,
                    "roe": None,
                    "roce": None,
                    "quarterly_revenue": None,
                    "quarterly_revenue_growth": None,
                    "quarterly_profit": None,
                    "quarterly_profit_growth": None,
                })
                continue

        return enriched_peers

    def get_stats(self) -> Dict:
        """Return statistics about the peer database"""
        return {
            "total_industries": len(self.industry_groups),
            "total_sectors": len(self.sector_groups),
            "total_stocks": len(self.stock_metadata),
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "cache_expires": (self.last_refresh + self.cache_duration).isoformat() if self.last_refresh else None
        }


print("\n" + "="*60)
print("INITIALIZING DYNAMIC PEER MANAGER")
print("="*60 + "\n")

peer_manager = DynamicPeerManager(CSV_PATH, CACHE_PATH)

print("\n" + "="*60)
print("PEER MANAGER READY")
print("="*60 + "\n")


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

        peers = peer_manager.get_peers(symbol, limit=6)

        # ---------- Quarterly Results ----------
        qf = ticker.quarterly_financials
        quarterly_results = None

        if qf is not None and not qf.empty:
            qf = qf.iloc[:, :8]

            def safe_row(name):
                if name not in qf.index:
                    return []
                return [
                    round(v, 2) if pd.notna(v) else None
                    for v in qf.loc[name].values
                ]

            quarterly_results = {
                "currency": "INR Crores",
                "quarters": [d.strftime("%b %Y") for d in qf.columns],
                "rows": {
                    "Sales": safe_row("Total Revenue"),
                    "Expenses": safe_row("Total Expenses"),
                    "Operating Profit": safe_row("Operating Income"),
                    "Net Profit": safe_row("Net Income"),
                    "EPS": safe_row("Diluted EPS"),
                }
            }

        # ---------- Current Stock Quarterly Metrics ----------
        current_stock_quarterly = {
            "quarterly_revenue": None,
            "quarterly_revenue_growth": None,
            "quarterly_profit": None,
            "quarterly_profit_growth": None,
        }

        if qf is not None and not qf.empty:
            try:
                if "Total Revenue" in qf.index:
                    revenues = qf.loc["Total Revenue"]
                    if len(revenues) >= 2:
                        current_stock_quarterly["quarterly_revenue"] = round(revenues.iloc[0] / 10000000, 2)
                        prev_revenue = revenues.iloc[1] / 10000000
                        current_stock_quarterly["quarterly_revenue_growth"] = round(
                            ((revenues.iloc[0] / 10000000 - prev_revenue) / prev_revenue) * 100, 2
                        )

                if "Net Income" in qf.index:
                    profits = qf.loc["Net Income"]
                    if len(profits) >= 2:
                        current_stock_quarterly["quarterly_profit"] = round(profits.iloc[0] / 10000000, 2)
                        prev_profit = profits.iloc[1] / 10000000
                        if prev_profit != 0:
                            current_stock_quarterly["quarterly_profit_growth"] = round(
                                ((profits.iloc[0] / 10000000 - prev_profit) / prev_profit) * 100, 2
                            )
            except:
                pass

        # ---------- Format Metrics ----------
        market_cap = info.get("marketCap")
        market_cap_formatted = None
        if market_cap:
            if market_cap >= 1e9:
                market_cap_formatted = f"₹{market_cap/1e9:.2f}B"
            elif market_cap >= 1e7:
                market_cap_formatted = f"₹{market_cap/1e7:.2f}Cr"
            else:
                market_cap_formatted = f"₹{market_cap:.2f}"

        pe_ratio = info.get("trailingPE")
        pe_formatted = round(pe_ratio, 2) if pe_ratio else None

        roe = info.get("returnOnEquity")
        roe_formatted = round(roe * 100, 2) if roe else None

        roce = info.get("returnOnAssets")
        roce_formatted = round(roce * 100, 2) if roce else None

        dividend_yield = info.get("dividendYield")
        dividend_formatted = round(dividend_yield * 100, 2) if dividend_yield else None

        result = {
            "company": {
                "name": info.get("longName") or symbol,
                "symbol": symbol,
                "price": last_price,
                "change_pct": change_pct,
                "website": info.get("website"),
                "sector": sector,
                "industry": industry,
                "is_listed": info.get("quoteType") == "EQUITY"
            },
            "metrics": {
                "market_cap": market_cap_formatted,
                "market_cap_cr": round(market_cap / 10000000, 2) if market_cap else None,
                "pe": pe_formatted,
                "roe": roe_formatted,
                "roce": roce_formatted,
                "book_value": info.get("bookValue"),
                "dividend_yield": dividend_formatted,
                "high_low": f"{info.get('fiftyTwoWeekHigh')} / {info.get('fiftyTwoWeekLow')}",
            },
            "sector_overview": {
                "sector": sector,
                "industry": industry,
                "peers": peers,
            },
            "quarterly_metrics": current_stock_quarterly,
            "quarterly_results": quarterly_results,
            "about": info.get("longBusinessSummary"),
        }

        return clean_nans(result)

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


# ---------- Chart ----------
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


# ---------- Peer Database Stats ----------
@app.get("/api/peers/stats")
def peer_stats():
    """Returns statistics about the peer database"""
    return peer_manager.get_stats()

