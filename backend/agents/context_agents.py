from backend.rag.retrieve import get_context
import yfinance as yf


class TechnicalContext:
    async def run(self, symbol: str) -> dict:
        """
        Deterministic technical snapshot using real price data.
        """

        data = yf.download(
            symbol + ".NS",
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if data.empty or len(data) < 50:
            return {
                "trend": "unknown",
                "support": "-",
                "resistance": "-",
                "momentum": "unknown",
            }

        close = data["Close"]

        # --- Moving averages ---
        ma20_series = close.rolling(20).mean()
        ma50_series = close.rolling(50).mean()

        ma20 = float(ma20_series.iloc[-1].item())
        ma50 = float(ma50_series.iloc[-1].item())
        last_price = float(close.iloc[-1].item())

        # --- Trend logic ---
        if ma20 > ma50:
            trend = "uptrend"
        elif ma20 < ma50:
            trend = "downtrend"
        else:
            trend = "sideways"

        # --- Support / Resistance ---
        recent = close.tail(20)
        support = round(recent.min().item(), 2)
        resistance = round(recent.max().item(), 2)


        # --- Momentum ---
        momentum = "positive" if last_price > ma20 else "weak"

        return {
            "trend": trend,
            "support": str(support),
            "resistance": str(resistance),
            "momentum": momentum,
        }



class NewsContext:
    async def run(self, symbol: str) -> dict:
        context = await get_context(f"{symbol} recent news earnings sentiment")
        return {
            "news_context": context,
            "sentiment": "positive" if context else "neutral",
        }


class MacroContext:
    async def run(self) -> dict:
        context = await get_context("India macro interest rates market sentiment")
        return {
            "macro_context": context,
            "regime": "risk-on" if context else "neutral",
        }


class RiskEvaluator:
    """
    Optional deterministic layer.
    Not used for trade calls — only informational.
    """

    def evaluate(
        self,
        technical: dict,
        news: dict,
        macro: dict,
    ) -> dict:
        score = 0

        if technical.get("trend") == "uptrend":
            score += 1
        if news.get("sentiment") == "positive":
            score += 1
        if macro.get("regime") == "risk-on":
            score += 1

        confidence = min(score * 30 + 10, 90)

        return {
            "confidence_score": confidence,
            "environment": "supportive" if confidence >= 60 else "mixed",
        }
