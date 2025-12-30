# backend/agents/orchestrator.py

from backend.agents.context_agents import (
    TechnicalContext,
    NewsContext,
    MacroContext,
    RiskEvaluator,
)
from backend.agents.synthesis_agent import SynthesisAgent


class SymbolAnalysisOrchestrator:
    """
    Orchestrates context generation + LLM synthesis.
    """

    async def run(self, symbol: str) -> dict:
        # ----- Build contexts -----
        technical = await TechnicalContext().run(symbol)
        news = await NewsContext().run(symbol)
        macro = await MacroContext().run()

        # ----- Optional deterministic evaluation -----
        risk_meta = RiskEvaluator().evaluate(
            technical=technical,
            news=news,
            macro=macro,
        )

        # ----- Prepare text context for LLM -----
        technical_text = f"""
Trend: {technical['trend']}
Momentum: {technical['momentum']}
Support: {technical['support']}
Resistance: {technical['resistance']}
"""

        news_text = news.get("news_context") or "No significant recent news."

        macro_text = macro.get("macro_context") or "Macro conditions neutral."

        # ----- LLM synthesis -----
        synthesis = await SynthesisAgent().run(
            symbol=symbol,
            technical_context=technical_text,
            news_context=news_text,
            macro_context=macro_text,
        )

        # ----- Attach deterministic metadata -----
        synthesis["confidence_meta"] = risk_meta

        return {
            "analysis": synthesis
        }
