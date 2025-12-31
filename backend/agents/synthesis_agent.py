# backend/agents/synthesis_agent.py

import json
import re
from typing import Dict, Any

from backend.openrouter import llm_complete


class SynthesisAgent:
    """
    Final synthesis layer.

    Responsibilities:
    - Combine outputs from upstream agents (technical, news, macro)
    - Produce a SINGLE structured intelligence object
    - Enforce deterministic JSON output
    - Apply guardrails & fallbacks for LLM instability
    """

    # ---------- PUBLIC API ----------

    async def run(
        self,
        symbol: str,
        technical_context: str,
        news_context: str,
        macro_context: str,
    ) -> Dict[str, Any]:

        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(
            symbol=symbol,
            technical_context=technical_context,
            news_context=news_context,
            macro_context=macro_context,
        )

        raw = await llm_complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=900,
            temperature=0.2,
        )

        parsed = self._safe_json_parse(raw)

        if not parsed:
            return self._fallback_response(symbol)

        return self._normalize(parsed, symbol)

    # ---------- PROMPTS ----------

    def _system_prompt(self) -> str:
        return """
You are a financial intelligence engine.

CRITICAL RULES (MANDATORY):
- Respond with ONLY valid JSON
- Do NOT include markdown
- Do NOT include explanations
- Do NOT include backticks
- Output must start with { and end with }

You are NOT making predictions or trade calls.
You are presenting structured market information.

If data is uncertain, use neutral language and placeholders.
"""

    def _user_prompt(
        self,
        symbol: str,
        technical_context: str,
        news_context: str,
        macro_context: str,
    ) -> str:
        return f"""
Analyze the stock: {symbol}

TECHNICAL CONTEXT:
{technical_context}

NEWS CONTEXT:
{news_context}

MACRO CONTEXT:
{macro_context}

Return JSON in the following EXACT structure:

{{
  "company_snapshot": {{
    "symbol": "{symbol}",
    "primary_bias": "bullish | neutral | bearish",
    "market_regime": "trending | ranging | volatile | uncertain"
  }},
  "market_structure": {{
    "trend": "string",
    "volatility": "string"
  }},
  "key_levels": {{
    "support": ["string"],
    "resistance": ["string"]
  }},
  "scenario_framework": [
    {{
      "scenario": "string",
      "trigger": "string",
      "focus": "string"
    }}
  ],
  "ai_observations": ["string"],
  "risk_notes": ["string"]
}}
"""

    # ---------- JSON SAFETY ----------

    def _safe_json_parse(self, text: str) -> Dict[str, Any]:
        """
        Attempts to safely parse JSON from LLM output.
        Handles markdown, partial responses, and noise.
        """
        if not text or not text.strip():
            return {}

        cleaned = text.strip()

        # Remove markdown fences if present
        cleaned = re.sub(r"^```json", "", cleaned)
        cleaned = re.sub(r"^```", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Try extracting first JSON object
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return {}

    # ---------- NORMALIZATION ----------

    def _normalize(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Ensures required keys exist so frontend never breaks.
        """

        data.setdefault("company_snapshot", {})
        data.setdefault("market_structure", {})
        data.setdefault("key_levels", {})
        data.setdefault("scenario_framework", [])
        data.setdefault("ai_observations", [])
        data.setdefault("risk_notes", [])

        cs = data["company_snapshot"]
        cs.setdefault("symbol", symbol)
        cs.setdefault("primary_bias", "neutral")
        cs.setdefault("market_regime", "uncertain")

        ms = data["market_structure"]
        ms.setdefault("trend", "unclear")
        ms.setdefault("volatility", "unknown")

        kl = data["key_levels"]
        kl.setdefault("support", [])
        kl.setdefault("resistance", [])

        if not isinstance(data["scenario_framework"], list):
            data["scenario_framework"] = []

        if not isinstance(data["ai_observations"], list):
            data["ai_observations"] = []

        if not isinstance(data["risk_notes"], list):
            data["risk_notes"] = []

        return data

    # ---------- FALLBACK ----------

    def _fallback_response(self, symbol: str) -> Dict[str, Any]:
        """
        Deterministic fallback if LLM output is unusable.
        """
        return {
            "company_snapshot": {
                "symbol": symbol,
                "primary_bias": "neutral",
                "market_regime": "uncertain",
            },
            "market_structure": {
                "trend": "insufficient data",
                "volatility": "unknown",
            },
            "key_levels": {
                "support": [],
                "resistance": [],
            },
            "scenario_framework": [
                {
                    "scenario": "Base case",
                    "trigger": "No strong directional catalyst",
                    "focus": "Observe price behavior and volume",
                }
            ],
            "ai_observations": [
                "Model output could not be parsed reliably.",
                "Returned neutral structured intelligence.",
            ],
            "risk_notes": [
                "LLM responses are probabilistic.",
                "Use analysis as contextual input, not a directive.",
            ],
        }
 