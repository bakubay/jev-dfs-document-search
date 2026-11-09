"""Pinned, explicit pricing assumptions for reproducible cost estimates."""

from __future__ import annotations

from typing import Any

PRICING_AS_OF = "2026-11-04"
JEV_INPUT_USD_PER_MILLION = 0.042
SOL_INPUT_USD_PER_MILLION = 2.00
SOL_OUTPUT_USD_PER_MILLION = 10.00


def estimate_query_cost(usage: dict[str, Any]) -> dict[str, Any]:
    jev = float(usage.get("jev_input_tokens", 0)) / 1_000_000 * JEV_INPUT_USD_PER_MILLION
    openai_input = sum(float(usage.get(f"{role}_input_tokens", 0))
                       for role in ("writer", "selector", "navigation", "agent"))
    openai_output = sum(float(usage.get(f"{role}_output_tokens", 0))
                        for role in ("writer", "selector", "navigation", "agent"))
    sol = (
        openai_input / 1_000_000 * SOL_INPUT_USD_PER_MILLION
        + openai_output / 1_000_000 * SOL_OUTPUT_USD_PER_MILLION
    )
    return {
        "estimated_query_cost_usd": round(jev + sol, 8),
        "estimated_jev_cost_usd": round(jev, 8),
        "estimated_sol_cost_usd": round(sol, 8),
        "pricing_as_of": PRICING_AS_OF,
        "assumptions": {
            "jev_input_usd_per_million": JEV_INPUT_USD_PER_MILLION,
            "jev_output_usd_per_million": 0.0,
            "sol_short_context_standard_input_usd_per_million": SOL_INPUT_USD_PER_MILLION,
            "sol_short_context_standard_output_usd_per_million": SOL_OUTPUT_USD_PER_MILLION,
        },
    }

