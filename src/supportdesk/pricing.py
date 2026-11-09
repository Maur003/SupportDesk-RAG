"""Token prices for the two configured Gemini ids.

The rates are the constants in `config`, read on 2026-10-30. An unknown model
id has no price. This module does not call Gemini.
"""

from datetime import date
from decimal import Decimal

from supportdesk.config import (
    GEMINI_2_5_FLASH,
    GEMINI_2_5_FLASH_INPUT,
    GEMINI_2_5_FLASH_OUTPUT,
    GEMINI_3_8_FLASH,
    GEMINI_3_8_FLASH_INPUT_AFTER,
    GEMINI_3_8_FLASH_INPUT_UNTIL,
    GEMINI_3_8_FLASH_OUTPUT_AFTER,
    GEMINI_3_8_FLASH_OUTPUT_UNTIL,
    GENERATOR_PRICE_UNTIL,
)


def rates(model: str, on: date) -> tuple[Decimal, Decimal]:
    """Return USD per 1,000,000 input tokens and per 1,000,000 output tokens."""

    if model == GEMINI_3_8_FLASH:
        if on <= GENERATOR_PRICE_UNTIL:
            return GEMINI_3_8_FLASH_INPUT_UNTIL, GEMINI_3_8_FLASH_OUTPUT_UNTIL
        return GEMINI_3_8_FLASH_INPUT_AFTER, GEMINI_3_8_FLASH_OUTPUT_AFTER
    if model == GEMINI_2_5_FLASH:
        return GEMINI_2_5_FLASH_INPUT, GEMINI_2_5_FLASH_OUTPUT
    raise ValueError(f"no price for {model}")


def cost_usd(model: str, input_tokens: int, output_tokens: int, on: date) -> Decimal:
    in_rate, out_rate = rates(model, on)
    return (Decimal(input_tokens) * in_rate + Decimal(output_tokens) * out_rate) / Decimal(
        1_000_000
    )


def format_usd(amount: Decimal) -> str:
    return format(amount.quantize(Decimal("0.00000001")), "f")
