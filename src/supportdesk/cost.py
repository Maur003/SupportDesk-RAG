"""Cost of 100 scripted questions before and after the cache and the cheap model.

Token counts are fixed inputs. This process does not call Gemini and does not
read a live usage report. Cache hits add no generation cost. The access scope
on every lookup is `public`.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from supportdesk.cache import ScopeCache, access_scope
from supportdesk.config import get_settings
from supportdesk.pricing import cost_usd, format_usd, rates
from supportdesk.routing import choose_model

BILLED_AS_OF = date(2026, 11, 8)
ASSUMED_INPUT_TOKENS = 200
ASSUMED_OUTPUT_TOKENS = 80
N_LONG = 45
N_SEMANTIC = 5
N_LONG_REPEATS = 20
N_SHORT = 20
N_SHORT_REPEATS = 10
SIMULATED_CORPUS = "cost-sim"
SIMULATED_SEMANTIC_DISTANCE = 0.05


@dataclass(frozen=True)
class CostReport:
    billed_as_of: date
    n: int
    assumed_input_tokens: int
    assumed_output_tokens: int
    scope: str
    exact_hits: int
    semantic_hits: int
    cheap_billed: int
    generator_billed: int
    cost_before: Decimal
    cost_after: Decimal
    generator_model: str
    cheap_model: str
    generator_in: Decimal
    generator_out: Decimal
    cheap_in: Decimal
    cheap_out: Decimal
    generator_in_after: Decimal
    generator_out_after: Decimal
    simple_query_max_chars: int
    semantic_cache_distance: float


def workload() -> list[tuple[str, list[float]]]:
    long_queries = [_long(index) for index in range(N_LONG)]
    short_queries = [f"short {index:03d}" for index in range(N_SHORT)]
    items = [(query, _one_hot(index)) for index, query in enumerate(long_queries)]
    items.extend((_partner(index), _one_hot(index)) for index in range(N_SEMANTIC))
    items.extend((long_queries[index], _one_hot(index)) for index in range(N_LONG_REPEATS))
    items.extend((query, _one_hot(index)) for index, query in enumerate(short_queries))
    items.extend((short_queries[index], _one_hot(index)) for index in range(N_SHORT_REPEATS))
    return items


def simulate() -> CostReport:
    settings = get_settings()
    scope = access_scope("customer")
    cache = ScopeCache(SIMULATED_SEMANTIC_DISTANCE)
    exact_hits = 0
    semantic_hits = 0
    cheap_billed = 0
    generator_billed = 0
    after = Decimal(0)
    for query, vector in workload():
        model = choose_model(
            query,
            simple_max_chars=settings.simple_query_max_chars,
            cheap_model=settings.cheap_model,
            generator_model=settings.generator_model,
        )
        kind = cache.lookup(query, scope, model, vector, SIMULATED_CORPUS)
        if kind == "exact":
            exact_hits += 1
            continue
        if kind == "semantic":
            semantic_hits += 1
            continue
        cache.store(query, scope, model, vector, SIMULATED_CORPUS)
        after += cost_usd(model, ASSUMED_INPUT_TOKENS, ASSUMED_OUTPUT_TOKENS, BILLED_AS_OF)
        if model == settings.cheap_model:
            cheap_billed += 1
        else:
            generator_billed += 1
    n = exact_hits + semantic_hits + cheap_billed + generator_billed
    before = (
        cost_usd(
            settings.generator_model, ASSUMED_INPUT_TOKENS, ASSUMED_OUTPUT_TOKENS, BILLED_AS_OF
        )
        * n
    )
    generator_in, generator_out = rates(settings.generator_model, BILLED_AS_OF)
    cheap_in, cheap_out = rates(settings.cheap_model, BILLED_AS_OF)
    later_in, later_out = rates(settings.generator_model, date(2027, 1, 1))
    return CostReport(
        billed_as_of=BILLED_AS_OF,
        n=n,
        assumed_input_tokens=ASSUMED_INPUT_TOKENS,
        assumed_output_tokens=ASSUMED_OUTPUT_TOKENS,
        scope=scope,
        exact_hits=exact_hits,
        semantic_hits=semantic_hits,
        cheap_billed=cheap_billed,
        generator_billed=generator_billed,
        cost_before=before,
        cost_after=after,
        generator_model=settings.generator_model,
        cheap_model=settings.cheap_model,
        generator_in=generator_in,
        generator_out=generator_out,
        cheap_in=cheap_in,
        cheap_out=cheap_out,
        generator_in_after=later_in,
        generator_out_after=later_out,
        simple_query_max_chars=settings.simple_query_max_chars,
        semantic_cache_distance=SIMULATED_SEMANTIC_DISTANCE,
    )


def render(report: CostReport | None = None) -> str:
    report = simulate() if report is None else report
    lines = [
        (
            f"prices_read=2026-10-30 billed_as_of={report.billed_as_of.isoformat()} "
            f"n={report.n} assumed_input_tokens={report.assumed_input_tokens} "
            f"assumed_output_tokens={report.assumed_output_tokens} "
            f"scope={report.scope} token_source=assumed gemini_calls=0"
        ),
        (
            f"generator={report.generator_model} usd_per_1m_in={report.generator_in} "
            f"usd_per_1m_out={report.generator_out}"
        ),
        (
            f"cheap={report.cheap_model} usd_per_1m_in={report.cheap_in} "
            f"usd_per_1m_out={report.cheap_out}"
        ),
        (
            "generator_after_2026-12-31 "
            f"usd_per_1m_in={report.generator_in_after} usd_per_1m_out={report.generator_out_after}"
        ),
        (
            "simulation=true api_semantic_cache=disabled "
            f"simple_query_max_chars={report.simple_query_max_chars} "
            f"simulated_semantic_cache_distance={report.semantic_cache_distance}"
        ),
        (
            f"exact_hits={report.exact_hits} semantic_hits={report.semantic_hits} "
            f"cheap_billed={report.cheap_billed} generator_billed={report.generator_billed}"
        ),
        (
            f"cost_per_100_before={format_usd(report.cost_before)} "
            f"cost_per_100_after={format_usd(report.cost_after)}"
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    print(render(), end="")


def _long(index: int) -> str:
    return f"northline billing question number {index:03d} " + ("detail " * 12)


def _partner(index: int) -> str:
    return f"northline semantic partner number {index:03d} " + ("other " * 12)


def _one_hot(index: int, size: int = N_LONG) -> list[float]:
    vector = [0.0] * size
    vector[index] = 1.0
    return vector


if __name__ == "__main__":
    main()
