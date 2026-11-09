"""Measured cosine distance between the Fiber 500 and Fiber 1000 pages.

These are the opening plan paragraphs and the two gold price questions.
The numbers are from BAAI/bge-small-en-v1.5 at the pinned revision.
"""

import pytest

from supportdesk.cache import cosine_distance
from supportdesk.config import get_settings
from supportdesk.embed_text import embedding_input
from supportdesk.embedder import BgeEmbedder

FIBER_500_QUESTION = "How much does the Fiber 500 plan cost per month?"
FIBER_1000_QUESTION = "What is the monthly price of Fiber 1000?"
FIBER_500_TEXT = (
    "The Fiber 500 plan costs 79 dollars per month. It includes unlimited data at speeds "
    "up to 500 Mbps download and 500 Mbps upload on the Northline wireline network. "
    "A residential gateway is included for the length of the subscription. The plan is "
    "month to month unless the customer signs a separate term agreement. Taxes are billed "
    "in addition to the 79 dollar charge. The plan does not include mobile service."
)
FIBER_1000_TEXT = (
    "The Fiber 1000 plan costs 99 dollars per month. Download and upload speeds are up to "
    "1000 Mbps on fiber. The same residential gateway used on Fiber 500 is included. "
    "Customers who already have Fiber 500 can upgrade without a truck roll when the optical "
    "terminal supports it. The 99 dollar price is the recurring service charge and does not "
    "include taxes or one-time installation."
)
# Measured 2026-11-08 with the pinned BGE revision. Both sit above the old 0.05 cutoff.
QUERY_DISTANCE = 0.154993
PASSAGE_DISTANCE = 0.081194


def test_fiber_500_and_fiber_1000_distances() -> None:
    settings = get_settings()
    embedder = BgeEmbedder(settings.embedding_model, settings.embedding_revision)
    query_distance = cosine_distance(
        embedder.embed_query(FIBER_500_QUESTION),
        embedder.embed_query(FIBER_1000_QUESTION),
    )
    passages = embedder.embed_passages(
        [
            embedding_input(
                "Fiber 500 residential plan", "Fiber 500 residential plan", FIBER_500_TEXT
            ),
            embedding_input(
                "Fiber 1000 residential plan", "Fiber 1000 residential plan", FIBER_1000_TEXT
            ),
        ]
    )
    passage_distance = cosine_distance(passages[0], passages[1])
    assert query_distance == pytest.approx(QUERY_DISTANCE, abs=1e-5)
    assert passage_distance == pytest.approx(PASSAGE_DISTANCE, abs=1e-5)
    assert query_distance > 0.05
    assert passage_distance > 0.05
