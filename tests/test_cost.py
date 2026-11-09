import json
import logging
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from supportdesk.auth import issue_token
from supportdesk.cache import (
    ScopeCache,
    access_scope,
    exact_cache_key,
    lookup_exact,
    lookup_semantic,
    normalize_query,
    store_answer,
)
from supportdesk.compress import ContextChunk, compress_context
from supportdesk.config import GEMINI_2_5_FLASH, GEMINI_3_8_FLASH, get_settings
from supportdesk.cost import render, simulate
from supportdesk.db import make_session_factory
from supportdesk.llm import ModelTurn
from supportdesk.main import create_app
from supportdesk.models import Chunk, Document, RequestLog, ResponseCache
from supportdesk.pricing import cost_usd, format_usd, rates
from supportdesk.routing import choose_model

LONG = "northline billing question number 000 " + ("detail " * 12)
PARTNER = "northline semantic partner number 000 " + ("other " * 12)
VECTOR = [1.0, *([0.0] * 383)]
OTHER = [0.0, 1.0, *([0.0] * 382)]
AS_OF = date(2026, 11, 8)
FINANCE_ANSWER = "FINANCE_CACHE_ONLY"


class _Embedder:
    def embed_query(self, text: str) -> list[float]:
        return list(VECTOR)


class _Reranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.5 for _passage in passages]


class _Counter:
    def __init__(
        self,
        text: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def generate(self, messages: list[dict]) -> ModelTurn:
        self.calls += 1
        return ModelTurn(
            self.text,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


def test_exact_key_includes_scope_and_ignores_case_and_spacing() -> None:
    assert normalize_query("  Hello   WORLD ") == "hello world"
    assert exact_cache_key("Hello", "public", "m") == exact_cache_key("  hello ", "public", "m")
    assert exact_cache_key("hello", "public", "m") != exact_cache_key(
        "hello", access_scope("finance"), "m"
    )
    assert access_scope("customer") == "public"
    assert access_scope("finance") == "agent,finance,public"


def test_prices_step_up_after_the_recorded_date() -> None:
    assert rates(GEMINI_3_8_FLASH, date(2026, 12, 31)) == (Decimal("0.75"), Decimal("3.75"))
    assert rates(GEMINI_3_8_FLASH, date(2027, 1, 1)) == (Decimal("1.50"), Decimal("7.50"))
    assert rates(GEMINI_2_5_FLASH, date(2027, 1, 1)) == (Decimal("0.30"), Decimal("2.50"))
    with pytest.raises(ValueError, match="no price"):
        rates("gemini-other", AS_OF)


def test_short_question_uses_the_cheap_model() -> None:
    assert (
        choose_model(
            "short 000",
            simple_max_chars=80,
            cheap_model=GEMINI_2_5_FLASH,
            generator_model=GEMINI_3_8_FLASH,
        )
        == GEMINI_2_5_FLASH
    )
    assert (
        choose_model(
            LONG,
            simple_max_chars=80,
            cheap_model=GEMINI_2_5_FLASH,
            generator_model=GEMINI_3_8_FLASH,
        )
        == GEMINI_3_8_FLASH
    )


def test_compress_keeps_the_higher_score_inside_the_cap() -> None:
    chunks = [
        ContextChunk("a", "public", "a", 0.1, 100),
        ContextChunk("b", "public", "b", 0.9, 100),
        ContextChunk("a", "public", "a2", 0.4, 100),
        ContextChunk("c", "public", "c", 0.2, 100),
    ]
    kept = compress_context(chunks, max_tokens=150, min_score=None)
    assert [item.id for item in kept] == ["b"]
    floored = compress_context(
        [
            ContextChunk("a", "public", "a", 0.2, 10),
            ContextChunk("b", "public", "b", 0.8, 10),
        ],
        max_tokens=2000,
        min_score=0.5,
    )
    assert [item.id for item in floored] == ["b"]
    five = [
        ContextChunk(str(index), "public", "t", score, 400)
        for index, score in enumerate((0.1, 0.2, 0.3, 0.4, 0.5))
    ]
    assert len(compress_context(five, max_tokens=2000, min_score=None)) == 5
    assert (
        compress_context(
            [ContextChunk("a", "public", "t", 1.0, 400)], max_tokens=200, min_score=None
        )
        == []
    )


def test_scope_cache_misses_a_different_role() -> None:
    cache = ScopeCache(0.05)
    cache.store(LONG, access_scope("finance"), GEMINI_3_8_FLASH, VECTOR)
    assert cache.lookup(LONG, access_scope("customer"), GEMINI_3_8_FLASH, VECTOR) is None
    assert cache.lookup(LONG, access_scope("finance"), GEMINI_3_8_FLASH, VECTOR) == "exact"
    assert cache.lookup(PARTNER, access_scope("finance"), GEMINI_3_8_FLASH, VECTOR) == "semantic"
    assert cache.lookup(PARTNER, access_scope("finance"), GEMINI_3_8_FLASH, OTHER) is None


def test_cost_script_counts_assumed_tokens() -> None:
    report = simulate()
    assert report.n == 100
    assert report.exact_hits == 30
    assert report.semantic_hits == 5
    assert report.cheap_billed == 20
    assert report.generator_billed == 45
    assert report.billed_as_of == AS_OF
    one = cost_usd(GEMINI_3_8_FLASH, 200, 80, AS_OF)
    cheap = cost_usd(GEMINI_2_5_FLASH, 200, 80, AS_OF)
    assert report.cost_before == one * 100
    assert report.cost_after == one * 45 + cheap * 20
    text = render(report)
    assert "n=100" in text
    assert "token_source=assumed" in text
    assert "gemini_calls=0" in text
    assert "scope=public" in text
    assert f"cost_per_100_before={format_usd(report.cost_before)}" in text
    assert f"cost_per_100_after={format_usd(report.cost_after)}" in text


def test_exact_hit_skips_the_model_and_bills_zero(db_session, caplog) -> None:
    _seed(db_session)
    llm = _Counter("public answer", input_tokens=200, output_tokens=80)
    client = _client(db_session, llm)
    with caplog.at_level(logging.INFO, logger="supportdesk"):
        first = client.post("/ask", headers=_bearer("customer"), json={"query": LONG})
        second = client.post(
            "/ask",
            headers=_bearer("customer"),
            json={"query": "  " + LONG.upper() + "  "},
        )
    assert first.status_code == 200
    assert first.json()["cache_hit"] is None
    assert first.json()["model"] == GEMINI_3_8_FLASH
    assert first.json()["answer"] == "public answer"
    assert second.json()["cache_hit"] == "exact"
    assert second.json()["answer"] == "public answer"
    assert second.json()["steps"] == 0
    assert llm.calls == 1
    events = [event for event in _events(caplog) if event.get("route") == "ask"]
    assert events[0]["cost_usd"] == pytest.approx(
        float(cost_usd(GEMINI_3_8_FLASH, 200, 80, AS_OF)), abs=1e-8
    )
    assert events[0]["llm_input_tokens"] == 200
    assert events[0]["llm_output_tokens"] == 80
    assert events[1]["cache_hit"] == "exact"
    assert events[1]["cost_usd"] == 0
    assert events[1]["llm_input_tokens"] is None
    db_session.rollback()
    rows = db_session.scalars(select(RequestLog).order_by(RequestLog.id)).all()
    assert rows[0].model == GEMINI_3_8_FLASH
    assert rows[1].cache_hit == "exact"
    assert rows[1].cost_usd == 0


def test_customer_cannot_read_a_finance_cache_row(db_session) -> None:
    _seed(db_session)
    store_answer(
        db_session,
        query=LONG,
        scope=access_scope("finance"),
        model=GEMINI_3_8_FLASH,
        embedding=VECTOR,
        answer=FINANCE_ANSWER,
        steps=1,
    )
    customer_llm = _Counter("public answer")
    customer = _client(db_session, customer_llm)
    missed = customer.post("/ask", headers=_bearer("customer"), json={"query": LONG})
    assert missed.json()["answer"] == "public answer"
    assert FINANCE_ANSWER not in missed.text
    assert missed.json()["cache_hit"] is None
    assert customer_llm.calls == 1
    finance_llm = _Counter("should not run")
    finance = _client(db_session, finance_llm)
    hit = finance.post("/ask", headers=_bearer("finance"), json={"query": LONG})
    assert hit.json()["answer"] == FINANCE_ANSWER
    assert hit.json()["cache_hit"] == "exact"
    assert finance_llm.calls == 0


def test_exact_lookup_also_filters_the_scope_column(db_session) -> None:
    key = exact_cache_key(LONG, access_scope("customer"), GEMINI_3_8_FLASH)
    db_session.add(
        ResponseCache(
            cache_key=key,
            access_scope=access_scope("finance"),
            model=GEMINI_3_8_FLASH,
            query_text=normalize_query(LONG),
            embedding=VECTOR,
            response={"answer": FINANCE_ANSWER, "steps": 1},
        )
    )
    db_session.commit()
    assert lookup_exact(db_session, key, access_scope("customer"), GEMINI_3_8_FLASH) is None
    finance_key = exact_cache_key(LONG, access_scope("finance"), GEMINI_3_8_FLASH)
    assert lookup_exact(db_session, finance_key, access_scope("finance"), GEMINI_3_8_FLASH) is None
    assert lookup_exact(db_session, key, access_scope("finance"), GEMINI_3_8_FLASH) is not None


def test_semantic_lookup_stays_inside_one_scope(db_session) -> None:
    store_answer(
        db_session,
        query=LONG,
        scope=access_scope("finance"),
        model=GEMINI_3_8_FLASH,
        embedding=VECTOR,
        answer=FINANCE_ANSWER,
        steps=1,
    )
    assert (
        lookup_semantic(db_session, VECTOR, access_scope("finance"), GEMINI_3_8_FLASH, 0.05)
        is not None
    )
    assert (
        lookup_semantic(db_session, VECTOR, access_scope("customer"), GEMINI_3_8_FLASH, 0.05)
        is None
    )
    assert (
        lookup_semantic(db_session, OTHER, access_scope("finance"), GEMINI_3_8_FLASH, 0.05) is None
    )


def test_same_vector_reuses_the_answer_for_that_role(db_session) -> None:
    _seed(db_session)
    llm = _Counter("alpha")
    client = _client(db_session, llm)
    first = client.post("/ask", headers=_bearer("customer"), json={"query": LONG})
    second = client.post("/ask", headers=_bearer("customer"), json={"query": PARTNER})
    assert first.json()["cache_hit"] is None
    assert second.json()["cache_hit"] == "semantic"
    assert second.json()["answer"] == "alpha"
    assert llm.calls == 1


def test_over_cap_context_refuses_without_the_model(db_session, caplog) -> None:
    _seed(db_session, tokens=10)
    llm = _Counter("should not run")
    client = _client(db_session, llm, max_context_tokens=1)
    with caplog.at_level(logging.INFO, logger="supportdesk"):
        response = client.post("/ask", headers=_bearer("customer"), json={"query": LONG})
    assert response.status_code == 200
    assert response.json()["status"] == "refused"
    assert response.json()["answer"] == "I don't know"
    assert llm.calls == 0
    event = next(item for item in _events(caplog) if item.get("route") == "ask")
    assert event["retrieved_ids"] == ["public-doc#0"]
    assert event["status"] == "refused"


def test_ask_reports_the_routed_model(db_session) -> None:
    _seed(db_session)
    llm = _Counter("paged")
    client = _client(db_session, llm)
    short = client.post("/ask", headers=_bearer("customer"), json={"query": "short 000"})
    long = client.post("/ask", headers=_bearer("customer"), json={"query": LONG})
    assert short.json()["model"] == GEMINI_2_5_FLASH
    assert long.json()["model"] == GEMINI_3_8_FLASH
    assert short.json()["cache_hit"] is None
    assert long.json()["cache_hit"] is None
    assert llm.calls == 2


def _client(db_session, llm, **kwargs) -> TestClient:
    app = create_app(
        session_factory=make_session_factory(db_session.get_bind()),
        embedder=_Embedder(),
        reranker=_Reranker(),
        llm=llm,
        price_as_of=AS_OF,
        **kwargs,
    )
    return TestClient(app)


def _bearer(role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(role, get_settings().jwt_secret)}"}


def _seed(db_session, tokens: int = 4) -> None:
    db_session.add(
        Document(
            id="public-doc",
            title="Public page",
            source_path="public/public-doc.md",
            access_level="public",
            synthetic=True,
            content_hash="a" * 64,
        )
    )
    db_session.flush()
    db_session.add(
        Chunk(
            id="public-doc#0",
            doc_id="public-doc",
            ordinal=0,
            title="Public page",
            text="A public billing sentence.",
            heading_path="Billing",
            access_level="public",
            content_hash="d" * 64,
            token_count=tokens,
            embedding=VECTOR,
            embedding_model="test",
        )
    )
    db_session.commit()


def _events(caplog) -> list[dict]:
    events = []
    for record in caplog.records:
        if record.name != "supportdesk":
            continue
        try:
            events.append(json.loads(record.getMessage()))
        except json.JSONDecodeError:
            continue
    return events
