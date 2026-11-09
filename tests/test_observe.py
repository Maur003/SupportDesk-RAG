import json
import logging

from fastapi.testclient import TestClient
from sqlalchemy import select

from supportdesk.auth import issue_token
from supportdesk.config import get_settings
from supportdesk.crm_client import CrmClient
from supportdesk.db import make_session_factory
from supportdesk.llm import ModelTurn
from supportdesk.main import create_app
from supportdesk.models import Chunk, Document, RequestLog

PASSAGE = "VISIBLE_PASSAGE_9K is the household outage sentence."


class _Embedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, *([0.0] * 383)]


class _Reranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.5 for _passage in passages]


class _Answer:
    def generate(self, messages: list[dict]) -> ModelTurn:
        return ModelTurn("The page is public.", input_tokens=11, output_tokens=7)


def test_search_logs_ids_and_scores_without_chunk_text(db_session, caplog) -> None:
    _seed(db_session)
    client = _client(db_session)
    with caplog.at_level(logging.INFO, logger="supportdesk"):
        response = client.post(
            "/search",
            headers={**_bearer("customer"), "X-Trace-Id": "trace-fixed"},
            json={"query": "household outage"},
        )
    assert response.status_code == 200
    assert response.headers["x-trace-id"] == "trace-fixed"
    assert PASSAGE in response.json()["hits"][0]["text"]
    logged = _events(caplog)
    event = next(item for item in logged if item.get("trace_id") == "trace-fixed")
    assert event["route"] == "search"
    assert event["role"] == "customer"
    assert event["status"] == "ok"
    assert event["retrieved_ids"] == ["public-doc#0"]
    assert event["scores"] == [0.5]
    assert event["retrieved_tokens"] == 4
    assert event["llm_input_tokens"] is None
    assert event["llm_output_tokens"] is None
    assert event["embed_ms"] >= 0
    assert event["search_ms"] >= 0
    assert event["rerank_ms"] >= 0
    assert PASSAGE not in json.dumps(event)
    row = db_session.scalars(select(RequestLog).where(RequestLog.trace_id == "trace-fixed")).one()
    assert row.retrieved_ids == ["public-doc#0"]
    assert row.scores == [0.5]
    assert row.retrieved_tokens == 4
    assert PASSAGE not in json.dumps(
        {
            "ids": row.retrieved_ids,
            "scores": row.scores,
            "status": row.status,
            "error_class": row.error_class,
        }
    )


def test_ask_records_model_token_counts_from_the_turn(db_session, caplog) -> None:
    _seed(db_session)
    client = _client(db_session, llm=_Answer())
    with caplog.at_level(logging.INFO, logger="supportdesk"):
        response = client.post(
            "/ask",
            headers=_bearer("customer"),
            json={"query": "household outage"},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    event = next(item for item in _events(caplog) if item.get("route") == "ask")
    assert event["status"] == "answered"
    assert event["llm_input_tokens"] == 11
    assert event["llm_output_tokens"] == 7
    assert event["retrieved_tokens"] == 4
    assert event["retrieved_ids"] == ["public-doc#0"]
    assert PASSAGE not in json.dumps(event)


def test_empty_retrieval_is_logged_without_a_model_call(db_session, caplog) -> None:
    called = {"n": 0}

    class _LLM:
        def generate(self, messages: list[dict]) -> ModelTurn:
            called["n"] += 1
            return ModelTurn("no")

    client = _client(db_session, llm=_LLM())
    with caplog.at_level(logging.INFO, logger="supportdesk"):
        response = client.post("/ask", headers=_bearer("customer"), json={"query": "student fax"})
    assert response.status_code == 200
    assert response.json()["status"] == "refused"
    assert called["n"] == 0
    event = next(item for item in _events(caplog) if item.get("route") == "ask")
    assert event["status"] == "refused"
    assert event["retrieved_ids"] == []
    assert event["retrieved_tokens"] == 0
    assert event["llm_ms"] == 0
    assert event["llm_input_tokens"] is None


def test_missing_token_counts_as_an_http_error(caplog) -> None:
    client = TestClient(create_app())
    with caplog.at_level(logging.INFO, logger="supportdesk"):
        response = client.post("/search", json={"query": "household"})
    assert response.status_code == 401
    event = next(item for item in _events(caplog) if item.get("route") == "search")
    assert event["status"] == "401"
    assert event["error_class"] == "http_401"


def test_metrics_count_the_logged_request(db_session) -> None:
    _seed(db_session)
    client = _client(db_session)
    response = client.post(
        "/search",
        headers=_bearer("customer"),
        json={"query": "household outage"},
    )
    assert response.status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    assert 'supportdesk_requests_total{route="search",status="ok"}' in body
    assert 'supportdesk_stage_seconds_count{stage="embed"}' in body
    assert 'supportdesk_stage_seconds_count{stage="rerank"}' in body
    assert 'supportdesk_stage_seconds_count{stage="llm"}' in body


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


def _client(db_session, llm=None) -> TestClient:
    app = create_app(
        session_factory=make_session_factory(db_session.get_bind()),
        embedder=_Embedder(),
        reranker=_Reranker(),
        llm=llm,
        crm=CrmClient("http://127.0.0.1:9", "test-key"),
    )
    return TestClient(app)


def _bearer(role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(role, get_settings().jwt_secret, 'ACCT-4401')}"}


def _seed(db_session) -> None:
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
            text=PASSAGE,
            heading_path="Outage",
            access_level="public",
            content_hash="d" * 64,
            token_count=4,
            embedding=[1.0, *([0.0] * 383)],
            embedding_model="test",
        )
    )
    db_session.commit()
