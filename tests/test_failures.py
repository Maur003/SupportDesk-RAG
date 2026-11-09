import httpx
import pytest
from fastapi.testclient import TestClient

from supportdesk.agent import REFUSAL, UNAVAILABLE, run_agent
from supportdesk.auth import issue_token
from supportdesk.breaker import CircuitBreaker
from supportdesk.config import get_settings
from supportdesk.confirmations import ConfirmationStore
from supportdesk.crm_client import CrmClient, CrmError
from supportdesk.db import make_session_factory
from supportdesk.embedder import EmbedderError
from supportdesk.llm import GuardedLLM, LLMUnavailable, ModelTurn, ToolCall
from supportdesk.main import create_app
from supportdesk.models import Chunk, Document

FINANCE_CANARY = "FINANCE_CANARY_7K2M"


class _Vector:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, *([0.0] * 383)]


class _Down:
    def embed_query(self, text: str) -> list[float]:
        raise EmbedderError("embedder unavailable")


class _Reranker:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls += 1
        return [0.0 for _passage in passages]


class _Script:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def generate(self, messages: list[dict]) -> ModelTurn:
        self.seen.append(list(messages))
        return self.turns.pop(0)


class _Boom:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[dict]) -> ModelTurn:
        self.calls += 1
        raise LLMUnavailable("down")


class _Status(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(str(status_code))


class _RetryThenOk:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[dict]) -> ModelTurn:
        self.calls += 1
        if self.calls < 3:
            raise _Status(503)
        return ModelTurn("ok", None)


class _Always:
    def __init__(self, exc: Exception) -> None:
        self.calls = 0
        self.exc = exc

    def generate(self, messages: list[dict]) -> ModelTurn:
        self.calls += 1
        raise self.exc


class _Statuses(httpx.BaseTransport):
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = list(statuses)
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(self.statuses.pop(0), text="no")


def test_empty_retrieval_does_not_call_the_model(db_session) -> None:
    llm = _Script([ModelTurn("should not run", None)])
    client = _ask(db_session, llm, _Vector(), _Reranker())
    response = client.post("/ask", headers=_bearer("customer"), json={"query": "student fax mesh"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert body["answer"] == REFUSAL
    assert body["steps"] == 0
    assert llm.seen == []


def test_embedder_down_uses_keyword_search_and_keeps_the_role_filter(db_session) -> None:
    _seed(db_session, "The millrace plan includes fiber.", f"Finance only {FINANCE_CANARY}.")
    reranker = _Reranker()
    llm = _Script(
        [
            ModelTurn("", ToolCall("search_knowledge_base", {"query": "millrace"})),
            ModelTurn("The millrace plan includes fiber.", None),
        ]
    )
    client = _ask(db_session, llm, _Down(), reranker)
    search = client.post("/search", headers=_bearer("customer"), json={"query": "millrace"})
    assert search.status_code == 200
    blob = " ".join(hit["text"] for hit in search.json()["hits"])
    assert "millrace" in blob
    assert FINANCE_CANARY not in blob
    asked = client.post("/ask", headers=_bearer("customer"), json={"query": "millrace"})
    assert asked.status_code == 200
    assert asked.json()["status"] == "answered"
    tool = llm.seen[1][-1]["content"]
    assert "millrace" in str(tool)
    assert FINANCE_CANARY not in str(tool)
    assert reranker.calls == 0


def test_embedder_down_and_no_keyword_hit_refuses_without_the_model(db_session) -> None:
    _seed(db_session, "A household can read this billing page.", "Finance ledger.")
    reranker = _Reranker()
    llm = _Script([ModelTurn("should not run", None)])
    client = _ask(db_session, llm, _Down(), reranker)
    response = client.post("/ask", headers=_bearer("customer"), json={"query": "zzznomatch"})
    assert response.status_code == 200
    assert response.json()["status"] == "refused"
    assert response.json()["answer"] == REFUSAL
    assert llm.seen == []
    assert reranker.calls == 0


def test_empty_tool_search_stops_without_another_model_call() -> None:
    llm = _Script(
        [
            ModelTurn("", ToolCall("search_knowledge_base", {"query": "nothing"})),
            ModelTurn("should not run", None),
        ]
    )
    result = run_agent(
        "nothing",
        "customer",
        llm=llm,
        search_fn=lambda _query: [],
        crm=CrmClient("http://127.0.0.1:9", "test-key"),
        confirmations=ConfirmationStore(),
    )
    assert result.status == "refused"
    assert result.answer == REFUSAL
    assert len(llm.seen) == 1
    assert llm.turns


def test_model_down_returns_citations_without_an_answer(db_session) -> None:
    _seed(db_session, "A household can read this page.", "Finance ledger.")
    llm = _Boom()
    client = _ask(db_session, llm, _Vector(), _Reranker())
    response = client.post("/ask", headers=_bearer("customer"), json={"query": "household page"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["answer"] == UNAVAILABLE
    assert [hit["id"] for hit in body["citations"]] == ["public-doc#0"]
    assert llm.calls == 1


def test_open_model_circuit_skips_the_next_call_until_the_reset(db_session) -> None:
    _seed(db_session, "A household can read this page.", "Finance ledger.")
    now = [0.0]
    breaker = CircuitBreaker(2, 10, clock=lambda: now[0])
    llm = _Boom()
    client = _ask(db_session, llm, _Vector(), _Reranker(), llm_breaker=breaker)
    for _ in range(2):
        failed = client.post("/ask", headers=_bearer("customer"), json={"query": "household page"})
        assert failed.json()["status"] == "unavailable"
    blocked = client.post("/ask", headers=_bearer("customer"), json={"query": "household page"})
    assert blocked.json()["status"] == "unavailable"
    assert blocked.json()["citations"][0]["id"] == "public-doc#0"
    assert llm.calls == 2
    now[0] = 10
    llm.generate = lambda messages: ModelTurn("back", None)  # type: ignore[method-assign]
    recovered = client.post("/ask", headers=_bearer("customer"), json={"query": "household page"})
    assert recovered.json()["status"] == "answered"
    assert recovered.json()["answer"] == "back"


def test_generator_retries_only_429_and_5xx() -> None:
    inner = _RetryThenOk()
    slept: list[float] = []
    guarded = GuardedLLM(
        inner,
        CircuitBreaker(3, 30),
        attempts=3,
        backoff=0.2,
        sleep=slept.append,
        rng=lambda: 0.0,
    )
    turn = guarded.generate([{"role": "user", "text": "hi"}])
    assert turn.text == "ok"
    assert inner.calls == 3
    assert slept == [0.1, 0.2]

    timed = _Always(TimeoutError("timed out"))
    slept.clear()
    guarded = GuardedLLM(
        timed,
        CircuitBreaker(3, 30),
        attempts=3,
        backoff=0.2,
        sleep=slept.append,
        rng=lambda: 0.0,
    )
    with pytest.raises(LLMUnavailable):
        guarded.generate([{"role": "user", "text": "hi"}])
    assert timed.calls == 1
    assert slept == []

    rejected = _Always(_Status(400))
    with pytest.raises(_Status):
        GuardedLLM(rejected, CircuitBreaker(3, 30), attempts=3, sleep=slept.append).generate(
            [{"role": "user", "text": "hi"}]
        )
    assert rejected.calls == 1


def test_crm_circuit_opens_on_5xx_and_timeout_and_ignores_client_errors() -> None:
    transport = _Statuses([500, 500, 200])
    breaker = CircuitBreaker(2, 30, clock=lambda: 0.0)
    client = CrmClient(
        "http://crm",
        "test-key",
        transport=transport,
        attempts=1,
        breaker=breaker,
        sleep=lambda _delay: None,
    )
    with pytest.raises(CrmError):
        client.get_ticket("TCK-1001")
    with pytest.raises(CrmError):
        client.get_ticket("TCK-1001")
    with pytest.raises(CrmError, match="circuit open"):
        client.get_ticket("TCK-1001")
    assert transport.calls == 2
    client.close()

    denied = _Statuses([400, 400])
    client = CrmClient(
        "http://crm",
        "test-key",
        transport=denied,
        attempts=1,
        breaker=CircuitBreaker(1, 30),
        sleep=lambda _delay: None,
    )
    with pytest.raises(CrmError) as caught:
        client.get_ticket("TCK-1001")
    assert caught.value.status_code == 400
    with pytest.raises(CrmError) as caught:
        client.get_ticket("TCK-1001")
    assert caught.value.status_code == 400
    assert denied.calls == 2
    client.close()


def test_confirm_fails_fast_when_the_crm_circuit_is_open() -> None:
    transport = _Statuses([500, 500])
    breaker = CircuitBreaker(1, 30, clock=lambda: 0.0)
    crm = CrmClient(
        "http://crm",
        "test-key",
        transport=transport,
        attempts=1,
        breaker=breaker,
        sleep=lambda _delay: None,
    )
    llm = _Script(
        [
            ModelTurn(
                "",
                ToolCall(
                    "create_ticket",
                    {"summary": "Porch line has no dial tone", "account_id": "ACCT-4402"},
                ),
            )
        ]
    )
    client = TestClient(create_app(llm=llm, crm=crm, crm_breaker=breaker, prefetch_retrieval=False))
    asked = client.post(
        "/ask",
        headers=_bearer("customer"),
        json={"query": "Open a ticket for the porch line."},
    )
    confirmation_id = asked.json()["confirmation_id"]
    failed = client.post(f"/confirmations/{confirmation_id}", headers=_bearer("customer"))
    assert failed.status_code == 502
    blocked = client.post(f"/confirmations/{confirmation_id}", headers=_bearer("customer"))
    assert blocked.status_code == 503
    assert transport.calls == 1


def _ask(db_session, llm, embedder, reranker, llm_breaker=None) -> TestClient:
    app = create_app(
        session_factory=make_session_factory(db_session.get_bind()),
        embedder=embedder,
        reranker=reranker,
        llm=llm,
        llm_breaker=llm_breaker,
        crm=CrmClient("http://127.0.0.1:9", "test-key"),
    )
    return TestClient(app)


def _bearer(role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(role, get_settings().jwt_secret)}"}


def _seed(db_session, public_text: str, finance_text: str) -> None:
    db_session.add_all(
        [
            Document(
                id="public-doc",
                title="Public page",
                source_path="public/public-doc.md",
                access_level="public",
                synthetic=True,
                content_hash="a" * 64,
            ),
            Document(
                id="finance-doc",
                title="Finance page",
                source_path="finance/finance-doc.md",
                access_level="finance",
                synthetic=True,
                content_hash="c" * 64,
            ),
        ]
    )
    db_session.flush()
    db_session.add_all(
        [
            _chunk("public-doc", public_text, "public", "d"),
            _chunk("finance-doc", finance_text, "finance", "f"),
        ]
    )
    db_session.commit()


def _chunk(doc_id: str, text: str, access_level: str, salt: str) -> Chunk:
    return Chunk(
        id=f"{doc_id}#0",
        doc_id=doc_id,
        ordinal=0,
        title=doc_id,
        text=text,
        heading_path=doc_id,
        access_level=access_level,
        content_hash=salt * 64,
        token_count=4,
        embedding=[1.0, *([0.0] * 383)],
        embedding_model="test",
    )
