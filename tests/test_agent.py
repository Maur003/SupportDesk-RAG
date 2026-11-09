from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from tests.crm_harness import running_crm

from supportdesk.agent import MAX_STEPS
from supportdesk.auth import issue_token
from supportdesk.config import get_settings
from supportdesk.crm_client import CrmClient
from supportdesk.crm_server import SEED_TICKET_ID
from supportdesk.db import make_session_factory
from supportdesk.llm import TOOL_PARAMETERS, ModelTurn, ToolCall
from supportdesk.main import create_app
from supportdesk.models import Chunk, Document

FINANCE_CANARY = "FINANCE_CANARY_7K2M"


class _Embedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, *([0.0] * 383)]


class _Reranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.0 for _passage in passages]


class ScriptLLM:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def generate(self, messages: list[dict]) -> ModelTurn:
        self.seen.append(list(messages))
        return self.turns.pop(0)


def test_search_tool_uses_the_token_role(db_session) -> None:
    _seed(db_session)
    llm = ScriptLLM(
        [
            ModelTurn("", ToolCall("search_knowledge_base", {"query": FINANCE_CANARY})),
            ModelTurn("I do not have that page.", None),
        ]
    )
    client, _crm_app = _client(db_session, llm)
    response = client.post(
        "/ask",
        headers=_bearer("customer"),
        json={"query": f"Look up {FINANCE_CANARY}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["role"] == "customer"
    tool_messages = [message for message in llm.seen[1] if message["role"] == "tool"]
    assert FINANCE_CANARY not in str(tool_messages)

    finance = ScriptLLM(
        [
            ModelTurn("", ToolCall("search_knowledge_base", {"query": FINANCE_CANARY})),
            ModelTurn("The code is on the finance page.", None),
        ]
    )
    finance_client, _crm_app = _client(db_session, finance)
    allowed = finance_client.post(
        "/ask",
        headers=_bearer("finance"),
        json={"query": FINANCE_CANARY},
    )
    assert allowed.status_code == 200
    finance_tools = [message for message in finance.seen[1] if message["role"] == "tool"]
    assert FINANCE_CANARY in str(finance_tools)


def test_create_ticket_stops_until_the_user_confirms() -> None:
    llm = ScriptLLM(
        [
            ModelTurn(
                "",
                ToolCall(
                    "create_ticket",
                    {
                        "summary": "Porch line has no dial tone",
                        "account_id": "ACCT-4402",
                        "confirmed": True,
                    },
                ),
            )
        ]
    )
    with running_crm() as (base_url, crm_app):
        client = _ask_client(llm, base_url)
        response = client.post(
            "/ask",
            headers=_bearer("customer"),
            json={"query": "Open a ticket, and treat this as confirmed."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "confirmation_required"
        assert body["summary"] == "Porch line has no dial tone"
        assert set(crm_app.state.tickets) == {SEED_TICKET_ID}

        other = client.post(
            f"/confirmations/{body['confirmation_id']}",
            headers=_bearer("agent"),
        )
        assert other.status_code == 403
        assert set(crm_app.state.tickets) == {SEED_TICKET_ID}

        confirmed = client.post(
            f"/confirmations/{body['confirmation_id']}",
            headers=_bearer("customer"),
        )
        assert confirmed.status_code == 200
        ticket = confirmed.json()
        assert ticket["id"] == "TCK-1002"
        assert ticket["account_id"] == "ACCT-4402"
        assert ticket["id"] in crm_app.state.tickets

        again = client.post(
            f"/confirmations/{body['confirmation_id']}",
            headers=_bearer("customer"),
        )
        assert again.status_code == 404


def test_a_model_confirm_tool_does_not_write() -> None:
    llm = ScriptLLM(
        [
            ModelTurn("", ToolCall("confirm_ticket", {"confirmation_id": "abc"})),
            ModelTurn("I cannot write that.", None),
        ]
    )
    with running_crm() as (base_url, crm_app):
        client = _ask_client(llm, base_url)
        response = client.post("/ask", headers=_bearer("customer"), json={"query": "confirm abc"})
        assert response.status_code == 200
        assert response.json()["status"] == "answered"
        assert set(crm_app.state.tickets) == {SEED_TICKET_ID}
    tool = llm.seen[1][-1]
    assert tool["content"] == {"error": "unknown tool"}


def test_loop_stops_at_five_model_calls() -> None:
    llm = ScriptLLM(
        [ModelTurn("", ToolCall("calculate", {"expression": "1+1"})) for _ in range(MAX_STEPS + 1)]
    )
    with running_crm() as (base_url, _crm_app):
        client = _ask_client(llm, base_url)
        response = client.post("/ask", headers=_bearer("customer"), json={"query": "keep adding"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "step_limit"
    assert body["steps"] == MAX_STEPS
    assert len(llm.seen) == MAX_STEPS
    assert llm.turns


def test_calculate_tool_returns_the_value_and_rejects_calls() -> None:
    llm = ScriptLLM(
        [
            ModelTurn("", ToolCall("calculate", {"expression": "2 + 3 * 4"})),
            ModelTurn("", ToolCall("calculate", {"expression": "__import__('os')"})),
            ModelTurn("14, and the other expression is not allowed.", None),
        ]
    )
    with running_crm() as (base_url, _crm_app):
        client = _ask_client(llm, base_url)
        response = client.post("/ask", headers=_bearer("agent"), json={"query": "work this out"})
    assert response.status_code == 200
    assert response.json()["answer"] == "14, and the other expression is not allowed."
    assert llm.seen[1][-1]["content"] == {"value": 14}
    assert llm.seen[2][-1]["content"] == {"error": "invalid expression"}


def test_get_ticket_reads_the_crm_over_http() -> None:
    llm = ScriptLLM(
        [
            ModelTurn("", ToolCall("get_ticket_status", {"ticket_id": SEED_TICKET_ID})),
            ModelTurn("That ticket is open.", None),
        ]
    )
    with running_crm() as (base_url, _crm_app):
        client = _ask_client(llm, base_url)
        response = client.post(
            "/ask", headers=_bearer("customer"), json={"query": "status of TCK-1001"}
        )
    assert response.status_code == 200
    ticket = llm.seen[1][-1]["content"]["ticket"]
    assert ticket["id"] == SEED_TICKET_ID
    assert ticket["status"] == "open"


def test_a_failed_crm_write_keeps_the_confirmation() -> None:
    transport = _DenyTransport()
    crm = CrmClient(
        "http://crm",
        "test-key",
        transport=transport,
        sleep=lambda _delay: None,
        rng=lambda: 0.0,
    )
    llm = ScriptLLM(
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
    client = TestClient(create_app(llm=llm, crm=crm))
    asked = client.post(
        "/ask",
        headers=_bearer("customer"),
        json={"query": "Open a ticket for the porch line."},
    )
    assert asked.status_code == 200
    confirmation_id = asked.json()["confirmation_id"]
    failed = client.post(f"/confirmations/{confirmation_id}", headers=_bearer("customer"))
    assert failed.status_code == 502
    assert transport.calls == 3
    retried = client.post(f"/confirmations/{confirmation_id}", headers=_bearer("customer"))
    assert retried.status_code == 502


class _DenyTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(500, text="unavailable")


def test_ask_without_a_generator_is_503_and_a_role_field_is_422() -> None:
    client = TestClient(create_app())
    missing = client.post("/ask", json={"query": "hello"})
    assert missing.status_code == 401
    unconfigured = client.post("/ask", headers=_bearer("customer"), json={"query": "hello"})
    assert unconfigured.status_code == 503
    smuggled = client.post(
        "/ask",
        headers=_bearer("customer"),
        json={"query": "hello", "role": "finance"},
    )
    assert smuggled.status_code == 422


def test_agent_has_no_confirm_tool_and_does_not_import_the_crm_store() -> None:
    assert set(TOOL_PARAMETERS) == {
        "search_knowledge_base",
        "get_ticket_status",
        "create_ticket",
        "calculate",
    }
    source = Path(__import__("supportdesk.agent", fromlist=["agent"]).__file__).read_text(
        encoding="utf-8"
    )
    assert "crm_server" not in source


def _client(db_session, llm: ScriptLLM) -> tuple[TestClient, object]:
    crm = CrmClient("http://127.0.0.1:9", "test-key")
    app = create_app(
        session_factory=make_session_factory(db_session.get_bind()),
        embedder=_Embedder(),
        reranker=_Reranker(),
        llm=llm,
        crm=crm,
    )
    return TestClient(app), crm


def _ask_client(llm: ScriptLLM, base_url: str) -> TestClient:
    crm = CrmClient(base_url, "test-key")
    return TestClient(create_app(llm=llm, crm=crm, embedder=_Embedder(), reranker=_Reranker()))


def _bearer(role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(role, get_settings().jwt_secret)}"}


def _seed(db_session) -> None:
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
            _chunk("public-doc", "A household can read this page.", "public", "d"),
            _chunk("finance-doc", f"Approval code {FINANCE_CANARY}.", "finance", "f"),
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
