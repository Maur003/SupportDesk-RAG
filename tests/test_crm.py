import httpx
import pytest
from tests.crm_harness import running_crm

from supportdesk.crm_client import CrmClient, CrmError
from supportdesk.crm_server import SEED_TICKET_ID


class _ScriptTransport(httpx.BaseTransport):
    def __init__(self, statuses: list[int], *, raises: Exception | None = None) -> None:
        self.statuses = list(statuses)
        self.raises = raises
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.raises is not None and self.calls == 1:
            raise self.raises
        status = self.statuses.pop(0)
        return httpx.Response(
            status,
            json={
                "id": SEED_TICKET_ID,
                "status": "open",
                "summary": "Fiber drop",
                "account_id": "ACCT-4401",
            },
        )


def test_client_reads_and_creates_against_the_crm_app() -> None:
    with running_crm() as (base_url, crm_app):
        client = CrmClient(base_url, "test-key")
        ticket = client.get_ticket(SEED_TICKET_ID)
        assert ticket["summary"] == "Fiber drop at 18 North Mill Road"
        created = client.create_ticket("No dial tone on the porch line", "ACCT-4402")
        assert created["id"] == "TCK-1002"
        assert created["id"] in crm_app.state.tickets
        client.close()


def test_wrong_api_key_is_401_and_is_not_retried() -> None:
    with running_crm() as (base_url, _crm_app):
        client = CrmClient(base_url, "other-key")
        with pytest.raises(CrmError) as caught:
            client.get_ticket(SEED_TICKET_ID)
        assert caught.value.status_code == 401
        client.close()


def test_retries_only_429_and_5xx() -> None:
    slept: list[float] = []
    transport = _ScriptTransport([500, 429, 200])
    client = CrmClient(
        "http://crm",
        "test-key",
        transport=transport,
        attempts=3,
        backoff=0.2,
        sleep=slept.append,
        rng=lambda: 0.0,
    )
    ticket = client.get_ticket(SEED_TICKET_ID)
    assert ticket["id"] == SEED_TICKET_ID
    assert transport.calls == 3
    assert slept == [0.1, 0.2]
    client.close()

    slept.clear()
    denied = _ScriptTransport([400])
    client = CrmClient(
        "http://crm",
        "test-key",
        transport=denied,
        sleep=slept.append,
        rng=lambda: 0.0,
    )
    with pytest.raises(CrmError) as caught:
        client.get_ticket(SEED_TICKET_ID)
    assert caught.value.status_code == 400
    assert denied.calls == 1
    assert slept == []
    client.close()


def test_timeout_is_not_retried() -> None:
    slept: list[float] = []
    transport = _ScriptTransport([], raises=httpx.ReadTimeout("timed out"))
    client = CrmClient(
        "http://crm",
        "test-key",
        transport=transport,
        sleep=slept.append,
        rng=lambda: 0.0,
    )
    with pytest.raises(CrmError) as caught:
        client.get_ticket(SEED_TICKET_ID)
    assert caught.value.status_code is None
    assert transport.calls == 1
    assert slept == []
    client.close()
