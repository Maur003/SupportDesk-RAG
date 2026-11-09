"""Mock CRM. A separate FastAPI app. The support API reaches it only over HTTP.

Tickets live in this process. The support service does not import them.
`TCK-1001` is seeded so a status lookup has a synthetic Northline ticket to read.
"""

import os
import secrets

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

SEED_TICKET_ID = "TCK-1001"


class TicketCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    account_id: str = Field(min_length=1)


def create_crm_app(api_key: str | None = None) -> FastAPI:
    key = api_key if api_key is not None else os.environ.get("CRM_API_KEY", "supportdesk-crm-dev")
    app = FastAPI(title="Northline mock CRM", version="0.1.0")
    app.state.api_key = key
    app.state.next_number = 1002
    app.state.tickets = {
        SEED_TICKET_ID: {
            "id": SEED_TICKET_ID,
            "status": "open",
            "summary": "Fiber drop at 18 North Mill Road",
            "account_id": "ACCT-4401",
        }
    }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tickets/{ticket_id}")
    def get_ticket(ticket_id: str, request: Request) -> dict[str, str]:
        _require_key(request, app.state.api_key)
        ticket = app.state.tickets.get(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="ticket not found")
        return ticket

    @app.post("/tickets")
    def create_ticket(body: TicketCreate, request: Request) -> dict[str, str]:
        _require_key(request, app.state.api_key)
        ticket_id = f"TCK-{app.state.next_number}"
        app.state.next_number += 1
        ticket = {
            "id": ticket_id,
            "status": "open",
            "summary": body.summary,
            "account_id": body.account_id,
        }
        app.state.tickets[ticket_id] = ticket
        return ticket

    return app


def _require_key(request: Request, expected: str) -> None:
    got = request.headers.get("x-api-key", "")
    if not _keys_match(got, expected):
        raise HTTPException(status_code=401, detail="invalid api key")


def _keys_match(got: str, expected: str) -> bool:
    got_b = got.encode()
    expected_b = expected.encode()
    if len(got_b) != len(expected_b):
        return False
    return secrets.compare_digest(got_b, expected_b)


app = create_crm_app()
