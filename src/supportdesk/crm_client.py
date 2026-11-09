"""HTTP client for the mock CRM.

Retries only HTTP 429 and 5xx. Other 4xx responses and timeouts are returned
to the caller on the first attempt. The delay is
`backoff * 2^attempt * uniform(0.5, 1.0)`.
"""

import random
import time
from collections.abc import Callable

import httpx

RETRYABLE = frozenset({429, 500, 502, 503, 504})


class CrmError(Exception):
    def __init__(self, status_code: int | None, detail: str) -> None:
        self.status_code = status_code
        super().__init__(detail)


class CrmClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 2.0,
        attempts: int = 3,
        backoff: float = 0.2,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._attempts = attempts
        self._backoff = backoff
        self._sleep = sleep
        self._rng = rng
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"X-API-Key": api_key},
        )

    def close(self) -> None:
        self._client.close()

    def get_ticket(self, ticket_id: str) -> dict:
        return self._send("GET", f"/tickets/{ticket_id}").json()

    def create_ticket(self, summary: str, account_id: str) -> dict:
        body = {"summary": summary, "account_id": account_id}
        return self._send("POST", "/tickets", json=body).json()

    def _send(self, method: str, path: str, json: dict | None = None) -> httpx.Response:
        last_status: int | None = None
        last_body = ""
        for attempt in range(self._attempts):
            try:
                response = self._client.request(method, path, json=json)
            except httpx.TimeoutException as exc:
                raise CrmError(None, "timeout") from exc
            if response.status_code in RETRYABLE and attempt + 1 < self._attempts:
                last_status = response.status_code
                last_body = response.text
                self._sleep(self._backoff * (2**attempt) * (0.5 + 0.5 * self._rng()))
                continue
            if response.status_code >= 400:
                raise CrmError(response.status_code, response.text)
            return response
        raise CrmError(last_status, last_body)
