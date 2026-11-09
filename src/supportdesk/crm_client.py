"""HTTP client for the mock CRM.

Retries only HTTP 429 and 5xx. Other 4xx responses and timeouts are returned
to the caller on the first attempt. The delay is
`backoff * 2^attempt * uniform(0.5, 1.0)`.
"""

import random
import time
from collections.abc import Callable

import httpx

from supportdesk.breaker import RETRYABLE_STATUS, CircuitBreaker, retry_delay


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
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._attempts = attempts
        self._backoff = backoff
        self._sleep = sleep
        self._rng = rng
        self._breaker = breaker
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
        if self._breaker is not None and not self._breaker.allow():
            raise CrmError(None, "circuit open")
        last_status: int | None = None
        last_body = ""
        for attempt in range(self._attempts):
            try:
                response = self._client.request(method, path, json=json)
            except httpx.TimeoutException as exc:
                self._fail()
                raise CrmError(None, "timeout") from exc
            if response.status_code in RETRYABLE_STATUS and attempt + 1 < self._attempts:
                last_status = response.status_code
                last_body = response.text
                self._sleep(retry_delay(self._backoff, attempt, self._rng))
                continue
            if response.status_code in RETRYABLE_STATUS:
                self._fail()
                raise CrmError(response.status_code, response.text)
            if response.status_code >= 400:
                self._end_trial()
                raise CrmError(response.status_code, response.text)
            self._succeed()
            return response
        self._fail()
        raise CrmError(last_status, last_body)

    def _succeed(self) -> None:
        if self._breaker is not None:
            self._breaker.record_success()

    def _fail(self) -> None:
        if self._breaker is not None:
            self._breaker.record_failure()

    def _end_trial(self) -> None:
        if self._breaker is not None:
            self._breaker.end_trial()
