"""Pending ticket writes. The model can request one. Only a later user request performs it.

A confirmation is bound to the token `sub` and expires after five minutes.
"""

import time
import uuid
from collections.abc import Callable

CONFIRMATION_TTL_SECONDS = 300


class ConfirmationStore:
    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        ttl_seconds: int = CONFIRMATION_TTL_SECONDS,
    ) -> None:
        self._clock = clock or time.time
        self._ttl_seconds = ttl_seconds
        self._pending: dict[str, dict[str, str | float]] = {}

    def create(self, *, role: str, sub: str, summary: str, account_id: str) -> str:
        confirmation_id = uuid.uuid4().hex
        self._pending[confirmation_id] = {
            "role": role,
            "sub": sub,
            "summary": summary,
            "account_id": account_id,
            "created_at": self._clock(),
        }
        return confirmation_id

    def get(self, confirmation_id: str, role: str, sub: str) -> dict[str, str]:
        item = self._pending.get(confirmation_id)
        if item is None or self._expired(item):
            self._pending.pop(confirmation_id, None)
            raise KeyError(confirmation_id)
        if item["role"] != role or item["sub"] != sub:
            raise PermissionError(confirmation_id)
        return {
            "role": str(item["role"]),
            "sub": str(item["sub"]),
            "summary": str(item["summary"]),
            "account_id": str(item["account_id"]),
        }

    def discard(self, confirmation_id: str) -> None:
        self._pending.pop(confirmation_id, None)

    def _expired(self, item: dict[str, str | float]) -> bool:
        created = item["created_at"]
        if not isinstance(created, float):
            return True
        return self._clock() - created >= self._ttl_seconds
