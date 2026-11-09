"""Pending ticket writes. The model can request one. Only a later user request performs it."""

import uuid


class ConfirmationStore:
    def __init__(self) -> None:
        self._pending: dict[str, dict[str, str]] = {}

    def create(self, *, role: str, summary: str, account_id: str) -> str:
        confirmation_id = uuid.uuid4().hex
        self._pending[confirmation_id] = {
            "role": role,
            "summary": summary,
            "account_id": account_id,
        }
        return confirmation_id

    def get(self, confirmation_id: str, role: str) -> dict[str, str]:
        item = self._pending.get(confirmation_id)
        if item is None:
            raise KeyError(confirmation_id)
        if item["role"] != role:
            raise PermissionError(confirmation_id)
        return item

    def discard(self, confirmation_id: str) -> None:
        self._pending.pop(confirmation_id, None)
