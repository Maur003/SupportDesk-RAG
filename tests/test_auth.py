import base64
import json

import jwt
import pytest

from supportdesk.auth import TokenError, issue_token, read_role
from supportdesk.config import get_settings
from supportdesk.token import main

SECRET = "test-secret"


def test_round_trip_reads_the_role_claim() -> None:
    token = issue_token("finance", SECRET)
    assert read_role(token, SECRET) == "finance"


def test_wrong_secret_expired_token_and_unknown_role_are_rejected() -> None:
    token = issue_token("customer", SECRET)
    with pytest.raises(TokenError):
        read_role(token, "other-secret")
    expired = issue_token("customer", SECRET, ttl_seconds=-10)
    with pytest.raises(TokenError):
        read_role(expired, SECRET)
    forged = jwt.encode({"role": "root", "exp": 9_999_999_999}, SECRET, algorithm="HS256")
    with pytest.raises(TokenError):
        read_role(forged, SECRET)


def test_alg_none_token_is_rejected() -> None:
    token = _unsigned({"alg": "none", "typ": "JWT"}, {"role": "finance"})
    with pytest.raises(TokenError):
        read_role(token, SECRET)


def test_token_cli_prints_a_usable_token(capsys: pytest.CaptureFixture[str]) -> None:
    main(["agent"])
    token = capsys.readouterr().out.strip()
    assert read_role(token, get_settings().jwt_secret) == "agent"


def test_token_cli_rejects_an_unknown_role() -> None:
    with pytest.raises(SystemExit):
        main(["root"])


def _unsigned(header: dict, payload: dict) -> str:
    return ".".join((_b64(header), _b64(payload), ""))


def _b64(value: dict) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
