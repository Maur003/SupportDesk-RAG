import base64
import json

import jwt
import pytest

from supportdesk.auth import TokenError, issue_token, read_role
from supportdesk.config import get_settings
from supportdesk.token import main

SECRET = "test-secret"


def test_round_trip_reads_the_role_and_subject() -> None:
    from supportdesk.auth import read_principal

    token = issue_token("finance", SECRET, "ACCT-4401")
    assert read_role(token, SECRET) == "finance"
    assert read_principal(token, SECRET).sub == "ACCT-4401"


def test_a_token_without_sub_is_rejected() -> None:
    from supportdesk.auth import read_principal

    forged = jwt.encode({"role": "customer", "exp": 9_999_999_999}, SECRET, algorithm="HS256")
    with pytest.raises(TokenError, match="subject"):
        read_principal(forged, SECRET)


def test_wrong_secret_expired_token_and_unknown_role_are_rejected() -> None:
    token = issue_token("customer", SECRET, "ACCT-4401")
    with pytest.raises(TokenError):
        read_role(token, "other-secret")
    expired = issue_token("customer", SECRET, "ACCT-4401", ttl_seconds=-10)
    with pytest.raises(TokenError):
        read_role(expired, SECRET)
    forged = jwt.encode(
        {"role": "root", "sub": "ACCT-4401", "exp": 9_999_999_999}, SECRET, algorithm="HS256"
    )
    with pytest.raises(TokenError):
        read_role(forged, SECRET)


def test_alg_none_token_is_rejected() -> None:
    token = _unsigned({"alg": "none", "typ": "JWT"}, {"role": "finance"})
    with pytest.raises(TokenError):
        read_role(token, SECRET)


def test_token_cli_prints_a_usable_token(capsys: pytest.CaptureFixture[str]) -> None:
    main(["agent", "ACCT-4401"])
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
