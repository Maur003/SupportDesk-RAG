"""HS256 tokens. The API trusts `role` and `sub`, not fields in the question."""

import time
from dataclasses import dataclass

import jwt

from supportdesk.access import ROLE_ACCESS

ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 3600


class TokenError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    role: str
    sub: str


def issue_token(
    role: str,
    secret: str,
    sub: str,
    *,
    ttl_seconds: int = TOKEN_TTL_SECONDS,
) -> str:
    if role not in ROLE_ACCESS:
        raise ValueError(f"unknown role: {role}")
    if not secret:
        raise ValueError("jwt secret is empty")
    if not isinstance(sub, str) or not sub.strip() or sub != sub.strip():
        raise ValueError("sub is empty")
    now = int(time.time())
    payload = {"role": role, "sub": sub, "iat": now, "exp": now + ttl_seconds}
    token = jwt.encode(payload, secret, algorithm=ALGORITHM)
    if isinstance(token, bytes):
        return token.decode("ascii")
    return token


def read_principal(token: str, secret: str) -> Principal:
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise TokenError("invalid token") from exc
    role = payload.get("role")
    sub = payload.get("sub")
    if not isinstance(role, str) or role not in ROLE_ACCESS:
        raise TokenError("invalid role")
    if not isinstance(sub, str) or not sub.strip() or sub != sub.strip():
        raise TokenError("invalid subject")
    return Principal(role=role, sub=sub)


def read_role(token: str, secret: str) -> str:
    return read_principal(token, secret).role
