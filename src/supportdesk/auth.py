"""HS256 role tokens. The role claim is the only access input the API trusts."""

import time

import jwt

from supportdesk.access import ROLE_ACCESS

ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 3600


class TokenError(Exception):
    pass


def issue_token(role: str, secret: str, *, ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    if role not in ROLE_ACCESS:
        raise ValueError(f"unknown role: {role}")
    if not secret:
        raise ValueError("jwt secret is empty")
    now = int(time.time())
    payload = {"role": role, "iat": now, "exp": now + ttl_seconds}
    token = jwt.encode(payload, secret, algorithm=ALGORITHM)
    if isinstance(token, bytes):
        return token.decode("ascii")
    return token


def read_role(token: str, secret: str) -> str:
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise TokenError("invalid token") from exc
    role = payload.get("role")
    if not isinstance(role, str) or role not in ROLE_ACCESS:
        raise TokenError("invalid role")
    return role
