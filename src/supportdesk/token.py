"""Sign a local role token. This is not a login endpoint."""

import sys

from supportdesk.access import ROLE_ACCESS
from supportdesk.auth import issue_token
from supportdesk.config import get_settings


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    roles = ", ".join(ROLE_ACCESS)
    if len(args) != 1 or args[0] not in ROLE_ACCESS:
        raise SystemExit(f"usage: python -m supportdesk.token <{roles}>")
    print(issue_token(args[0], get_settings().jwt_secret))


if __name__ == "__main__":
    main()
