"""Role to access-level mapping. The SQL filter in later phases uses this tuple."""

ACCESS_LEVELS = ("public", "agent", "finance")

ROLE_ACCESS: dict[str, tuple[str, ...]] = {
    "customer": ("public",),
    "agent": ("public", "agent"),
    "finance": ("public", "agent", "finance"),
}


def allowed_levels(role: str) -> tuple[str, ...]:
    try:
        return ROLE_ACCESS[role]
    except KeyError as exc:
        raise ValueError(f"unknown role: {role}") from exc
