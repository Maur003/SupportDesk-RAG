def embedding_input(title: str, heading_path: str, text: str) -> str:
    """String Phase 2 embeds for one chunk.

    The stored chunk text stays the body only. Title and heading path are
    metadata, repeated here so a query that names the plan or the section
    can match. Parts are separated by newlines. Empty parts are skipped.
    """

    parts = [title.strip(), heading_path.strip(), text.strip()]
    return "\n".join(part for part in parts if part)
