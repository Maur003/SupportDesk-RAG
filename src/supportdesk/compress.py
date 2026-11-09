"""Drop duplicate and overflowing chunks before they enter the prompt.

Retrieval order is unchanged. This runs only on the ask path.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextChunk:
    id: str
    access_level: str
    text: str
    score: float
    tokens: int


def compress_context(
    chunks: list[ContextChunk],
    *,
    max_tokens: int,
    min_score: float | None,
) -> list[ContextChunk]:
    """Keep the higher score for a duplicate id, then fill the token cap.

    Chunks are considered from the highest score down. The first one that
    would pass the cap stops the fill, including a single chunk larger than
    the cap. A score below `min_score` is dropped first when that floor is set.
    """

    best: dict[str, ContextChunk] = {}
    for chunk in chunks:
        if min_score is not None and chunk.score < min_score:
            continue
        current = best.get(chunk.id)
        if current is None or chunk.score > current.score:
            best[chunk.id] = chunk
    ordered = sorted(best.values(), key=lambda chunk: (-chunk.score, chunk.id))
    kept: list[ContextChunk] = []
    used = 0
    for chunk in ordered:
        if used + chunk.tokens > max_tokens:
            break
        kept.append(chunk)
        used += chunk.tokens
    return kept
