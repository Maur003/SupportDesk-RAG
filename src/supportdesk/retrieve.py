"""Exact cosine search. The access filter is SQL, not a Python post-filter."""

from sqlalchemy import any_, bindparam, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Session

from supportdesk.embed_text import embedding_input
from supportdesk.embedder import BgeEmbedder
from supportdesk.models import Chunk, access_level_enum

RETRIEVAL_K = 5


def embed_missing(session: Session, embedder: BgeEmbedder) -> int:
    rows = session.scalars(
        select(Chunk)
        .where((Chunk.embedding.is_(None)) | (Chunk.embedding_model != embedder.model_id))
        .order_by(Chunk.id)
    ).all()
    if not rows:
        return 0
    texts = [embedding_input(row.title, row.heading_path, row.text) for row in rows]
    vectors = embedder.embed_passages(texts)
    for row, vector in zip(rows, vectors, strict=True):
        row.embedding = vector
        row.embedding_model = embedder.model_id
    return len(rows)


def search_chunks(
    session: Session,
    query_vector: list[float],
    allowed: tuple[str, ...],
    k: int = RETRIEVAL_K,
) -> list[tuple[str, float]]:
    """Return up to k (chunk id, cosine distance) pairs the role may see.

    `access_level = ANY(:allowed)` is the filter. Rows outside that list are
    not fetched and then dropped.
    """

    if k < 1:
        raise ValueError("k must be positive")
    distance = Chunk.embedding.cosine_distance(query_vector)
    allowed_param = bindparam("allowed", type_=ARRAY(access_level_enum))
    statement = (
        select(Chunk.id, distance.label("distance"))
        .where(Chunk.embedding.is_not(None))
        .where(Chunk.access_level == any_(allowed_param))
        .order_by(distance)
        .limit(k)
    )
    rows = session.execute(statement, {"allowed": list(allowed)}).all()
    return [(str(row.id), float(row.distance)) for row in rows]
