"""Vector search, keyword search, and reciprocal rank fusion.

The access filter is SQL, not a Python post-filter. Keyword search uses the
same `access_level = ANY(:allowed)` predicate as the vector query.
"""

import time
from dataclasses import dataclass

from sqlalchemy import any_, bindparam, func, literal_column, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Session

from supportdesk.embed_text import embedding_input
from supportdesk.embedder import BgeEmbedder, EmbedderError
from supportdesk.models import Chunk, access_level_enum

RETRIEVAL_K = 5
CANDIDATE_K = 20
RRF_K = 60


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


def keyword_statement(k: int):
    """SQL for keyword search. The access predicate is part of the statement."""

    query_param = bindparam("query", unique=False)
    tsquery = func.websearch_to_tsquery(literal_column("'english'"), query_param)
    rank = func.ts_rank_cd(Chunk.search_vector, tsquery)
    allowed_param = bindparam("allowed", type_=ARRAY(access_level_enum))
    return (
        select(Chunk.id, rank.label("rank"))
        .where(Chunk.embedding.is_not(None))
        .where(Chunk.search_vector.op("@@")(tsquery))
        .where(Chunk.access_level == any_(allowed_param))
        .order_by(rank.desc(), Chunk.id)
        .limit(k)
    )


def keyword_search(
    session: Session,
    query: str,
    allowed: tuple[str, ...],
    k: int = CANDIDATE_K,
) -> list[tuple[str, float]]:
    """Return up to k (chunk id, ts_rank_cd) pairs, highest rank first.

    The query is the question text. `websearch_to_tsquery` ANDs the remaining
    words after English stop-words are removed. Rank ties break on chunk id.
    """

    if k < 1:
        raise ValueError("k must be positive")
    statement = keyword_statement(k)
    rows = session.execute(statement, {"query": query, "allowed": list(allowed)}).all()
    return [(str(row.id), float(row.rank)) for row in rows]


def fuse_rankings(rankings: list[list[str]], *, rrf_k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Rank 1 is the best row in each list.

    Score is the sum of `1 / (rrf_k + rank)`. A chunk in only one list still
    scores. Equal scores break on chunk id, so the order does not depend on
    dict order.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for chunk_id in ranking:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [(chunk_id, scores[chunk_id]) for chunk_id in ordered]


def hybrid_search(
    session: Session,
    query: str,
    query_vector: list[float],
    allowed: tuple[str, ...],
    k: int = RETRIEVAL_K,
    candidate_k: int = CANDIDATE_K,
) -> list[tuple[str, float]]:
    """Fuse the top `candidate_k` vector hits and keyword hits, then keep k.

    The vector query is the embedding of `query`. The keyword query is `query`
    itself, without the embedding instruction prefix.
    """

    if k < 1:
        raise ValueError("k must be positive")
    vector_ids = [
        chunk_id
        for chunk_id, _distance in search_chunks(session, query_vector, allowed, candidate_k)
    ]
    keyword_ids = [
        chunk_id for chunk_id, _rank in keyword_search(session, query, allowed, candidate_k)
    ]
    return fuse_rankings([vector_ids, keyword_ids])[:k]


def candidate_lists(
    session: Session,
    query: str,
    query_vector: list[float],
    allowed: tuple[str, ...],
    candidate_k: int = CANDIDATE_K,
) -> tuple[list[str], list[str], list[str]]:
    """Vector ids, keyword ids, and the fused ids capped at `candidate_k`."""

    vector_ids = [
        chunk_id
        for chunk_id, _distance in search_chunks(session, query_vector, allowed, candidate_k)
    ]
    keyword_ids = [
        chunk_id for chunk_id, _rank in keyword_search(session, query, allowed, candidate_k)
    ]
    fused = [chunk_id for chunk_id, _score in fuse_rankings([vector_ids, keyword_ids])]
    return vector_ids, keyword_ids, fused[:candidate_k]


def rerank_order(chunk_ids: list[str], logits: list[float], k: int) -> list[str]:
    """Highest raw logit first. Equal logits break on chunk id."""

    if k < 1:
        raise ValueError("k must be positive")
    if len(chunk_ids) != len(logits):
        raise ValueError("logits must match the candidate list")
    ranked = sorted(
        zip(chunk_ids, logits, strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    return [chunk_id for chunk_id, _logit in ranked[:k]]


@dataclass(frozen=True)
class Retrieval:
    ids: list[str]
    scores: list[float]
    embed_seconds: float
    search_seconds: float
    rerank_seconds: float


def retrieve(
    session: Session,
    query: str,
    embedder: object,
    allowed: tuple[str, ...],
    reranker: object,
    *,
    k: int = RETRIEVAL_K,
    candidate_k: int = CANDIDATE_K,
    rerank_enabled: bool = True,
) -> list[str]:
    """Search, or keyword-only when the embedder cannot encode the query."""

    return timed_retrieve(
        session,
        query,
        embedder,
        allowed,
        reranker,
        k=k,
        candidate_k=candidate_k,
        rerank_enabled=rerank_enabled,
    ).ids


def timed_retrieve(
    session: Session,
    query: str,
    embedder: object,
    allowed: tuple[str, ...],
    reranker: object,
    *,
    k: int = RETRIEVAL_K,
    candidate_k: int = CANDIDATE_K,
    rerank_enabled: bool = True,
    clock: object = time.perf_counter,
) -> Retrieval:
    """Same ids as `retrieve`, plus the stage times and the score of each id.

    Embedder failure uses keyword rank as the score and leaves rerank at 0.
    The keyword fallback keeps `access_level = ANY(:allowed)`.
    """

    started = clock()
    try:
        query_vector = embedder.embed_query(query)
    except EmbedderError:
        embed_seconds = clock() - started
        started = clock()
        pairs = keyword_search(session, query, allowed, k)
        return Retrieval(
            ids=[chunk_id for chunk_id, _rank in pairs],
            scores=[rank for _chunk_id, rank in pairs],
            embed_seconds=embed_seconds,
            search_seconds=clock() - started,
            rerank_seconds=0.0,
        )
    embed_seconds = clock() - started
    pairs, search_seconds, rerank_seconds = _ranked(
        session,
        query,
        query_vector,
        allowed,
        reranker,
        k=k,
        candidate_k=candidate_k,
        rerank_enabled=rerank_enabled,
        clock=clock,
    )
    return Retrieval(
        ids=[chunk_id for chunk_id, _score in pairs],
        scores=[score for _chunk_id, score in pairs],
        embed_seconds=embed_seconds,
        search_seconds=search_seconds,
        rerank_seconds=rerank_seconds,
    )


def search(
    session: Session,
    query: str,
    query_vector: list[float],
    allowed: tuple[str, ...],
    reranker: object,
    *,
    k: int = RETRIEVAL_K,
    candidate_k: int = CANDIDATE_K,
    rerank_enabled: bool = True,
) -> list[str]:
    """Return k chunk ids. The flag selects reranked fusion or the RRF cutoff.

    Candidates are already limited by `access_level = ANY(:allowed)`. The
    reranker does not see chunks outside that list.
    """

    pairs, _search_seconds, _rerank_seconds = _ranked(
        session,
        query,
        query_vector,
        allowed,
        reranker,
        k=k,
        candidate_k=candidate_k,
        rerank_enabled=rerank_enabled,
        clock=time.perf_counter,
    )
    return [chunk_id for chunk_id, _score in pairs]


def _ranked(
    session: Session,
    query: str,
    query_vector: list[float],
    allowed: tuple[str, ...],
    reranker: object,
    *,
    k: int,
    candidate_k: int,
    rerank_enabled: bool,
    clock: object,
) -> tuple[list[tuple[str, float]], float, float]:
    if k < 1:
        raise ValueError("k must be positive")
    started = clock()
    vector_ids, keyword_ids, fused = candidate_lists(
        session, query, query_vector, allowed, candidate_k
    )
    search_seconds = clock() - started
    fused_scores = dict(fuse_rankings([vector_ids, keyword_ids]))
    if not rerank_enabled:
        kept = fused[:k]
        return [(chunk_id, fused_scores[chunk_id]) for chunk_id in kept], search_seconds, 0.0
    started = clock()
    rows = session.scalars(select(Chunk).where(Chunk.id.in_(fused))).all()
    by_id = {row.id: row for row in rows}
    passages = [
        embedding_input(by_id[chunk_id].title, by_id[chunk_id].heading_path, by_id[chunk_id].text)
        for chunk_id in fused
    ]
    logits = reranker.score(query, passages)
    order = rerank_order(fused, logits, k)
    by_logit = dict(zip(fused, logits, strict=True))
    pairs = [(chunk_id, by_logit[chunk_id]) for chunk_id in order]
    return pairs, search_seconds, clock() - started
