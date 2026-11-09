"""Exact response cache, then a semantic lookup in the same access scope.

The exact key hashes the normalized question, the model id, the access scope,
and the corpus version. SQL lookups repeat the scope predicate, so a finance
row is not returned for a customer key. A row older than the TTL is a miss.
Semantic lookup is off when the distance threshold is None.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from supportdesk.access import allowed_levels
from supportdesk.models import Chunk, ResponseCache


def normalize_query(text: str) -> str:
    return " ".join(text.casefold().split())


def access_scope(role: str) -> str:
    return ",".join(sorted(allowed_levels(role)))


def corpus_version(hashes: list[str]) -> str:
    """Hash of every chunk content hash. Order does not matter."""

    material = "\n".join(sorted(hashes))
    return hashlib.sha256(material.encode()).hexdigest()


def load_corpus_version(session: Session) -> str:
    hashes = list(session.scalars(select(Chunk.content_hash)))
    return corpus_version(hashes)


def exact_cache_key(query: str, scope: str, model: str, corpus: str) -> str:
    material = f"exact\n{model}\n{scope}\n{corpus}\n{normalize_query(query)}"
    return hashlib.sha256(material.encode()).hexdigest()


def cosine_distance(left: list[float], right: list[float]) -> float:
    """Cosine distance for the in-memory cache. pgvector `<=>` is the SQL form."""

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1.0 - (dot / (left_norm * right_norm))


def accept_semantic(distance: float, threshold: float | None) -> bool:
    if threshold is None:
        return False
    return distance <= threshold


def cacheable_answer(tools: tuple[str, ...] | list[str]) -> bool:
    """Store an answer only when search was used and no other tool was."""

    return bool(tools) and set(tools) == {"search_knowledge_base"}


@dataclass(frozen=True)
class CacheHit:
    kind: str
    answer: str
    steps: int


def lookup_exact(
    session: Session,
    key: str,
    scope: str,
    model: str,
    *,
    ttl_seconds: int,
    now: datetime | None = None,
) -> CacheHit | None:
    row = session.scalar(
        select(ResponseCache).where(
            ResponseCache.cache_key == key,
            ResponseCache.access_scope == scope,
            ResponseCache.model == model,
            ResponseCache.created_at >= _cutoff(now, ttl_seconds),
        )
    )
    if row is None:
        return None
    return _hit("exact", row)


def lookup_semantic(
    session: Session,
    vector: list[float],
    scope: str,
    model: str,
    corpus: str,
    threshold: float | None,
    *,
    ttl_seconds: int,
    now: datetime | None = None,
) -> CacheHit | None:
    if threshold is None:
        return None
    distance = ResponseCache.embedding.cosine_distance(vector)
    found = session.execute(
        select(ResponseCache, distance)
        .where(ResponseCache.access_scope == scope)
        .where(ResponseCache.model == model)
        .where(ResponseCache.corpus_version == corpus)
        .where(ResponseCache.embedding.is_not(None))
        .where(ResponseCache.created_at >= _cutoff(now, ttl_seconds))
        .order_by(distance)
        .limit(1)
    ).first()
    if found is None:
        return None
    row, scored = found
    if not accept_semantic(float(scored), threshold):
        return None
    return _hit("semantic", row)


def store_answer(
    session: Session,
    *,
    query: str,
    scope: str,
    model: str,
    corpus: str,
    embedding: list[float] | None,
    answer: str,
    steps: int,
) -> None:
    session.add(
        ResponseCache(
            cache_key=exact_cache_key(query, scope, model, corpus),
            access_scope=scope,
            model=model,
            corpus_version=corpus,
            query_text=normalize_query(query),
            embedding=embedding,
            response={"answer": answer, "steps": steps},
        )
    )
    try:
        session.commit()
    except IntegrityError:
        session.rollback()


def _cutoff(now: datetime | None, ttl_seconds: int) -> datetime:
    if ttl_seconds < 1:
        raise ValueError("ttl_seconds must be >= 1")
    moment = now or datetime.now(UTC)
    return moment - timedelta(seconds=ttl_seconds)


def _hit(kind: str, row: ResponseCache) -> CacheHit:
    body = row.response
    return CacheHit(kind=kind, answer=str(body["answer"]), steps=int(body.get("steps", 0)))


@dataclass
class _MemoryRow:
    key: str
    scope: str
    model: str
    corpus: str
    vector: list[float] | None


@dataclass
class ScopeCache:
    """Same hit rules as the SQL lookups, for the cost script.

    The API reads `response_cache`. This object does not replace that table.
    """

    distance_threshold: float | None
    _rows: list[_MemoryRow] = field(default_factory=list)

    def lookup(
        self,
        query: str,
        scope: str,
        model: str,
        vector: list[float] | None,
        corpus: str,
    ) -> str | None:
        key = exact_cache_key(query, scope, model, corpus)
        for row in self._rows:
            if (
                row.key == key
                and row.scope == scope
                and row.model == model
                and row.corpus == corpus
            ):
                return "exact"
        if vector is None:
            return None
        best: float | None = None
        for row in self._rows:
            if (
                row.scope != scope
                or row.model != model
                or row.corpus != corpus
                or row.vector is None
            ):
                continue
            distance = cosine_distance(vector, row.vector)
            if best is None or distance < best:
                best = distance
        if best is not None and accept_semantic(best, self.distance_threshold):
            return "semantic"
        return None

    def store(
        self,
        query: str,
        scope: str,
        model: str,
        vector: list[float] | None,
        corpus: str,
    ) -> None:
        self._rows.append(
            _MemoryRow(exact_cache_key(query, scope, model, corpus), scope, model, corpus, vector)
        )
