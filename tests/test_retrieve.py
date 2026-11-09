from sqlalchemy import any_, bindparam, select
from sqlalchemy.dialects.postgresql import ARRAY

from supportdesk.models import Chunk, Document, access_level_enum
from supportdesk.retrieve import (
    fuse_rankings,
    hybrid_search,
    keyword_search,
    keyword_statement,
    rerank_order,
    search,
    search_chunks,
)


class _LengthReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(len(passage)) for passage in passages]


def _vector(first: float, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * 382)]


def test_access_filter_is_sql_any(db_session) -> None:
    db_session.add(
        Document(
            id="public-doc",
            title="Public page",
            source_path="public/public-doc.md",
            access_level="public",
            synthetic=True,
            content_hash="a" * 64,
        )
    )
    db_session.add(
        Document(
            id="finance-doc",
            title="Finance page",
            source_path="finance/finance-doc.md",
            access_level="finance",
            synthetic=True,
            content_hash="b" * 64,
        )
    )
    db_session.flush()
    shared = _vector(1.0)
    db_session.add(
        Chunk(
            id="public-doc#0",
            doc_id="public-doc",
            ordinal=0,
            title="Public page",
            text="A household can read this page.",
            heading_path="Public page",
            access_level="public",
            content_hash="c" * 64,
            token_count=4,
            embedding=shared,
            embedding_model="test",
        )
    )
    db_session.add(
        Chunk(
            id="finance-doc#0",
            doc_id="finance-doc",
            ordinal=0,
            title="Finance page",
            text="FINANCE_ONLY_FACT",
            heading_path="Finance page",
            access_level="finance",
            content_hash="d" * 64,
            token_count=2,
            embedding=shared,
            embedding_model="test",
        )
    )
    db_session.commit()

    hits = search_chunks(db_session, shared, ("public",), k=5)
    assert [chunk_id for chunk_id, _distance in hits] == ["public-doc#0"]

    distance = Chunk.embedding.cosine_distance(shared)
    statement = (
        select(Chunk.id)
        .where(Chunk.embedding.is_not(None))
        .where(Chunk.access_level == any_(bindparam("allowed", type_=ARRAY(access_level_enum))))
        .order_by(distance)
        .limit(5)
    )
    sql = str(statement.compile(compile_kwargs={"literal_binds": False}))
    assert "ANY" in sql


def test_rerank_order_uses_the_logit_and_breaks_ties_on_id() -> None:
    assert rerank_order(["b", "a", "c"], [0.1, 0.1, 2.0], k=2) == ["c", "a"]


def test_rerank_flag_keeps_fusion_or_reorders_it(db_session) -> None:
    db_session.add(
        Document(
            id="public-doc",
            title="Public page",
            source_path="public/public-doc.md",
            access_level="public",
            synthetic=True,
            content_hash="a" * 64,
        )
    )
    db_session.add(
        Document(
            id="finance-doc",
            title="Finance page",
            source_path="finance/finance-doc.md",
            access_level="finance",
            synthetic=True,
            content_hash="b" * 64,
        )
    )
    db_session.flush()
    db_session.add(
        Chunk(
            id="public-doc#0",
            doc_id="public-doc",
            ordinal=0,
            title="Public page",
            text="alpha",
            heading_path="Public page",
            access_level="public",
            content_hash="c" * 64,
            token_count=1,
            embedding=_vector(1.0),
            embedding_model="test",
        )
    )
    db_session.add(
        Chunk(
            id="public-doc#1",
            doc_id="public-doc",
            ordinal=1,
            title="Public page",
            text="beta " * 40,
            heading_path="Public page",
            access_level="public",
            content_hash="e" * 64,
            token_count=40,
            embedding=_vector(0.0, 1.0),
            embedding_model="test",
        )
    )
    db_session.add(
        Chunk(
            id="finance-doc#0",
            doc_id="finance-doc",
            ordinal=0,
            title="Finance page",
            text="beta " * 80,
            heading_path="Finance page",
            access_level="finance",
            content_hash="d" * 64,
            token_count=80,
            embedding=_vector(1.0),
            embedding_model="test",
        )
    )
    db_session.commit()

    allowed = ("public",)
    query_vector = _vector(1.0)
    fused = search(
        db_session,
        "alpha",
        query_vector,
        allowed,
        _LengthReranker(),
        rerank_enabled=False,
    )
    reranked = search(
        db_session,
        "alpha",
        query_vector,
        allowed,
        _LengthReranker(),
        rerank_enabled=True,
    )
    assert fused[0] == "public-doc#0"
    assert reranked[0] == "public-doc#1"
    assert "finance-doc#0" not in fused
    assert "finance-doc#0" not in reranked


def test_rrf_sums_reciprocal_ranks_and_breaks_ties_on_id() -> None:
    fused = fuse_rankings([["a", "b"], ["b", "c"]], rrf_k=60)
    scores = dict(fused)
    assert scores["a"] == 1.0 / 61
    assert scores["b"] == 1.0 / 61 + 1.0 / 62
    assert scores["c"] == 1.0 / 62
    assert [chunk_id for chunk_id, _score in fused] == ["b", "a", "c"]

    tied = fuse_rankings([["m", "z"], ["z", "m"]], rrf_k=60)
    assert [chunk_id for chunk_id, _score in tied] == ["m", "z"]


def test_keyword_and_hybrid_keep_the_access_filter(db_session) -> None:
    db_session.add(
        Document(
            id="public-doc",
            title="Public page",
            source_path="public/public-doc.md",
            access_level="public",
            synthetic=True,
            content_hash="a" * 64,
        )
    )
    db_session.add(
        Document(
            id="finance-doc",
            title="Finance page",
            source_path="finance/finance-doc.md",
            access_level="finance",
            synthetic=True,
            content_hash="b" * 64,
        )
    )
    db_session.flush()
    db_session.add(
        Chunk(
            id="public-doc#0",
            doc_id="public-doc",
            ordinal=0,
            title="Public page",
            text="A household can read this page.",
            heading_path="Public page",
            access_level="public",
            content_hash="c" * 64,
            token_count=4,
            embedding=_vector(1.0),
            embedding_model="test",
        )
    )
    db_session.add(
        Chunk(
            id="public-doc#1",
            doc_id="public-doc",
            ordinal=1,
            title="Public page",
            text="visibleword is on the household page.",
            heading_path="Public page",
            access_level="public",
            content_hash="e" * 64,
            token_count=5,
            embedding=_vector(0.0, 1.0),
            embedding_model="test",
        )
    )
    db_session.add(
        Chunk(
            id="finance-doc#0",
            doc_id="finance-doc",
            ordinal=0,
            title="Finance page",
            text="visibleword is on the finance page.",
            heading_path="Finance page",
            access_level="finance",
            content_hash="d" * 64,
            token_count=5,
            embedding=_vector(1.0),
            embedding_model="test",
        )
    )
    db_session.commit()

    allowed = ("public",)
    keyword_hits = keyword_search(db_session, "visibleword", allowed, k=5)
    assert [chunk_id for chunk_id, _rank in keyword_hits] == ["public-doc#1"]

    hybrid_hits = hybrid_search(db_session, "visibleword", _vector(1.0), allowed, k=5)
    hybrid_ids = [chunk_id for chunk_id, _score in hybrid_hits]
    assert hybrid_ids[0] == "public-doc#1"
    assert "finance-doc#0" not in hybrid_ids

    sql = str(keyword_statement(5).compile(compile_kwargs={"literal_binds": False}))
    assert "ANY" in sql
    assert "@@" in sql
    assert "ts_rank_cd" in sql
    assert "websearch_to_tsquery" in sql
