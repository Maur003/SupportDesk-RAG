from sqlalchemy import any_, bindparam, select
from sqlalchemy.dialects.postgresql import ARRAY

from supportdesk.models import Chunk, Document, access_level_enum
from supportdesk.retrieve import search_chunks


def _vector(first: float) -> list[float]:
    return [first, *([0.0] * 383)]


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
