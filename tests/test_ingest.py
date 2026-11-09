from sqlalchemy import func, select

from supportdesk.ingest.pipeline import ingest_prepared
from supportdesk.ingest.prepare import prepare_documents
from supportdesk.ingest.tokenize import WhitespaceCodec
from supportdesk.ingest.types import LoadedDocument
from supportdesk.models import Chunk, Document


def _document(slug: str, body: str, raw: bytes) -> LoadedDocument:
    return LoadedDocument(
        slug=slug,
        title=slug,
        access_level="public",
        synthetic=True,
        source_path=f"public/{slug}.md",
        body=body,
        raw_bytes=raw,
    )


def _prepare(documents: list[LoadedDocument]):
    return prepare_documents(
        documents,
        codec=WhitespaceCodec(),
        budget=8,
        overlap=2,
    )


def test_reingest_of_the_same_bytes_writes_nothing(db_session) -> None:
    prepared = _prepare([_document("sample", "Hello Northline customers today.", b"v1")])
    first = ingest_prepared(db_session, prepared, delete_missing=False)
    db_session.commit()
    second = ingest_prepared(db_session, prepared, delete_missing=False)
    db_session.commit()

    assert first.inserted == 1
    assert first.chunks_written == 1
    assert second.unchanged == 1
    assert second.inserted == 0
    assert second.updated == 0
    assert second.chunks_written == 0
    stored = db_session.get(Chunk, "sample#0")
    document = db_session.get(Document, "sample")
    assert stored is not None
    assert document is not None
    assert document.synthetic is True
    assert stored.embedding is None
    assert stored.title == "sample"
    assert stored.search_vector is not None
    assert stored.access_level == "public"
    matched = db_session.scalar(
        select(func.count())
        .select_from(Chunk)
        .where(Chunk.search_vector.op("@@")(func.plainto_tsquery("english", "sample")))
    )
    assert matched == 1


def test_changed_bytes_replace_orphan_chunks(db_session) -> None:
    long_body = "# One\n\n" + " ".join(f"a{i}" for i in range(15))
    long_body += "\n\n# Two\n\n" + " ".join(f"b{i}" for i in range(15))
    first = _prepare([_document("sample", long_body, b"long")])
    ingest_prepared(db_session, first, delete_missing=False)
    db_session.commit()
    before = db_session.scalar(select(func.count()).select_from(Chunk))
    assert before is not None and before > 1

    second = _prepare([_document("sample", "Short replacement text.", b"short")])
    report = ingest_prepared(db_session, second, delete_missing=False)
    db_session.commit()

    assert report.updated == 1
    assert report.deleted_chunks > 0
    remaining = db_session.scalars(select(Chunk).where(Chunk.doc_id == "sample")).all()
    assert [row.id for row in remaining] == ["sample#0"]
    assert remaining[0].text == "Short replacement text."


def test_delete_missing_removes_documents_that_left_the_corpus(db_session) -> None:
    prepared = _prepare(
        [
            _document("keep", "Keep this document.", b"keep"),
            _document("drop", "Drop this document.", b"drop"),
        ]
    )
    ingest_prepared(db_session, prepared, delete_missing=False)
    db_session.commit()
    kept = [item for item in prepared if item.slug == "keep"]
    report = ingest_prepared(db_session, kept, delete_missing=True)
    db_session.commit()

    assert report.deleted_documents == 1
    assert db_session.get(Document, "drop") is None
    assert db_session.get(Document, "keep") is not None
