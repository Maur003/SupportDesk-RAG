from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from supportdesk.ingest.types import IngestReport, PreparedDocument
from supportdesk.models import Chunk, Document


def ingest_prepared(
    session: Session,
    documents: list[PreparedDocument],
    *,
    delete_missing: bool,
) -> IngestReport:
    inserted = 0
    updated = 0
    unchanged = 0
    deleted_chunks = 0
    chunks_written = 0

    for document in documents:
        existing = session.get(Document, document.slug)
        if existing is not None and existing.content_hash == document.content_hash:
            rows = session.scalars(
                select(Chunk).where(Chunk.doc_id == document.slug).order_by(Chunk.ordinal)
            ).all()
            old_hashes = [row.content_hash for row in rows]
            new_hashes = [chunk.content_hash for chunk in document.chunks]
            if old_hashes == new_hashes:
                if existing.source_path != document.source_path:
                    existing.source_path = document.source_path
                unchanged += 1
                continue

        if existing is None:
            session.add(
                Document(
                    id=document.slug,
                    title=document.title,
                    source_path=document.source_path,
                    access_level=document.access_level,
                    synthetic=document.synthetic,
                    content_hash=document.content_hash,
                )
            )
            session.flush()
            inserted += 1
            existing_chunks = []
        else:
            existing.title = document.title
            existing.source_path = document.source_path
            existing.access_level = document.access_level
            existing.synthetic = document.synthetic
            existing.content_hash = document.content_hash
            existing.updated_at = datetime.now(UTC)
            updated += 1
            existing_chunks = session.scalars(
                select(Chunk).where(Chunk.doc_id == document.slug)
            ).all()

        by_id = {row.id: row for row in existing_chunks}
        seen: set[str] = set()
        for chunk in document.chunks:
            seen.add(chunk.id)
            row = by_id.get(chunk.id)
            if row is not None and row.content_hash == chunk.content_hash:
                continue
            if row is None:
                session.add(
                    Chunk(
                        id=chunk.id,
                        doc_id=document.slug,
                        ordinal=chunk.ordinal,
                        text=chunk.text,
                        heading_path=chunk.heading_path,
                        access_level=document.access_level,
                        content_hash=chunk.content_hash,
                        token_count=chunk.token_count,
                    )
                )
            else:
                row.ordinal = chunk.ordinal
                row.text = chunk.text
                row.heading_path = chunk.heading_path
                row.access_level = document.access_level
                row.content_hash = chunk.content_hash
                row.token_count = chunk.token_count
                row.embedding = None
                row.embedding_model = None
            chunks_written += 1

        for row in existing_chunks:
            if row.id not in seen:
                session.delete(row)
                deleted_chunks += 1

    deleted_documents = 0
    if delete_missing:
        session.flush()
        keep = {document.slug for document in documents}
        stale_stmt = select(Document)
        if keep:
            stale_stmt = stale_stmt.where(Document.id.not_in(sorted(keep)))
        stale = session.scalars(stale_stmt).all()
        for row in stale:
            session.delete(row)
            deleted_documents += 1

    return IngestReport(
        documents_seen=len(documents),
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        deleted_chunks=deleted_chunks,
        deleted_documents=deleted_documents,
        chunks_written=chunks_written,
    )
