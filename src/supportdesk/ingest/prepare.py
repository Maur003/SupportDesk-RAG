from supportdesk.ingest.chunker import chunk_document
from supportdesk.ingest.hashing import chunk_hash, document_hash
from supportdesk.ingest.tokenize import TokenCodec
from supportdesk.ingest.types import LoadedDocument, PreparedChunk, PreparedDocument


def prepare_documents(
    documents: list[LoadedDocument],
    *,
    codec: TokenCodec,
    budget: int,
    overlap: int,
) -> list[PreparedDocument]:
    prepared: list[PreparedDocument] = []
    for document in documents:
        drafts = chunk_document(
            document.body,
            title=document.title,
            budget=budget,
            overlap=overlap,
            codec=codec,
        )
        if not drafts:
            raise ValueError(f"{document.slug} produced no chunks")
        chunks = tuple(
            PreparedChunk(
                id=f"{document.slug}#{draft.ordinal}",
                ordinal=draft.ordinal,
                text=draft.text,
                heading_path=draft.heading_path,
                token_count=draft.token_count,
                content_hash=chunk_hash(draft.heading_path, draft.text, document.access_level),
            )
            for draft in drafts
        )
        prepared.append(
            PreparedDocument(
                slug=document.slug,
                title=document.title,
                source_path=document.source_path,
                access_level=document.access_level,
                synthetic=document.synthetic,
                content_hash=document_hash(document.raw_bytes),
                chunks=chunks,
            )
        )
    return prepared
