from dataclasses import dataclass


class CorpusError(ValueError):
    """A corpus file breaks a loader rule."""


@dataclass(frozen=True)
class LoadedDocument:
    slug: str
    title: str
    access_level: str
    synthetic: bool
    source_path: str
    body: str
    raw_bytes: bytes


@dataclass(frozen=True)
class ChunkDraft:
    ordinal: int
    text: str
    heading_path: str
    token_count: int


@dataclass(frozen=True)
class PreparedChunk:
    id: str
    ordinal: int
    text: str
    heading_path: str
    token_count: int
    content_hash: str


@dataclass(frozen=True)
class PreparedDocument:
    slug: str
    title: str
    source_path: str
    access_level: str
    synthetic: bool
    content_hash: str
    chunks: tuple[PreparedChunk, ...]


@dataclass(frozen=True)
class IngestReport:
    documents_seen: int
    inserted: int
    updated: int
    unchanged: int
    deleted_chunks: int
    deleted_documents: int
    chunks_written: int

    def as_dict(self) -> dict[str, int]:
        return {
            "documents_seen": self.documents_seen,
            "inserted": self.inserted,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "deleted_chunks": self.deleted_chunks,
            "deleted_documents": self.deleted_documents,
            "chunks_written": self.chunks_written,
        }
