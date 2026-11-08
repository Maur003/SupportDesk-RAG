from datetime import datetime

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Computed, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ENUM, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from supportdesk.config import BGE_SMALL_DIMENSIONS

access_level_enum = ENUM(
    "public",
    "agent",
    "finance",
    name="access_level",
    create_type=False,
)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    access_level: Mapped[str] = mapped_column(access_level_enum, nullable=False)
    synthetic: Mapped[bool] = mapped_column(nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("doc_id", "ordinal", name="uq_chunks_doc_ordinal"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    heading_path: Mapped[str] = mapped_column(Text, nullable=False)
    access_level: Mapped[str] = mapped_column(access_level_enum, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        VECTOR(BGE_SMALL_DIMENSIONS), nullable=True
    )
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(heading_path, '') || ' ' || text)",
            persisted=True,
        ),
        nullable=True,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
