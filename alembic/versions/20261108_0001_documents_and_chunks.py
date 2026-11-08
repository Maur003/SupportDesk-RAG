"""Create documents and chunks.

Revision ID: 20261108_0001
Revises:
Create Date: 2026-11-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

revision: str = "20261108_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

access_level = postgresql.ENUM(
    "public",
    "agent",
    "finance",
    name="access_level",
    create_type=False,
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    access_level.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("source_path", sa.String(), nullable=False),
        sa.Column("access_level", access_level, nullable=False),
        sa.Column("synthetic", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "doc_id", sa.String(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("heading_path", sa.Text(), nullable=False),
        sa.Column("access_level", access_level, nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", VECTOR(384), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('english', coalesce(heading_path, '') || ' ' || text)",
                persisted=True,
            ),
        ),
        sa.UniqueConstraint("doc_id", "ordinal", name="uq_chunks_doc_ordinal"),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("ix_chunks_access_level", "chunks", ["access_level"])
    op.create_index(
        "ix_chunks_search_vector",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_search_vector", table_name="chunks")
    op.drop_index("ix_chunks_access_level", table_name="chunks")
    op.drop_index("ix_chunks_doc_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.execute("DROP TYPE IF EXISTS access_level")
