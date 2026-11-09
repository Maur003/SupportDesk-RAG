"""Store the document title on each chunk and include it in the tsvector.

Revision ID: 20261108_0002
Revises: 20261108_0001
Create Date: 2026-11-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261108_0002"
down_revision: str | None = "20261108_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW = (
    "to_tsvector('english', coalesce(title, '') || ' ' || "
    "coalesce(heading_path, '') || ' ' || text)"
)
_OLD = "to_tsvector('english', coalesce(heading_path, '') || ' ' || text)"


def upgrade() -> None:
    op.add_column("chunks", sa.Column("title", sa.Text(), nullable=True))
    op.execute(
        "UPDATE chunks SET title = documents.title "
        "FROM documents WHERE chunks.doc_id = documents.id"
    )
    op.alter_column("chunks", "title", nullable=False)
    op.drop_index("ix_chunks_search_vector", table_name="chunks")
    op.drop_column("chunks", "search_vector")
    op.add_column(
        "chunks",
        sa.Column("search_vector", postgresql.TSVECTOR(), sa.Computed(_NEW, persisted=True)),
    )
    op.create_index(
        "ix_chunks_search_vector",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_search_vector", table_name="chunks")
    op.drop_column("chunks", "search_vector")
    op.add_column(
        "chunks",
        sa.Column("search_vector", postgresql.TSVECTOR(), sa.Computed(_OLD, persisted=True)),
    )
    op.create_index(
        "ix_chunks_search_vector",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.drop_column("chunks", "title")
