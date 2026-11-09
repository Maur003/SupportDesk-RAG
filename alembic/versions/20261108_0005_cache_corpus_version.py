"""Record the corpus version on each cached answer.

Revision ID: 20261108_0005
Revises: 20261108_0004
Create Date: 2026-11-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261108_0005"
down_revision: str | None = "20261108_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("response_cache", sa.Column("corpus_version", sa.String(length=64), nullable=True))
    op.execute("DELETE FROM response_cache")
    op.alter_column("response_cache", "corpus_version", nullable=False)
    op.create_index(
        "ix_response_cache_scope_model_version",
        "response_cache",
        ["access_scope", "model", "corpus_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_response_cache_scope_model_version", table_name="response_cache")
    op.drop_column("response_cache", "corpus_version")
