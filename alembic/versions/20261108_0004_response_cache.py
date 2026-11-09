"""Cache one answer per exact key, and record the request cost.

Revision ID: 20261108_0004
Revises: 20261108_0003
Create Date: 2026-11-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR

revision: str = "20261108_0004"
down_revision: str | None = "20261108_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("request_log", sa.Column("model", sa.String(), nullable=True))
    op.add_column("request_log", sa.Column("cost_usd", sa.Numeric(16, 8), nullable=True))
    op.add_column("request_log", sa.Column("cache_hit", sa.String(), nullable=True))
    op.create_table(
        "response_cache",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("access_scope", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("embedding", VECTOR(384), nullable=True),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("cache_key", name="uq_response_cache_key"),
    )
    op.create_index(
        "ix_response_cache_scope_model",
        "response_cache",
        ["access_scope", "model"],
    )


def downgrade() -> None:
    op.drop_index("ix_response_cache_scope_model", table_name="response_cache")
    op.drop_table("response_cache")
    op.drop_column("request_log", "cache_hit")
    op.drop_column("request_log", "cost_usd")
    op.drop_column("request_log", "model")
