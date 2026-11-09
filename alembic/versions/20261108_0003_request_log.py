"""Store one request trace. Chunk text is not a column.

Revision ID: 20261108_0003
Revises: 20261108_0002
Create Date: 2026-11-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261108_0003"
down_revision: str | None = "20261108_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "request_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("trace_id", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("route", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("embed_ms", sa.Float(), nullable=False),
        sa.Column("search_ms", sa.Float(), nullable=False),
        sa.Column("rerank_ms", sa.Float(), nullable=False),
        sa.Column("llm_ms", sa.Float(), nullable=False),
        sa.Column("retrieved_tokens", sa.Integer(), nullable=True),
        sa.Column("llm_input_tokens", sa.Integer(), nullable=True),
        sa.Column("llm_output_tokens", sa.Integer(), nullable=True),
        sa.Column("retrieved_ids", sa.JSON(), nullable=False),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("error_class", sa.String(), nullable=True),
        sa.UniqueConstraint("trace_id", name="uq_request_log_trace_id"),
    )


def downgrade() -> None:
    op.drop_table("request_log")
