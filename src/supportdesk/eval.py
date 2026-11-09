"""Baseline vector retrieval scores. No generator and no Gemini call."""

import json
import random
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from supportdesk.access import allowed_levels
from supportdesk.config import get_settings
from supportdesk.db import make_engine, make_session_factory
from supportdesk.embedder import QUERY_INSTRUCTION, BgeEmbedder
from supportdesk.eval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from supportdesk.models import Chunk
from supportdesk.retrieve import RETRIEVAL_K, search_chunks

GOLD_PATH = Path("data/gold/questions.json")
REPORT_PATH = Path("reports/retrieval.json")
RANDOM_SEED = 0
# Guard, not the reported result. The measured answerable recall@5 on this
# corpus was 0.6310. About five single-support misses still clear 0.50.
ANSWERABLE_RECALL_FLOOR = 0.50


def main() -> None:
    settings = get_settings()
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    questions = [question for question in gold["questions"] if question["status"] != "unfilled"]
    embedder = BgeEmbedder(settings.embedding_model, settings.embedding_revision)
    engine = make_engine(settings.database_url)
    factory = make_session_factory(engine)
    with factory() as session:
        report = evaluate(session, embedder, questions, k=RETRIEVAL_K)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render(report))
    enforce_floors(report)


def enforce_floors(report: dict) -> None:
    tables = {table["group"]: table for table in report["tables"]}
    recall = tables["answerable"]["vector"]["recall_at_5"]
    if recall < ANSWERABLE_RECALL_FLOOR:
        raise SystemExit(
            f"answerable recall@5 {recall:.4f} is below the guard {ANSWERABLE_RECALL_FLOOR:.2f}"
        )
    for group in ("permission_trap", "unanswerable"):
        if tables[group]["vector"] > 0:
            raise SystemExit(f"{group} vector rate is above 0")


def evaluate(
    session: Session,
    embedder: BgeEmbedder,
    questions: list[dict],
    *,
    k: int,
) -> dict:
    embedded = session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
    )
    if not embedded:
        raise SystemExit("no embeddings in the database; run make ingest")
    catalog = {
        row.id: (row.access_level, row.text)
        for row in session.scalars(select(Chunk).where(Chunk.embedding.is_not(None)))
    }
    rng = random.Random(RANDOM_SEED)
    scored: list[dict] = []
    for question in questions:
        role = question["user_role"]
        allowed = allowed_levels(role)
        # Sort so seed 0 does not follow Postgres row order.
        pool = sorted(chunk_id for chunk_id, (level, _text) in catalog.items() if level in allowed)
        query_vector = embedder.embed_query(question["question"])
        retrieved = [
            chunk_id for chunk_id, _distance in search_chunks(session, query_vector, allowed, k)
        ]
        random_ids = rng.sample(pool, k) if len(pool) >= k else list(pool)
        relevant = {support["chunk_id"] for support in question["supports"]}
        kind = question["type"]
        if kind == "answerable":
            vector_metrics = _rank_metrics(retrieved, relevant, k)
            random_metrics = _rank_metrics(random_ids, relevant, k)
        elif kind == "permission_trap":
            vector_metrics = {"leak": _leaked(retrieved, relevant, catalog, question)}
            random_metrics = {"leak": _leaked(random_ids, relevant, catalog, question)}
        elif kind == "unanswerable":
            vector_metrics = {"absent_hit": _absent_hit(retrieved, catalog, question)}
            random_metrics = {"absent_hit": _absent_hit(random_ids, catalog, question)}
        else:
            raise ValueError(f"unknown question type: {kind}")
        scored.append(
            {
                "id": question["id"],
                "type": kind,
                "role": role,
                "wording": question.get("wording", "lexical"),
                "vector": vector_metrics,
                "random": random_metrics,
            }
        )
    corpus_size = len(catalog)
    return {
        "corpus_size": corpus_size,
        "k": k,
        "random_seed": RANDOM_SEED,
        "embedding_model": embedder.model_id,
        "query_instruction": QUERY_INSTRUCTION,
        "passage_input": "title, heading_path, text",
        "tables": _tables(scored, corpus_size),
        "questions": scored,
    }


def _rank_metrics(retrieved: list[str], relevant: set[str], k: int) -> dict[str, float]:
    return {
        "recall_at_5": recall_at_k(retrieved, relevant, k),
        "precision_at_5": precision_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        "ndcg_at_5": ndcg_at_k(retrieved, relevant, k),
    }


def _leaked(retrieved: list[str], forbidden: set[str], catalog: dict, question: dict) -> float:
    if set(retrieved) & forbidden:
        return 1.0
    facts = [support["fact"] for support in question["supports"]]
    for chunk_id in retrieved:
        text = catalog[chunk_id][1]
        if any(fact in text for fact in facts):
            return 1.0
    return 0.0


def _absent_hit(retrieved: list[str], catalog: dict, question: dict) -> float:
    blob = "\n".join(catalog[chunk_id][1] for chunk_id in retrieved).lower()
    for term in question["absent_terms"]:
        if term.lower() in blob:
            return 1.0
    return 0.0


def _tables(scored: list[dict], corpus_size: int) -> list[dict]:
    answerable = [row for row in scored if row["type"] == "answerable"]
    groups = [
        ("answerable", answerable),
        ("lexical", [row for row in answerable if row["wording"] != "paraphrase"]),
        ("paraphrase", [row for row in answerable if row["wording"] == "paraphrase"]),
        ("customer", [row for row in answerable if row["role"] == "customer"]),
        ("agent", [row for row in answerable if row["role"] == "agent"]),
        ("finance", [row for row in answerable if row["role"] == "finance"]),
        ("permission_trap", [row for row in scored if row["type"] == "permission_trap"]),
        ("unanswerable", [row for row in scored if row["type"] == "unanswerable"]),
    ]
    tables = []
    for name, rows in groups:
        if not rows:
            continue
        tables.append(_summarize(name, rows, corpus_size))
    return tables


def _summarize(name: str, rows: list[dict], corpus_size: int) -> dict:
    if rows[0]["type"] == "answerable":
        wins, losses, ties = _compare(rows, "recall_at_5", higher_is_better=True)
        return {
            "group": name,
            "n": len(rows),
            "corpus_size": corpus_size,
            "vector": _mean_metrics(rows, "vector"),
            "random": _mean_metrics(rows, "random"),
            "recall_at_5_wins": wins,
            "recall_at_5_losses": losses,
            "recall_at_5_ties": ties,
        }
    key = "leak" if rows[0]["type"] == "permission_trap" else "absent_hit"
    wins, losses, ties = _compare(rows, key, higher_is_better=False)
    return {
        "group": name,
        "n": len(rows),
        "corpus_size": corpus_size,
        "metric": key,
        "vector": _mean_flag(rows, "vector", key),
        "random": _mean_flag(rows, "random", key),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def _compare(rows: list[dict], key: str, *, higher_is_better: bool) -> tuple[int, int, int]:
    wins = losses = ties = 0
    for row in rows:
        vector = row["vector"][key]
        baseline = row["random"][key]
        if vector == baseline:
            ties += 1
        elif (vector > baseline) == higher_is_better:
            wins += 1
        else:
            losses += 1
    return wins, losses, ties


def _mean_metrics(rows: list[dict], side: str) -> dict[str, float]:
    keys = ("recall_at_5", "precision_at_5", "mrr", "ndcg_at_5")
    return {key: sum(row[side][key] for row in rows) / len(rows) for key in keys}


def _mean_flag(rows: list[dict], side: str, key: str) -> float:
    return sum(row[side][key] for row in rows) / len(rows)


def render(report: dict) -> str:
    lines = [
        f"corpus_size: {report['corpus_size']}",
        f"k: {report['k']}",
        f"random_seed: {report['random_seed']}",
        f"embedding_model: {report['embedding_model']}",
        f"query_instruction: {report['query_instruction']}",
        f"passage_input: {report['passage_input']}",
        "",
        "answerable groups compare vector retrieval with a random draw",
        "from the chunks that role may see.",
        "wins, losses, and ties are per question on recall@5. recall@20 is not reported.",
        "",
    ]
    for table in report["tables"]:
        if "recall_at_5_wins" in table:
            lines.extend(_render_rank(table))
        else:
            lines.extend(_render_flag(table))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_rank(table: dict) -> list[str]:
    header = (
        f"{table['group']} n={table['n']} corpus_size={table['corpus_size']} "
        f"recall@5 wins={table['recall_at_5_wins']} "
        f"losses={table['recall_at_5_losses']} ties={table['recall_at_5_ties']}"
    )
    columns = ("recall_at_5", "precision_at_5", "mrr", "ndcg_at_5")
    return [
        header,
        "method     recall@5  precision@5  MRR     nDCG@5",
        _rank_row("vector", table["vector"], columns),
        _rank_row("random", table["random"], columns),
    ]


def _rank_row(name: str, metrics: dict, columns: tuple[str, ...]) -> str:
    cells = "  ".join(f"{metrics[column]:.4f}" for column in columns)
    return f"{name:<10} {cells}"


def _render_flag(table: dict) -> list[str]:
    return [
        (
            f"{table['group']} n={table['n']} corpus_size={table['corpus_size']} "
            f"metric={table['metric']} wins={table['wins']} "
            f"losses={table['losses']} ties={table['ties']}"
        ),
        f"vector {table['vector']:.4f}",
        f"random {table['random']:.4f}",
    ]


if __name__ == "__main__":
    main()
