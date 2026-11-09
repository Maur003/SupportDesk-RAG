"""Rerank, hybrid, vector, and random retrieval scores. No Gemini call."""

import json
import random
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from supportdesk.access import allowed_levels
from supportdesk.config import get_settings
from supportdesk.db import make_engine, make_session_factory
from supportdesk.embed_text import embedding_input
from supportdesk.embedder import QUERY_INSTRUCTION, BgeEmbedder
from supportdesk.eval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from supportdesk.models import Chunk
from supportdesk.reranker import MiniLMReranker
from supportdesk.retrieve import (
    CANDIDATE_K,
    RETRIEVAL_K,
    RRF_K,
    candidate_lists,
    fuse_rankings,
    keyword_search,
    rerank_order,
    search_chunks,
)

GOLD_PATH = Path("data/gold/questions.json")
REPORT_PATH = Path("reports/retrieval.json")
RANDOM_SEED = 0
# Guards, not the reported result. The 2026-11-08 OR run measured answerable
# recall@5 at 0.6310 vector, 0.4524 hybrid, and 0.7024 rerank. AND hybrid was
# 0.6310. The hybrid guard sits under the OR number. The rerank guard sits
# under 0.7024.
VECTOR_RECALL_FLOOR = 0.50
HYBRID_RECALL_FLOOR = 0.40
RERANK_RECALL_FLOOR = 0.55


def main() -> None:
    settings = get_settings()
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    questions = [question for question in gold["questions"] if question["status"] != "unfilled"]
    embedder = BgeEmbedder(settings.embedding_model, settings.embedding_revision)
    reranker = MiniLMReranker(settings.rerank_model, settings.rerank_revision)
    engine = make_engine(settings.database_url)
    factory = make_session_factory(engine)
    with factory() as session:
        report = evaluate(session, embedder, reranker, questions, k=RETRIEVAL_K)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render(report))
    enforce_floors(report)


def enforce_floors(report: dict) -> None:
    tables = {table["group"]: table for table in report["tables"]}
    vector_recall = tables["answerable"]["vector"]["recall_at_5"]
    if vector_recall < VECTOR_RECALL_FLOOR:
        raise SystemExit(
            f"answerable vector recall@5 {vector_recall:.4f} "
            f"is below the guard {VECTOR_RECALL_FLOOR:.2f}"
        )
    hybrid_recall = tables["answerable"]["hybrid"]["recall_at_5"]
    if hybrid_recall < HYBRID_RECALL_FLOOR:
        raise SystemExit(
            f"answerable hybrid recall@5 {hybrid_recall:.4f} "
            f"is below the guard {HYBRID_RECALL_FLOOR:.2f}"
        )
    rerank_recall = tables["answerable"]["rerank"]["recall_at_5"]
    if rerank_recall < RERANK_RECALL_FLOOR:
        raise SystemExit(
            f"answerable rerank recall@5 {rerank_recall:.4f} "
            f"is below the guard {RERANK_RECALL_FLOOR:.2f}"
        )
    for group in ("permission_trap", "unanswerable"):
        for side in ("vector", "hybrid", "rerank"):
            if tables[group][side] > 0:
                raise SystemExit(f"{group} {side} rate is above 0")


def evaluate(
    session: Session,
    embedder: BgeEmbedder,
    reranker: MiniLMReranker,
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
        row.id: (row.access_level, row.text, row.title, row.heading_path)
        for row in session.scalars(select(Chunk).where(Chunk.embedding.is_not(None)))
    }
    rng = random.Random(RANDOM_SEED)
    scored: list[dict] = []
    for question in questions:
        role = question["user_role"]
        allowed = allowed_levels(role)
        # Sort so seed 0 does not follow Postgres row order.
        pool = sorted(chunk_id for chunk_id, (level, *_rest) in catalog.items() if level in allowed)
        query_vector = embedder.embed_query(question["question"])
        retrieved = [
            chunk_id for chunk_id, _distance in search_chunks(session, query_vector, allowed, k)
        ]
        random_ids = rng.sample(pool, k) if len(pool) >= k else list(pool)
        vector_ids, keyword_ids, fused_pool = candidate_lists(
            session, question["question"], query_vector, allowed, CANDIDATE_K
        )
        fused = fused_pool[:k]
        passages = [_passage(catalog, chunk_id) for chunk_id in fused_pool]
        reranked = rerank_order(fused_pool, reranker.score(question["question"], passages), k)
        and_keyword_ids = [
            chunk_id
            for chunk_id, _rank in keyword_search(
                session, question["question"], allowed, CANDIDATE_K, mode="and"
            )
        ]
        and_pool = [chunk_id for chunk_id, _score in fuse_rankings([vector_ids, and_keyword_ids])][
            :CANDIDATE_K
        ]
        and_fused = and_pool[:k]
        and_passages = [_passage(catalog, chunk_id) for chunk_id in and_pool]
        and_reranked = rerank_order(and_pool, reranker.score(question["question"], and_passages), k)
        relevant = {support["chunk_id"] for support in question["supports"]}
        kind = question["type"]
        if kind == "answerable":
            rerank_metrics = _rank_metrics(reranked, relevant, k)
            hybrid_metrics = _rank_metrics(fused, relevant, k)
            and_rerank_metrics = _rank_metrics(and_reranked, relevant, k)
            and_hybrid_metrics = _rank_metrics(and_fused, relevant, k)
            vector_metrics = _rank_metrics(retrieved, relevant, k)
            random_metrics = _rank_metrics(random_ids, relevant, k)
        elif kind == "permission_trap":
            rerank_metrics = {"leak": _leaked(reranked, relevant, catalog, question)}
            hybrid_metrics = {"leak": _leaked(fused, relevant, catalog, question)}
            and_rerank_metrics = {"leak": _leaked(and_reranked, relevant, catalog, question)}
            and_hybrid_metrics = {"leak": _leaked(and_fused, relevant, catalog, question)}
            vector_metrics = {"leak": _leaked(retrieved, relevant, catalog, question)}
            random_metrics = {"leak": _leaked(random_ids, relevant, catalog, question)}
        elif kind == "unanswerable":
            rerank_metrics = {"absent_hit": _absent_hit(reranked, catalog, question)}
            hybrid_metrics = {"absent_hit": _absent_hit(fused, catalog, question)}
            and_rerank_metrics = {"absent_hit": _absent_hit(and_reranked, catalog, question)}
            and_hybrid_metrics = {"absent_hit": _absent_hit(and_fused, catalog, question)}
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
                "rerank": rerank_metrics,
                "hybrid": hybrid_metrics,
                "and_rerank": and_rerank_metrics,
                "and_hybrid": and_hybrid_metrics,
                "vector": vector_metrics,
                "random": random_metrics,
                "keyword_count": len(keyword_ids),
                "keyword_and_count": len(and_keyword_ids),
                "top5_changed": fused != retrieved,
                "rerank_changed": reranked != fused,
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
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "keyword": "to_tsquery english OR of lexemes, ts_rank_cd",
        "keyword_and": "websearch_to_tsquery english AND, ts_rank_cd",
        "rerank_model": reranker.model_id,
        "rerank_activation": "raw logit",
        "rerank_pool": "fused top candidate_k",
        "questions_scored": len(scored),
        "keyword_nonempty": sum(row["keyword_count"] > 0 for row in scored),
        "keyword_and_nonempty": sum(row["keyword_and_count"] > 0 for row in scored),
        "top5_changed": sum(row["top5_changed"] for row in scored),
        "rerank_changed": sum(row["rerank_changed"] for row in scored),
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


def _passage(catalog: dict, chunk_id: str) -> str:
    _level, text, title, heading_path = catalog[chunk_id]
    return embedding_input(title, heading_path, text)


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
        versus_hybrid = _compare(rows, "recall_at_5", "rerank", "hybrid", higher_is_better=True)
        versus_random = _compare(rows, "recall_at_5", "rerank", "random", higher_is_better=True)
        hybrid_or_vs_and = _compare(
            rows, "recall_at_5", "hybrid", "and_hybrid", higher_is_better=True
        )
        rerank_or_vs_and = _compare(
            rows, "recall_at_5", "rerank", "and_rerank", higher_is_better=True
        )
        return {
            "group": name,
            "n": len(rows),
            "corpus_size": corpus_size,
            "rerank": _mean_metrics(rows, "rerank"),
            "hybrid": _mean_metrics(rows, "hybrid"),
            "and_rerank": _mean_metrics(rows, "and_rerank"),
            "and_hybrid": _mean_metrics(rows, "and_hybrid"),
            "vector": _mean_metrics(rows, "vector"),
            "random": _mean_metrics(rows, "random"),
            "rerank_vs_hybrid": versus_hybrid,
            "rerank_vs_random": versus_random,
            "hybrid_or_vs_and": hybrid_or_vs_and,
            "rerank_or_vs_and": rerank_or_vs_and,
        }
    key = "leak" if rows[0]["type"] == "permission_trap" else "absent_hit"
    versus_hybrid = _compare(rows, key, "rerank", "hybrid", higher_is_better=False)
    versus_random = _compare(rows, key, "rerank", "random", higher_is_better=False)
    hybrid_or_vs_and = _compare(rows, key, "hybrid", "and_hybrid", higher_is_better=False)
    rerank_or_vs_and = _compare(rows, key, "rerank", "and_rerank", higher_is_better=False)
    return {
        "group": name,
        "n": len(rows),
        "corpus_size": corpus_size,
        "metric": key,
        "rerank": _mean_flag(rows, "rerank", key),
        "hybrid": _mean_flag(rows, "hybrid", key),
        "and_rerank": _mean_flag(rows, "and_rerank", key),
        "and_hybrid": _mean_flag(rows, "and_hybrid", key),
        "vector": _mean_flag(rows, "vector", key),
        "random": _mean_flag(rows, "random", key),
        "rerank_vs_hybrid": versus_hybrid,
        "rerank_vs_random": versus_random,
        "hybrid_or_vs_and": hybrid_or_vs_and,
        "rerank_or_vs_and": rerank_or_vs_and,
    }


def _compare(
    rows: list[dict],
    key: str,
    left: str,
    right: str,
    *,
    higher_is_better: bool,
) -> dict[str, int]:
    wins = losses = ties = 0
    win_ids: list[str] = []
    loss_ids: list[str] = []
    for row in rows:
        candidate = row[left][key]
        baseline = row[right][key]
        if candidate == baseline:
            ties += 1
        elif (candidate > baseline) == higher_is_better:
            wins += 1
            win_ids.append(row["id"])
        else:
            losses += 1
            loss_ids.append(row["id"])
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_ids": win_ids,
        "loss_ids": loss_ids,
    }


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
        f"candidate_k: {report['candidate_k']}",
        f"rrf_k: {report['rrf_k']}",
        f"keyword: {report['keyword']}",
        f"keyword_and: {report['keyword_and']}",
        f"rerank_model: {report['rerank_model']}",
        f"rerank_activation: {report['rerank_activation']}",
        f"rerank_pool: {report['rerank_pool']}",
        f"questions_scored: {report['questions_scored']}",
        f"keyword_nonempty: {report['keyword_nonempty']}",
        f"keyword_and_nonempty: {report['keyword_and_nonempty']}",
        f"top5_changed: {report['top5_changed']}",
        f"rerank_changed: {report['rerank_changed']}",
        "",
        "rerank scores the fused top candidate_k passages and keeps k.",
        "scores are raw logits. recall@20 is not reported.",
        "keyword OR is the ask path. keyword AND is websearch_to_tsquery, scored beside it.",
        "wins, losses, and ties are per question.",
        "",
    ]
    for table in report["tables"]:
        if "metric" in table:
            lines.extend(_render_flag(table))
        else:
            lines.extend(_render_rank(table))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_rank(table: dict) -> list[str]:
    columns = ("recall_at_5", "precision_at_5", "mrr", "ndcg_at_5")
    return [
        _group_header(table),
        _pair_line("recall@5 rerank vs hybrid", table["rerank_vs_hybrid"]),
        _pair_line("recall@5 rerank vs random", table["rerank_vs_random"]),
        _pair_line("recall@5 hybrid OR vs AND", table["hybrid_or_vs_and"]),
        _pair_line("recall@5 rerank OR vs AND", table["rerank_or_vs_and"]),
        "method      recall@5  precision@5  MRR     nDCG@5",
        _rank_row("rerank_or", table["rerank"], columns),
        _rank_row("hybrid_or", table["hybrid"], columns),
        _rank_row("rerank_and", table["and_rerank"], columns),
        _rank_row("hybrid_and", table["and_hybrid"], columns),
        _rank_row("vector", table["vector"], columns),
        _rank_row("random", table["random"], columns),
    ]


def _rank_row(name: str, metrics: dict, columns: tuple[str, ...]) -> str:
    cells = "  ".join(f"{metrics[column]:.4f}" for column in columns)
    return f"{name:<11} {cells}"


def _group_header(table: dict) -> str:
    return f"{table['group']} n={table['n']} corpus_size={table['corpus_size']}"


def _pair_line(label: str, counts: dict) -> str:
    line = f"{label} wins={counts['wins']} losses={counts['losses']} ties={counts['ties']}"
    if "OR vs AND" in label and (counts["wins"] or counts["losses"]):
        wins = ",".join(counts["win_ids"]) or "-"
        losses = ",".join(counts["loss_ids"]) or "-"
        line += f" win_ids={wins} loss_ids={losses}"
    return line


def _render_flag(table: dict) -> list[str]:
    metric = table["metric"]
    return [
        f"{_group_header(table)} metric={metric}",
        _pair_line(f"{metric} rerank vs hybrid", table["rerank_vs_hybrid"]),
        _pair_line(f"{metric} rerank vs random", table["rerank_vs_random"]),
        _pair_line(f"{metric} hybrid OR vs AND", table["hybrid_or_vs_and"]),
        _pair_line(f"{metric} rerank OR vs AND", table["rerank_or_vs_and"]),
        f"rerank_or {table['rerank']:.4f}",
        f"hybrid_or {table['hybrid']:.4f}",
        f"rerank_and {table['and_rerank']:.4f}",
        f"hybrid_and {table['and_hybrid']:.4f}",
        f"vector {table['vector']:.4f}",
        f"random {table['random']:.4f}",
    ]


if __name__ == "__main__":
    main()
