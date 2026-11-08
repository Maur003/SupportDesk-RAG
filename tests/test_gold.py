import json
from pathlib import Path

from supportdesk.access import allowed_levels
from supportdesk.config import Settings
from supportdesk.ingest.loaders import load_corpus
from supportdesk.ingest.prepare import prepare_documents
from supportdesk.ingest.tokenize import HuggingFaceCodec
from supportdesk.manifest import MANIFEST_PATH, build_manifest_payload

GOLD_PATH = Path("data/gold/questions.json")


def _chunks():
    settings = Settings(
        generator_model="gemini-3.8-flash",
        judge_model="gemini-2.5-flash",
        judge_temperature=0,
        chunk_token_budget=400,
        chunk_overlap_tokens=60,
        embedding_dimensions=384,
    )
    documents = load_corpus(Path(settings.corpus_dir))
    codec = HuggingFaceCodec(settings.embedding_model)
    prepared = prepare_documents(
        documents,
        codec=codec,
        budget=settings.chunk_token_budget,
        overlap=settings.chunk_overlap_tokens,
    )
    chunks = {}
    for document in prepared:
        for chunk in document.chunks:
            assert chunk.token_count <= settings.chunk_token_budget
            chunks[chunk.id] = (document.access_level, chunk.heading_path, chunk.text)
    return chunks


def test_manifest_matches_a_fresh_chunking() -> None:
    settings = Settings(
        generator_model="gemini-3.8-flash",
        judge_model="gemini-2.5-flash",
        judge_temperature=0,
    )
    fresh = build_manifest_payload(
        Path(settings.corpus_dir),
        settings.embedding_model,
        settings.chunk_token_budget,
        settings.chunk_overlap_tokens,
    )
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert committed == fresh


def test_gold_set_shape_and_spot_check() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    assert gold["synthetic"] is True
    assert gold["company"] == "Northline Telecom"
    questions = gold["questions"]
    by_author = {}
    for question in questions:
        by_author.setdefault((question["author"], question["type"]), []).append(question)
    assert len(by_author[("user", "answerable")]) == 10
    assert len(by_author[("llm", "answerable")]) == 20
    assert len(by_author[("llm", "unanswerable")]) == 5
    assert len(by_author[("llm", "permission_trap")]) == 5
    assert gold["provenance"]["answerable_author_reserved"] == 10
    assert gold["provenance"]["answerable_llm_drafted"] == 20

    chunks = _chunks()
    blob = "\n".join(heading + "\n" + text for _, heading, text in chunks.values()).lower()
    multi = 0
    later = 0
    for question in questions:
        assert question["synthetic"] is True
        supports = question["supports"]
        if question["author"] == "user" and question["status"] == "unfilled":
            assert supports == []
            assert question["question"] in (None, "")
            continue
        if question["type"] == "unanswerable":
            assert supports == []
            for term in question["absent_terms"]:
                assert term.lower() not in blob
            continue
        assert supports
        if len(supports) > 1:
            multi += 1
        for support in supports:
            access_level, _heading, text = chunks[support["chunk_id"]]
            assert support["fact"] in text
            ordinal = int(support["chunk_id"].rsplit("#", 1)[1])
            if ordinal > 0:
                later += 1
            if question["type"] == "answerable":
                assert access_level in allowed_levels(question["user_role"])
            else:
                assert question["type"] == "permission_trap"
                assert access_level not in allowed_levels(question["user_role"])
    assert multi >= 1
    assert later >= 1
