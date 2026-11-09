import json
import re
from collections import Counter
from pathlib import Path

from supportdesk.access import allowed_levels
from supportdesk.config import Settings
from supportdesk.embed_text import embedding_input
from supportdesk.ingest.loaders import load_corpus
from supportdesk.ingest.prepare import prepare_documents
from supportdesk.ingest.tokenize import HuggingFaceCodec
from supportdesk.manifest import MANIFEST_PATH, build_manifest_payload

# Function words. The overlap check is about content words shared with the chunk.
_STOP = {
    "about",
    "above",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "does",
    "during",
    "each",
    "from",
    "have",
    "into",
    "must",
    "once",
    "only",
    "onto",
    "over",
    "past",
    "rather",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "under",
    "upon",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "with",
    "without",
    "would",
    "your",
}

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
    codec = HuggingFaceCodec(settings.embedding_model, settings.embedding_revision)
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
            embedded = embedding_input(document.title, chunk.heading_path, chunk.text)
            assert len(codec.encode(embedded)) + 2 <= 512
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
        settings.embedding_revision,
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
    assert len(by_author[("llm", "answerable")]) == 42
    assert len(by_author[("llm", "unanswerable")]) == 5
    assert len(by_author[("llm", "permission_trap")]) == 5
    assert gold["provenance"]["answerable_author_reserved"] == 10
    assert gold["provenance"]["answerable_llm_drafted"] == 42
    assert gold["provenance"]["answerable_llm_drafted_paraphrase"] == 22

    chunks = _chunks()
    assert len(chunks) >= 250
    roles: Counter[str] = Counter()
    paraphrases = 0
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
        if question.get("wording") == "paraphrase":
            paraphrases += 1
            chunk_text = "\n".join(chunks[support["chunk_id"]][2] for support in supports)
            overlap = _content_words(question["question"]) & _content_words(chunk_text)
            question_words = _content_words(question["question"])
            assert overlap == set(), overlap
            assert question_words
        if question["author"] == "llm" and question["type"] == "answerable":
            roles[question["user_role"]] += 1
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
    assert paraphrases == 22
    assert roles["customer"] >= 8
    assert roles["agent"] >= 8
    assert roles["finance"] >= 8


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 3 and word not in _STOP and not word.isdigit()
    }
