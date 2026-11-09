from pathlib import Path

from supportdesk.ingest.loaders import load_corpus

CORPUS = Path("data/corpus")


def test_corpus_is_synthetic_and_covers_three_levels() -> None:
    documents = load_corpus(CORPUS)
    assert len(documents) == 121
    assert {document.synthetic for document in documents} == {True}
    assert {document.access_level for document in documents} == {"public", "agent", "finance"}
    slugs = {document.slug for document in documents}
    for slug in (
        "fiber-300-plan",
        "fiber-2000-plan",
        "outage-credit-business",
        "outage-credit-business-agent",
        "refund-timing",
        "refund-posting",
    ):
        assert slug in slugs
    suffixes = {Path(document.source_path).suffix for document in documents}
    assert ".md" in suffixes
    assert ".txt" in suffixes
    assert ".pdf" in suffixes


def test_canaries_exist_in_exactly_one_file() -> None:
    counts = {
        "FINANCE_CANARY_7K2M": 0,
        "AGENT_CANARY_4QP9": 0,
        "WRITEOFF_CANARY_8HN4": 0,
        "EXECUTIVE_CANARY_2VX6": 0,
    }
    for path in CORPUS.rglob("*"):
        if not path.is_file() or path.suffix == ".json":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for canary in counts:
            if canary in text:
                counts[canary] += 1
    assert counts == {
        "FINANCE_CANARY_7K2M": 1,
        "AGENT_CANARY_4QP9": 1,
        "WRITEOFF_CANARY_8HN4": 1,
        "EXECUTIVE_CANARY_2VX6": 1,
    }
