from supportdesk.ingest.chunker import chunk_document
from supportdesk.ingest.tokenize import WhitespaceCodec


def test_short_sections_pack_into_one_chunk() -> None:
    codec = WhitespaceCodec()
    body = "# Plans\n\nFiber is available.\n\n## Limits\n\nSpeeds are published per plan."
    chunks = chunk_document(body, title="Catalog", budget=40, overlap=4, codec=codec)
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0
    assert "Plans" in chunks[0].heading_path
    assert "Limits" in chunks[0].heading_path
    assert chunks[0].token_count <= 40


def test_preamble_uses_the_document_title() -> None:
    codec = WhitespaceCodec()
    chunks = chunk_document(
        "Read this first.",
        title="Welcome",
        budget=20,
        overlap=2,
        codec=codec,
    )
    assert chunks[0].heading_path == "Welcome"
    assert chunks[0].text == "Read this first."


def test_long_text_overlaps_and_stays_inside_the_budget() -> None:
    codec = WhitespaceCodec()
    words = [f"w{i}" for i in range(30)]
    chunks = chunk_document(
        " ".join(words),
        title="Long",
        budget=10,
        overlap=3,
        codec=codec,
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 10
        assert chunk.token_count == len(codec.encode(chunk.text))
    first = chunks[0].text.split()
    second = chunks[1].text.split()
    assert first[-3:] == second[:3]


def test_sentence_periods_survive_a_split() -> None:
    codec = WhitespaceCodec()
    sentences = [f"Rule {i} stays intact here." for i in range(12)]
    chunks = chunk_document(
        " ".join(sentences),
        title="Rules",
        budget=8,
        overlap=2,
        codec=codec,
    )
    assert len(chunks) > 1
    assert all(chunk.token_count <= 8 for chunk in chunks)
    combined = " ".join(chunk.text for chunk in chunks)
    assert "Rule 0 stays intact here." in combined


def test_ordinals_increase_across_the_document() -> None:
    codec = WhitespaceCodec()
    body = "# One\n\n" + " ".join(f"a{i}" for i in range(20))
    body += "\n\n# Two\n\n" + " ".join(f"b{i}" for i in range(20))
    chunks = chunk_document(body, title="Doc", budget=10, overlap=2, codec=codec)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert len(chunks) > 2
